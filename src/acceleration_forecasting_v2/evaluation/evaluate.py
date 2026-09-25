"""予測結果の評価。SPEC.md 1.4に従い、record-levelとsegment-level(2段階平均)の
両方の集計値・bootstrap信頼区間を出力する。

旧`evaluation/evaluate.py`との違い: 過去実験どうしの比較表・画像出力・
複数の実験ツリーを横断する処理は持たない(成果物の乱立を防ぐ方針)。
モデル間の比較は`build_comparison_table`に各runのsummaryを渡して行う(SPEC 1.5)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import (
    BASELINE_METRIC_NAMES, METRIC_NAMES, bootstrap_record_level, bootstrap_segment_level,
    confidence_interval, evaluate_record_level, evaluate_segment_level, segment_means,
)
from .metrics import target_metrics


def per_target_frame(dataset_dir, prediction_dir, *, split="inference", min_target_months=8):
    """レコードごとの指標表を作る。

    Returns:
        (評価対象の表, 集計前の件数dict)。有効な正解月が`min_target_months`未満のレコードは
        評価対象から除外する(学習・検証側のeligibility基準と揃える)。
    """
    dataset_dir, prediction_dir = Path(dataset_dir), Path(prediction_dir)
    split_dir = dataset_dir / split
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    targets = np.load(split_dir / "target_values.npy")
    masks = np.load(split_dir / "target_masks.npy")
    baselines = np.load(split_dir / "guide_baselines.npy")
    predictions = pd.read_csv(prediction_dir / "predictions.csv", encoding="utf-8-sig")
    lookup = {str(trend_id): index for index, trend_id in enumerate(metadata["trend_id"])}

    rows, without_valid_target = [], 0
    for trend_id, group in predictions.groupby(predictions["trend_id"].astype(str), sort=False):
        if trend_id not in lookup:
            continue
        index = lookup[trend_id]
        ordered = group.sort_values("month_index")
        metrics = target_metrics(targets[index], masks[index], ordered["prediction_median"],
                                 ordered["prediction_p10"], ordered["prediction_p90"])
        if metrics is None:
            without_valid_target += 1
            continue
        baseline = target_metrics(targets[index], masks[index], baselines[index], baselines[index], baselines[index])
        row = metadata.iloc[index]
        rows.append({
            "trend_id": trend_id, "dataset_id": row["dataset_id"],
            **{key: row[key] for key in ("direction", "bin_start_m", "anchor_date") if key in metadata.columns},
            "ensemble_mean_std": float(pd.to_numeric(ordered["prediction_std"], errors="coerce").mean()),
            "baseline_MAE": baseline["MAE"], "baseline_RMSE": baseline["RMSE"], **metrics,
        })
    all_frame = pd.DataFrame(rows)
    frame = all_frame.loc[all_frame["valid_months"] >= int(min_target_months)].copy() if len(all_frame) else all_frame
    counts = {"prediction_records": int(predictions["trend_id"].nunique()), "without_valid_target": without_valid_target,
              "evaluable_records": int(len(all_frame)), "evaluated_records": int(len(frame))}
    return frame, counts


def evaluate(dataset_dir, prediction_dir, output_dir, *, split="inference", min_target_months=8,
             bootstrap=1000, seed=42):
    """評価を実行し、per-record表・per-segment表・bootstrap結果・summary JSONを書き出す。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, counts = per_target_frame(dataset_dir, prediction_dir, split=split, min_target_months=min_target_months)
    if frame.empty:
        raise ValueError("評価対象のレコードがありません(min_target_monthsを満たすレコードが0件)。")
    names = METRIC_NAMES + BASELINE_METRIC_NAMES
    frame.to_csv(output_dir / "evaluation_per_target.csv", index=False, encoding="utf-8-sig")
    segment_means(frame, names).to_csv(output_dir / "evaluation_per_segment.csv", encoding="utf-8-sig")

    summary = {
        "record_level": evaluate_record_level(frame, names),
        "segment_level": evaluate_segment_level(frame, names),
        "counts": {**counts, "segment_count": int(frame["dataset_id"].nunique())},
        "config": {"split": split, "min_target_months": int(min_target_months),
                   "bootstrap_iterations": int(bootstrap), "seed": int(seed)},
    }
    if int(bootstrap) > 0:
        record_boot = bootstrap_record_level(frame, iterations=bootstrap, seed=seed, metric_names=names)
        segment_boot = bootstrap_segment_level(frame, iterations=bootstrap, seed=seed, metric_names=names)
        record_boot.to_csv(output_dir / "bootstrap_record_level.csv", index=False)
        segment_boot.to_csv(output_dir / "bootstrap_segment_level.csv", index=False)
        summary["record_level_ci"] = confidence_interval(record_boot)
        summary["segment_level_ci"] = confidence_interval(segment_boot)
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_comparison_table(summaries: dict) -> pd.DataFrame:
    """複数run(例: epsilon / v_prediction)のsummaryを、指標×(run, 集計単位)の表にまとめる。

    SPEC 1.5: 比較表はrecord-levelとsegment-levelの両方を並べる。
    """
    metrics = [name for name in METRIC_NAMES + BASELINE_METRIC_NAMES
               if all(name in summary["record_level"] for summary in summaries.values())]
    table = {}
    for run, summary in summaries.items():
        table[f"{run}|record_level"] = [summary["record_level"][name] for name in metrics]
        table[f"{run}|segment_level"] = [summary["segment_level"][name] for name in metrics]
    return pd.DataFrame(table, index=metrics)
