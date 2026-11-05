"""検索したguideの質の比較(拡散モデルを学習せずに、検索の設定どうしを比べる、2026-11)。

同じ起点の集合に対して、検索の設定だけが違うデータセット(`rebuild-guides`の出力)を並べ、次の3つで比べる:

1. guide_copy_MAE: guideのsoftmax平均(`guide_baselines`)をそのまま予測にしたときの誤差。
2. added_value: ridge(履歴[+D4])にguideの変化量(12か月+その平均)を足したときのMAEから、足さないridgeの
   MAEを引いた値(負ならguideが履歴・D4に無い情報を持つ)。ridgeはその設定のmodel_trainで学習する。
   拡散モデルが履歴だけで分かる以上の情報をguideから引き出せるかの上限の目安で、最も重視する指標。
3. up_AUC: 大きく上昇する件(最後の有効3か月の平均 − 現在値 ≥ UP_THRESHOLD)を、予測の上昇量でどれだけ
   見分けられるか(guide_copy・ridge・ridge+guideのそれぞれ)。

誤差は実測の月だけで測る(合成の月はD型の直線なので含めるとD4に近いguideが有利になる)。基準の設定との差は
同じ起点どうしの対応で、セグメント単位のbootstrapで95%区間を出す(`evaluation.aggregate`と同じ考え方)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.datasets.cycle_features import CYCLE_FEATURE_NAME
from .baselines import (
    RIDGE_LAMBDA, _anchor_months, _load_history, apply_weighted_ridge, current_values, fit_weighted_ridge,
    history_features,
)

UP_THRESHOLD = 0.5
AGE_BINS = (6, 9, 12, 100)  # 補修からの経過月数の区分: 6〜8 / 9〜11 / 12〜


def _guide_features(split_dir, history):
    guide_mean = np.load(split_dir / "guide_baselines.npy").astype(float)
    delta = np.nan_to_num(guide_mean - current_values(history)[:, None])
    return np.column_stack([delta, delta.mean(1)])


def split_features(split_dir, *, use_cycle_features=True, with_guide=False):
    """ridge用の特徴量(先頭列は切片)。履歴の特徴 [+ D4(cycle_features.npy)] [+ guideの変化量]。"""
    split_dir = Path(split_dir)
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    history = _load_history(split_dir)
    columns = [history_features(history, _anchor_months(metadata))]
    if use_cycle_features:
        path = split_dir / f"{CYCLE_FEATURE_NAME}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"{path} がありません(add-cycle-featuresを先に実行するか、--no-cycle-featuresを指定)。")
        columns.append(np.nan_to_num(np.load(path).astype(float)))
    if with_guide:
        columns.append(_guide_features(split_dir, history))
    return np.column_stack(columns)


def real_masks(split_dir):
    """合成の月を除いた、実測の正解の月(bool、shape (N, 12))。"""
    split_dir = Path(split_dir)
    values = np.load(split_dir / "target_values.npy").astype(float)
    masks = np.load(split_dir / "target_masks.npy") > 0
    synthetic = split_dir / "target_synthetic_masks.npy"
    if synthetic.is_file():
        masks = masks & ~(np.load(synthetic) > 0)
    return masks & np.isfinite(values)


def _auc(score, label):
    if label.sum() == 0 or (~label).sum() == 0:
        return float("nan")
    rank = pd.Series(score).rank().to_numpy()
    positives, negatives = label.sum(), (~label).sum()
    return float((rank[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _last_valid_mean(values, mask, count=3):
    """各行で、有効な月の最後のcount個の平均(有効な月がなければNaN)。"""
    result = np.full(len(values), np.nan)
    for index, (row, valid) in enumerate(zip(values, mask)):
        picked = row[valid][-count:]
        if picked.size:
            result[index] = picked.mean()
    return result


def _record_mae(prediction, targets, mask):
    counts = mask.sum(1)
    errors = np.where(mask, np.abs(prediction - targets), 0.0).sum(1)
    return np.where(counts > 0, errors / np.maximum(counts, 1), np.nan)


def run_records(dataset_dir, *, split="model_validation", train_split="model_train", use_cycle_features=True,
                min_target_months=1, ridge_lambda=RIDGE_LAMBDA) -> pd.DataFrame:
    """1つのデータセット(検索の設定)の、起点ごとの指標。"""
    dataset_dir = Path(dataset_dir)
    split_dir, train_dir = dataset_dir / split, dataset_dir / train_split
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    history = _load_history(split_dir)
    current = current_values(history)
    targets = np.load(split_dir / "target_values.npy").astype(float)
    mask = real_masks(split_dir)

    base_model = fit_weighted_ridge(split_features(train_dir, use_cycle_features=use_cycle_features), train_dir,
                                    ridge_lambda=ridge_lambda)
    guide_model = fit_weighted_ridge(split_features(train_dir, use_cycle_features=use_cycle_features, with_guide=True),
                                     train_dir, ridge_lambda=ridge_lambda)
    predictions = {
        "guide_copy": np.load(split_dir / "guide_baselines.npy").astype(float),
        "ridge": apply_weighted_ridge(base_model, split_features(split_dir, use_cycle_features=use_cycle_features)),
        "ridge_guide": apply_weighted_ridge(guide_model, split_features(
            split_dir, use_cycle_features=use_cycle_features, with_guide=True)),
    }
    frame = pd.DataFrame({
        "trend_id": metadata["trend_id"].astype(str), "dataset_id": metadata["dataset_id"].astype(str),
        "valid_months": mask.sum(1), "guide_count": metadata["guide_count"].to_numpy(),
        "months_after_maintenance": (metadata["months_after_maintenance"].to_numpy()
                                    if "months_after_maintenance" in metadata.columns else np.nan),
    })
    frame["actual_rise"] = _last_valid_mean(targets, mask) - current
    for name, prediction in predictions.items():
        frame[f"{name}_MAE"] = _record_mae(prediction, targets, mask)
        frame[f"{name}_rise"] = _last_valid_mean(prediction, mask) - current
    frame["added_value"] = frame["ridge_guide_MAE"] - frame["ridge_MAE"]
    return frame.loc[frame["valid_months"] >= int(min_target_months)].reset_index(drop=True)


def _segment_means(frame, column):
    return frame.groupby("dataset_id")[column].mean()


def _bootstrap_ci(per_segment, iterations, seed):
    values = np.asarray(per_segment, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(int(iterations))]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_run(frame, *, bootstrap=2000, seed=42) -> dict:
    """1つの設定の要約(セグメント単位の平均と95%区間、上昇件のAUC、経過月数別)。"""
    summary = {"records": int(len(frame)), "segments": int(frame["dataset_id"].nunique()),
               "anchors_without_guides": int((frame["guide_count"] == 0).sum())}
    for column in ("guide_copy_MAE", "ridge_MAE", "ridge_guide_MAE", "added_value"):
        per_segment = _segment_means(frame, column)
        summary[column] = {"mean": float(per_segment.mean()), "ci": _bootstrap_ci(per_segment, bootstrap, seed)}
    up = (frame["actual_rise"] >= UP_THRESHOLD).to_numpy()
    summary["up_cases"] = int(up.sum())
    summary["actual_rise_of_up_cases"] = float(frame.loc[up, "actual_rise"].mean()) if up.any() else float("nan")
    for name in ("guide_copy", "ridge", "ridge_guide"):
        score = frame[f"{name}_rise"].fillna(0.0).to_numpy()
        summary[f"{name}_up_AUC"] = _auc(score, up)
        summary[f"{name}_predicted_rise_of_up_cases"] = float(score[up].mean()) if up.any() else float("nan")
    ages = pd.to_numeric(frame["months_after_maintenance"], errors="coerce")
    by_age = {}
    for low, high in zip(AGE_BINS[:-1], AGE_BINS[1:]):
        part = frame.loc[(ages >= low) & (ages < high)]
        if len(part):
            label = f"{low}-{high - 1}" if high < 100 else f"{low}+"
            by_age[label] = {"records": int(len(part)), **{column: float(_segment_means(part, column).mean())
                             for column in ("guide_copy_MAE", "ridge_MAE", "ridge_guide_MAE", "added_value")}}
    summary["by_months_after_maintenance"] = by_age
    return summary


def compare_to_baseline(frame, baseline, *, bootstrap=2000, seed=42) -> dict:
    """同じ起点どうしの差(frame − baseline)。負ならframeの設定が良い。セグメント単位のbootstrap。"""
    if frame["trend_id"].tolist() != baseline["trend_id"].tolist():
        raise ValueError("比べるデータセットの起点(trend_idの並び)が一致しません。")
    result = {}
    for column in ("guide_copy_MAE", "ridge_guide_MAE", "added_value"):
        diff = frame[column] - baseline[column]
        per_segment = diff.groupby(frame["dataset_id"]).mean()
        result[column] = {"mean_difference": float(per_segment.mean()),
                          "ci": _bootstrap_ci(per_segment, bootstrap, seed)}
    return result


def evaluate_guide_value(runs: dict, output_dir, *, baseline=None, split="model_validation",
                         use_cycle_features=True, min_target_months=1, bootstrap=2000, seed=42) -> dict:
    """`runs`は{名前: データセットのディレクトリ}。`baseline`(名前)を指定すると、他の設定との差も出す。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline is not None and baseline not in runs:
        raise ValueError(f"baseline {baseline!r} がrunsにありません。")
    frames = {name: run_records(path, split=split, use_cycle_features=use_cycle_features,
                                min_target_months=min_target_months) for name, path in runs.items()}
    result = {"split": split, "use_cycle_features": bool(use_cycle_features), "baseline": baseline, "runs": {}}
    for name, frame in frames.items():
        result["runs"][name] = summarize_run(frame, bootstrap=bootstrap, seed=seed)
        if baseline is not None and name != baseline:
            result["runs"][name]["difference_from_baseline"] = compare_to_baseline(
                frame, frames[baseline], bootstrap=bootstrap, seed=seed)
        frame.to_csv(output_dir / f"records_{name}.csv", index=False, encoding="utf-8-sig")
    (output_dir / "guide_value.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for name, item in result["runs"].items():
        rows.append({"run": name, "records": item["records"], "segments": item["segments"],
                     "guide_copy_MAE": item["guide_copy_MAE"]["mean"], "ridge_MAE": item["ridge_MAE"]["mean"],
                     "ridge_guide_MAE": item["ridge_guide_MAE"]["mean"], "added_value": item["added_value"]["mean"],
                     "added_value_ci_lo": item["added_value"]["ci"][0], "added_value_ci_hi": item["added_value"]["ci"][1],
                     "guide_copy_up_AUC": item["guide_copy_up_AUC"], "ridge_guide_up_AUC": item["ridge_guide_up_AUC"],
                     "anchors_without_guides": item["anchors_without_guides"]})
    pd.DataFrame(rows).to_csv(output_dir / "guide_value_summary.csv", index=False, encoding="utf-8-sig")
    return result
