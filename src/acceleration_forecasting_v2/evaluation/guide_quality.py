"""guideの質 × 生成精度の基準曲線(2026-10-15、手順0)。

`datasets/mixed_guide.py`で実guideと正解をλで混ぜたデータセットごとに、
拡散モデルのMAEと、次の比較対象を並べる:

- guide_copy: guideのsoftmax平均をそのまま予測にする(`guide_baselines`、評価の`baseline_MAE`)
- history_ridge: 履歴特徴のridge(`evaluation/baselines.py`、λによらず一定)
- history_guide_ridge: 履歴特徴 + guide平均の変化量のridge。同じλで混ぜたguideを
  `model_train`で学習する。「この質のguideから単純な学習器でどこまで取り出せるか」を表し、
  生成モジュールが目指すべき線になる。

MAEはすべて`evaluate`と同じ定義(`metrics.target_metrics`のレコードごとのMAEを、
`evaluate`が評価対象にしたレコード(`evaluation_per_target.csv`)で平均)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import (
    RIDGE_LAMBDA, _anchor_months, _load_history, apply_weighted_ridge, current_values, fit_weighted_ridge,
    history_features,
)
from .metrics import target_metrics


def history_guide_features(split_dir, anchor_months=None):
    """`history_features`の後ろに、guide平均の変化量(`guide_baselines − current`、12か月)と
    その12か月平均を足した特徴量。"""
    split_dir = Path(split_dir)
    history = _load_history(split_dir)
    guide_mean = np.load(split_dir / "guide_baselines.npy").astype(float)
    delta = np.nan_to_num(guide_mean - current_values(history)[:, None])
    return np.column_stack([history_features(history, anchor_months), delta, delta.mean(1)])


def _split_anchor_months(split_dir, use_anchor_month):
    if not use_anchor_month:
        return None
    return _anchor_months(pd.read_csv(Path(split_dir) / "metadata.csv", encoding="utf-8-sig"))


def fit_history_guide_ridge(train_dir, *, use_anchor_month=True, ridge_lambda=RIDGE_LAMBDA):
    """`model_train`で履歴+guide特徴のridgeを学習する(解法は`fit_history_ridge`と同じ)。"""
    months = _split_anchor_months(train_dir, use_anchor_month)
    model = fit_weighted_ridge(history_guide_features(train_dir, months), train_dir, ridge_lambda=ridge_lambda)
    return {**model, "use_anchor_month": months is not None}


def predict_history_guide_ridge(model, split_dir):
    months = _split_anchor_months(split_dir, model["use_anchor_month"])
    if model["use_anchor_month"] and months is None:
        raise ValueError("学習時はanchor月を使ったが、予測対象のmetadataにanchor_date列がありません。")
    return apply_weighted_ridge(model, history_guide_features(split_dir, months))


def _record_mae(prediction, targets, masks):
    result = target_metrics(targets, masks, prediction, prediction, prediction)
    return result["MAE"] if result is not None else np.nan


def guide_quality_row(dataset_dir, evaluation_dir, guide_fidelity_dir, *, lam, checkpoint_name, split="inference"):
    """曲線の1点分(1つのcheckpoint × 1つのλ)の値を集める。

    Returns:
        dict。MAE系はrecord-level(`*_MAE`)とsegment-level(`*_MAE_segment`)の両方。
    """
    dataset_dir, evaluation_dir = Path(dataset_dir), Path(evaluation_dir)
    split_dir = dataset_dir / split
    per_target = pd.read_csv(evaluation_dir / "evaluation_per_target.csv", encoding="utf-8-sig")
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    lookup = {str(trend_id): index for index, trend_id in enumerate(metadata["trend_id"])}
    positions = [lookup[str(trend_id)] for trend_id in per_target["trend_id"]]

    use_anchor_month = "anchor_date" in metadata.columns
    model = fit_history_guide_ridge(dataset_dir / "model_train", use_anchor_month=use_anchor_month)
    ridge = predict_history_guide_ridge(model, split_dir)
    targets = np.load(split_dir / "target_values.npy")
    masks = np.load(split_dir / "target_masks.npy")
    per_target["history_guide_ridge_MAE"] = [_record_mae(ridge[index], targets[index], masks[index])
                                             for index in positions]

    columns = {"model_MAE": "MAE", "guide_copy_MAE": "baseline_MAE", "history_ridge_MAE": "history_ridge_MAE",
               "history_guide_ridge_MAE": "history_guide_ridge_MAE", "model_mean_error": "mean_error",
               "model_correlation": "correlation"}
    segment = per_target.groupby("dataset_id")[list(columns.values())].mean()
    fidelity = json.loads((Path(guide_fidelity_dir) / "generated_vs_guide_fidelity.json").read_text(encoding="utf-8"))
    row = {"checkpoint": checkpoint_name, "lambda": float(lam), "dataset_dir": str(dataset_dir),
           "records": int(len(per_target)), "segments": int(len(segment)),
           "guide_vs_target_MAE": fidelity["guide_vs_target_MAE"]["mean"],
           "generated_vs_guide_MAE": fidelity["generated_vs_guide_MAE"]["mean"]}
    for name, column in columns.items():
        row[name] = float(per_target[column].mean(skipna=True))
        if name.endswith("_MAE"):
            row[f"{name}_segment"] = float(segment[column].mean(skipna=True))
    return row


# 系列の色はcheckpoint(実体)ごとに固定する(dataviz既定パレットのカテゴリ順)。
_CHECKPOINT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
_GUIDE_RIDGE_COLOR = "#4a3aa7"
_REFERENCE_COLOR = "#8a8a85"


def plot_guide_quality_curve(rows, output_path, *, dpi=150):
    """横軸=guideの誤差(guide_vs_target_MAE)、縦軸=MAEの曲線をPNGに描く。

    元データセットごとに1枚のパネル(縦軸は共有)。各パネルに、そのデータセットで評価した
    checkpointの拡散モデルの線、guide_copy、history_guide_ridge、history_ridge(水平線)を描く。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    frame["source"] = frame["dataset_dir"].map(lambda value: Path(value).name.rsplit("_lam", 1)[0])
    sources = list(dict.fromkeys(frame["source"]))
    checkpoints = list(dict.fromkeys(frame["checkpoint"]))
    colors = {name: _CHECKPOINT_COLORS[index % len(_CHECKPOINT_COLORS)] for index, name in enumerate(checkpoints)}

    fig, axes = plt.subplots(1, len(sources), figsize=(6.0 * len(sources), 4.6), sharey=True, squeeze=False)
    for axis, source in zip(axes[0], sources):
        part = frame.loc[frame["source"] == source]
        reference = part.groupby("lambda")[["guide_vs_target_MAE", "guide_copy_MAE", "history_guide_ridge_MAE",
                                            "history_ridge_MAE"]].mean().sort_values("guide_vs_target_MAE")
        axis.axhline(reference["history_ridge_MAE"].mean(), color=_REFERENCE_COLOR, linewidth=1.2, linestyle=":",
                     label="history ridge")
        axis.plot(reference["guide_vs_target_MAE"], reference["guide_copy_MAE"], color=_REFERENCE_COLOR,
                  linewidth=2, linestyle="--", marker="o", markersize=5, label="guide copy (softmax mean)")
        axis.plot(reference["guide_vs_target_MAE"], reference["history_guide_ridge_MAE"], color=_GUIDE_RIDGE_COLOR,
                  linewidth=2, linestyle="--", marker="s", markersize=5, label="history + guide ridge")
        for checkpoint, group in part.groupby("checkpoint", sort=False):
            group = group.sort_values("guide_vs_target_MAE")
            axis.plot(group["guide_vs_target_MAE"], group["model_MAE"], color=colors[checkpoint], linewidth=2,
                      marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                      label=f"diffusion: {checkpoint}")
        for lam, point in reference.iterrows():
            axis.annotate(f"λ={lam:.2f}",(point["guide_vs_target_MAE"], point["history_guide_ridge_MAE"]),
                          textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#555555")
        axis.set_title(source, fontsize=11)
        axis.set_xlabel("guide error vs target (guide_vs_target_MAE)")
        axis.grid(color="#e5e5e2", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0][0].set_ylabel("MAE vs target (record-level)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
