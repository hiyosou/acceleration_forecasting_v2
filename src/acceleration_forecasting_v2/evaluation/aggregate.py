"""record-level / segment-level(2段階平均)の集計とbootstrap。SPEC.md 1.4の数式そのもの。

record-level:  Σ_s Σ_{i∈R_s} m_i / Σ_s |R_s|                    (anchor単位の単純平均)
segment-level: mean_s( mean_{i∈R_s} m_i )                        (セグメント内平均→セグメント間平均)

bootstrapはどちらも`dataset_id`(セグメント)単位で復元抽出する。segment-levelは各反復で
「抽出されたセグメントのセグメント内平均」の単純平均を取る(anchor数の多いセグメントが
支配しない)。record-levelは旧実装と同じく、抽出したセグメントの全レコードを連結して
単純平均を取る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES = (
    "MAE", "MSE", "RMSE", "correlation", "peak_value_error", "peak_month_error",
    "coverage_p10_p90", "mean_interval_width", "median_adjacent_abs_difference",
    "adjacent_difference_MAE", "direction_sign_agreement", "ensemble_mean_std",
)
BASELINE_METRIC_NAMES = ("baseline_MAE", "baseline_RMSE")


def _present(frame, names):
    return [name for name in names if name in frame.columns]


def evaluate_record_level(per_target_metrics: pd.DataFrame, metric_names=METRIC_NAMES) -> dict:
    """各指標のanchor(レコード)単位の単純平均。NaNは除外して平均する。"""
    return {name: float(per_target_metrics[name].mean(skipna=True)) for name in _present(per_target_metrics, metric_names)}


def segment_means(per_target_metrics: pd.DataFrame, metric_names=METRIC_NAMES) -> pd.DataFrame:
    """dataset_idごとの各指標のセグメント内平均(第1段階)。index=dataset_id。"""
    names = _present(per_target_metrics, metric_names)
    return per_target_metrics.groupby("dataset_id")[names].mean()


def evaluate_segment_level(per_target_metrics: pd.DataFrame, metric_names=METRIC_NAMES) -> dict:
    """セグメント内平均→セグメント間平均の2段階平均(第2段階は各セグメントを同じ重みで平均)。"""
    means = segment_means(per_target_metrics, metric_names)
    return {name: float(means[name].mean(skipna=True)) for name in means.columns}


def bootstrap_segment_level(per_target_metrics: pd.DataFrame, *, iterations: int = 1000, seed: int = 42,
                            metric_names=METRIC_NAMES) -> pd.DataFrame:
    """dataset_idを復元抽出し、各反復で抽出セグメントのセグメント内平均を単純平均する。"""
    means = segment_means(per_target_metrics, metric_names)
    values = means.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(iterations)):
        picked = rng.integers(0, len(means), size=len(means))
        with np.errstate(all="ignore"):
            rows.append(np.nanmean(values[picked], axis=0))
    return pd.DataFrame(rows, columns=means.columns)


def bootstrap_record_level(per_target_metrics: pd.DataFrame, *, iterations: int = 1000, seed: int = 42,
                           metric_names=METRIC_NAMES) -> pd.DataFrame:
    """dataset_idを復元抽出し、抽出セグメントの全レコードを連結して単純平均する(旧実装と同じ)。"""
    names = _present(per_target_metrics, metric_names)
    groups = {key: group[names] for key, group in per_target_metrics.groupby("dataset_id")}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(iterations)):
        picked = rng.integers(0, len(keys), size=len(keys))
        combined = pd.concat([groups[keys[index]] for index in picked], ignore_index=True)
        rows.append(combined.mean(skipna=True).to_dict())
    return pd.DataFrame(rows, columns=names)


def confidence_interval(bootstrap_frame: pd.DataFrame) -> dict:
    """bootstrap結果から各指標の95%信頼区間 `{name}_ci95_low/high` を作る。"""
    output = {}
    for name in bootstrap_frame.columns:
        low, high = np.nanpercentile(bootstrap_frame[name].to_numpy(dtype=float), [2.5, 97.5])
        output[f"{name}_ci95_low"], output[f"{name}_ci95_high"] = float(low), float(high)
    return output
