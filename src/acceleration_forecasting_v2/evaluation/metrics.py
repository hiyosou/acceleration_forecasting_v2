"""1レコード(12か月)あたりの評価指標。`acceleration_forecasting_12m/.../evaluation/metrics.py`を
無変更で移植(正解のmask=0の月は全ての指標から除外される)。
"""

from __future__ import annotations

import numpy as np


def target_metrics(actual, mask, median, p10, p90):
    actual, mask = np.asarray(actual, float), np.asarray(mask, bool)
    median, p10, p90 = map(lambda value: np.asarray(value, float), (median, p10, p90))
    valid = mask & np.isfinite(actual) & np.isfinite(median)
    if not valid.any():
        return None
    error = median[valid] - actual[valid]
    correlation = (
        float(np.corrcoef(actual[valid], median[valid])[0, 1])
        if valid.sum() >= 2 and np.std(actual[valid]) > 0 and np.std(median[valid]) > 0 else np.nan
    )
    actual_indices, prediction_indices = np.flatnonzero(valid), np.flatnonzero(valid)
    actual_peak = actual_indices[int(np.argmax(actual[valid]))]
    prediction_peak = prediction_indices[int(np.argmax(median[valid]))]
    adjacent_valid = valid[:-1] & valid[1:]
    if adjacent_valid.any():
        actual_delta = np.diff(actual)[adjacent_valid]
        prediction_delta = np.diff(median)[adjacent_valid]
        adjacent_difference_mae = float(np.mean(np.abs(prediction_delta - actual_delta)))
        direction_sign_agreement = float(np.mean(np.sign(prediction_delta) == np.sign(actual_delta)))
    else:
        adjacent_difference_mae = np.nan
        direction_sign_agreement = np.nan
    return {
        "MAE": float(np.mean(np.abs(error))), "MSE": float(np.mean(error**2)),
        "RMSE": float(np.sqrt(np.mean(error**2))), "correlation": correlation,
        "peak_value_error": float(abs(np.max(median[valid]) - np.max(actual[valid]))),
        "peak_month_error": int(abs(prediction_peak - actual_peak)),
        "coverage_p10_p90": float(np.mean((actual[valid] >= p10[valid]) & (actual[valid] <= p90[valid]))),
        "mean_interval_width": float(np.mean(p90[valid] - p10[valid])),
        "median_adjacent_abs_difference": float(np.mean(np.abs(np.diff(median)))) if len(median) > 1 else 0.0,
        "adjacent_difference_MAE": adjacent_difference_mae,
        "direction_sign_agreement": direction_sign_agreement,
        "valid_months": int(valid.sum()),
    }
