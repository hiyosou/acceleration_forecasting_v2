"""evaluation(metrics / aggregate / evaluate)の単体テスト(SPEC.md 1.4の数式・手計算例に対応)。"""

import json

import numpy as np
import pandas as pd
import pytest

from test_training import _raw_split

from acceleration_forecasting_v2.datasets.build import RawSplit, write_dataset
from acceleration_forecasting_v2.evaluation.aggregate import (
    bootstrap_record_level, bootstrap_segment_level, confidence_interval,
    evaluate_record_level, evaluate_segment_level,
)
from acceleration_forecasting_v2.evaluation.evaluate import build_comparison_table, evaluate, per_target_frame
from acceleration_forecasting_v2.evaluation.metrics import target_metrics


# --- target_metrics ---------------------------------------------------------------

def test_target_metrics_hand_computed_example():
    m = target_metrics([1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 3, 5], [0, 1, 2, 3], [2, 3, 4, 4.5])
    assert m["MAE"] == pytest.approx(0.25)
    assert m["MSE"] == pytest.approx(0.25)
    assert m["RMSE"] == pytest.approx(0.5)
    assert m["peak_value_error"] == pytest.approx(1.0)
    assert m["peak_month_error"] == 0
    assert m["coverage_p10_p90"] == pytest.approx(1.0)
    assert m["mean_interval_width"] == pytest.approx(1.875)
    assert m["median_adjacent_abs_difference"] == pytest.approx(4 / 3)
    assert m["adjacent_difference_MAE"] == pytest.approx(1 / 3)
    assert m["direction_sign_agreement"] == pytest.approx(1.0)
    assert m["valid_months"] == 4


def test_target_metrics_ignores_masked_months():
    actual = [1.0, 2.0, np.nan, np.nan]
    m = target_metrics(actual, [1, 1, 0, 0], [1.5, 2.5, 99, 99], [0, 0, 0, 0], [9, 9, 9, 9])
    assert m["valid_months"] == 2
    assert m["MAE"] == pytest.approx(0.5)  # 99は無視される


def test_target_metrics_returns_none_when_no_valid_month():
    assert target_metrics([np.nan] * 3, [0, 0, 0], [1, 1, 1], [0, 0, 0], [2, 2, 2]) is None


def test_target_metrics_correlation_is_nan_for_constant_actual():
    m = target_metrics([2, 2, 2], [1, 1, 1], [1, 2, 3], [0, 0, 0], [5, 5, 5])
    assert np.isnan(m["correlation"])


def test_target_metrics_peak_month_error_and_coverage_miss():
    m = target_metrics([1, 5, 2, 1], [1, 1, 1, 1], [1, 2, 6, 1], [1, 1, 1, 1], [2, 2, 2, 2])
    assert m["peak_month_error"] == 1  # 正解のピークは2番目、予測のピークは3番目
    # 区間[1,2]に対し actual=[1,5,2,1]: 1→内, 5→外, 2→内, 1→内 で 3/4。
    assert m["coverage_p10_p90"] == pytest.approx(0.75)


# --- 集計(SPEC 1.4) ---------------------------------------------------------------

def _unbalanced_frame():
    # セグメントA: 10件(MAE 0.20)、セグメントB: 1件(MAE 0.50)。SPEC 1.4の手計算例。
    return pd.DataFrame({
        "dataset_id": ["A"] * 10 + ["B"],
        "MAE": [0.20] * 10 + [0.50],
        "correlation": [np.nan] * 10 + [0.9],
    })


def test_record_level_is_plain_mean_over_records():
    result = evaluate_record_level(_unbalanced_frame(), ("MAE",))
    assert result["MAE"] == pytest.approx(2.5 / 11)  # ≈ 0.227


def test_segment_level_is_two_stage_mean():
    result = evaluate_segment_level(_unbalanced_frame(), ("MAE",))
    assert result["MAE"] == pytest.approx(0.35)


def test_segment_level_equals_record_level_when_balanced():
    frame = pd.DataFrame({"dataset_id": ["A", "A", "B", "B"], "MAE": [0.1, 0.3, 0.2, 0.4]})
    assert evaluate_segment_level(frame, ("MAE",))["MAE"] == pytest.approx(evaluate_record_level(frame, ("MAE",))["MAE"])


def test_aggregation_skips_nan_at_both_stages():
    result = evaluate_segment_level(_unbalanced_frame(), ("correlation",))
    assert result["correlation"] == pytest.approx(0.9)  # セグメントAは全てNaN -> 除外
    assert evaluate_record_level(_unbalanced_frame(), ("correlation",))["correlation"] == pytest.approx(0.9)


def test_bootstrap_segment_level_resamples_whole_segments_with_equal_weight():
    frame = _unbalanced_frame()
    boot = bootstrap_segment_level(frame, iterations=200, seed=0, metric_names=("MAE",))
    # 2セグメントから2個を復元抽出 -> 取りうる値は {0.20, 0.35, 0.50} のみ。
    allowed = np.array([0.20, 0.35, 0.50])
    assert all(np.isclose(allowed, value).any() for value in boot["MAE"])
    assert len(set(np.round(boot["MAE"], 6))) == 3


def test_bootstrap_record_level_weights_segments_by_record_count():
    frame = _unbalanced_frame()
    boot = bootstrap_record_level(frame, iterations=200, seed=0, metric_names=("MAE",))
    # AのみでB無し=0.20、BのみでA無し=0.50、A+B=(10*0.2+0.5)/11、A2回=0.20、B2回=0.50。
    allowed = np.array([0.20, 0.50, 2.5 / 11])
    assert all(np.isclose(allowed, value).any() for value in boot["MAE"])


def test_bootstrap_is_reproducible_with_seed():
    frame = _unbalanced_frame()
    a = bootstrap_segment_level(frame, iterations=20, seed=3, metric_names=("MAE",))
    b = bootstrap_segment_level(frame, iterations=20, seed=3, metric_names=("MAE",))
    pd.testing.assert_frame_equal(a, b)


def test_confidence_interval_keys_and_ordering():
    boot = pd.DataFrame({"MAE": np.linspace(0.0, 1.0, 101)})
    ci = confidence_interval(boot)
    assert ci["MAE_ci95_low"] == pytest.approx(0.025)
    assert ci["MAE_ci95_high"] == pytest.approx(0.975)


# --- evaluate(end-to-end) ---------------------------------------------------------

DATASET_IDS = ["segA"] * 4 + ["segB"]


def _write_eval_dataset(tmp_path, *, mask_first_record_after=None):
    raw = _raw_split(5, seed=7)
    if mask_first_record_after is not None:
        raw.arrays["target_masks"][0, mask_first_record_after:] = 0.0
        raw.arrays["target_values"][0, mask_first_record_after:] = np.nan
    metadata = pd.DataFrame({"trend_id": [f"t{i}" for i in range(5)], "dataset_id": DATASET_IDS,
                             "direction": "U", "bin_start_m": 2000.0, "anchor_date": "2023-01-15"})
    inference = RawSplit(arrays=raw.arrays, metadata=metadata)
    dataset_dir = tmp_path / "dataset"
    write_dataset({"model_train": _raw_split(16, seed=1), "model_validation": _raw_split(8, seed=2),
                   "inference": inference}, dataset_dir)
    return dataset_dir, raw


def _write_predictions(tmp_path, raw, offsets):
    rows = []
    for index, offset in enumerate(offsets):
        target = np.nan_to_num(raw.arrays["target_values"][index], nan=0.0)
        for month in range(12):
            median = float(target[month]) + offset
            rows.append({"trend_id": f"t{index}", "dataset_id": DATASET_IDS[index], "month_index": month + 1,
                         "prediction_median": median, "prediction_p10": median - 1.0,
                         "prediction_p90": median + 1.0, "prediction_std": 0.2})
    prediction_dir = tmp_path / "pred"
    prediction_dir.mkdir()
    pd.DataFrame(rows).to_csv(prediction_dir / "predictions.csv", index=False)
    return prediction_dir


def test_evaluate_reports_record_and_segment_level_that_differ_when_unbalanced(tmp_path):
    dataset_dir, raw = _write_eval_dataset(tmp_path)
    prediction_dir = _write_predictions(tmp_path, raw, [0.1, 0.1, 0.1, 0.1, 0.5])
    summary = evaluate(dataset_dir, prediction_dir, tmp_path / "out", bootstrap=30)
    assert summary["record_level"]["MAE"] == pytest.approx((4 * 0.1 + 0.5) / 5, abs=1e-6)  # 0.18
    assert summary["segment_level"]["MAE"] == pytest.approx((0.1 + 0.5) / 2, abs=1e-6)  # 0.30
    assert summary["counts"]["segment_count"] == 2
    assert summary["counts"]["evaluated_records"] == 5


def test_evaluate_writes_all_outputs_and_confidence_intervals(tmp_path):
    dataset_dir, raw = _write_eval_dataset(tmp_path)
    prediction_dir = _write_predictions(tmp_path, raw, [0.1] * 5)
    summary = evaluate(dataset_dir, prediction_dir, tmp_path / "out", bootstrap=20)
    for name in ("evaluation_per_target.csv", "evaluation_per_segment.csv", "bootstrap_record_level.csv",
                 "bootstrap_segment_level.csv", "evaluation_summary.json"):
        assert (tmp_path / "out" / name).is_file()
    assert "MAE_ci95_low" in summary["record_level_ci"]
    assert "MAE_ci95_high" in summary["segment_level_ci"]
    saved = json.loads((tmp_path / "out" / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert saved["record_level"]["MAE"] == pytest.approx(0.1, abs=1e-6)


def test_evaluate_includes_guide_baseline_metrics_as_reference(tmp_path):
    dataset_dir, raw = _write_eval_dataset(tmp_path)
    prediction_dir = _write_predictions(tmp_path, raw, [0.3] * 5)
    summary = evaluate(dataset_dir, prediction_dir, tmp_path / "out", bootstrap=0)
    # テスト用データではguide_baselines==target(誤差0)、予測は+0.3ずれている。
    assert summary["record_level"]["baseline_MAE"] == pytest.approx(0.0, abs=1e-6)
    assert summary["record_level"]["MAE"] == pytest.approx(0.3, abs=1e-6)
    assert "record_level_ci" not in summary


def test_evaluate_excludes_records_with_too_few_valid_target_months(tmp_path):
    dataset_dir, raw = _write_eval_dataset(tmp_path, mask_first_record_after=4)
    prediction_dir = _write_predictions(tmp_path, raw, [0.1] * 5)
    frame, counts = per_target_frame(dataset_dir, prediction_dir, min_target_months=8)
    assert counts["evaluable_records"] == 5
    assert counts["evaluated_records"] == 4
    assert "t0" not in set(frame["trend_id"])
    _, lenient = per_target_frame(dataset_dir, prediction_dir, min_target_months=4)
    assert lenient["evaluated_records"] == 5


def test_evaluate_rejects_when_no_record_is_evaluable(tmp_path):
    dataset_dir, raw = _write_eval_dataset(tmp_path)
    prediction_dir = _write_predictions(tmp_path, raw, [0.1] * 5)
    with pytest.raises(ValueError, match="評価対象"):
        evaluate(dataset_dir, prediction_dir, tmp_path / "out", min_target_months=13)


def test_build_comparison_table_lays_out_both_levels_for_each_run():
    def summary(mae):
        names = ("MAE", "RMSE", "baseline_MAE")
        return {"record_level": {n: mae for n in names}, "segment_level": {n: mae * 2 for n in names}}
    table = build_comparison_table({"epsilon": summary(0.4), "v_prediction": summary(0.3)})
    assert list(table.columns) == ["epsilon|record_level", "epsilon|segment_level",
                                   "v_prediction|record_level", "v_prediction|segment_level"]
    assert table.loc["MAE", "epsilon|segment_level"] == pytest.approx(0.8)
    assert table.loc["MAE", "v_prediction|record_level"] == pytest.approx(0.3)
