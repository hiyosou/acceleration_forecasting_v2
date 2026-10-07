"""evaluation.visualize の単体テスト(予測結果の可視化、plot-prediction)。"""

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from test_training import _raw_split

from acceleration_forecasting_v2.datasets.build import write_dataset
from acceleration_forecasting_v2.evaluation.visualize import (
    classify_maintenance_relation,
    compute_fixed_bins,
    load_segment_measurement_history,
    months_since_maintenance,
    plot_prediction,
    plot_prediction_batch,
    reconstruct_segment_maintenance_events,
    render_generation_histogram,
    render_measurement_history,
    render_quartile_band,
    select_record,
    select_records,
)


# --- reconstruct_segment_maintenance_events ------------------------------------------------

def _trend_catalog_frame(rows):
    return pd.DataFrame(rows)


def test_reconstruct_segment_maintenance_events_dedupes_per_dataset_id_and_sorts():
    catalog = _trend_catalog_frame([
        {"dataset_id": "D_100-200_001", "direction": "D", "bin_start_m": 100.0, "bin_end_m": 200.0,
         "cutoff_maintenance_date": "2023-06-01", "maintenance_type": "real", "maintenance_description": "A"},
        {"dataset_id": "D_100-200_001", "direction": "D", "bin_start_m": 100.0, "bin_end_m": 200.0,
         "cutoff_maintenance_date": "2023-06-01", "maintenance_type": "real", "maintenance_description": "A"},
        {"dataset_id": "D_100-200_002", "direction": "D", "bin_start_m": 100.0, "bin_end_m": 200.0,
         "cutoff_maintenance_date": "2023-01-01", "maintenance_type": "real", "maintenance_description": "B"},
        {"dataset_id": "D_100-200_003", "direction": "D", "bin_start_m": 100.0, "bin_end_m": 200.0,
         "cutoff_maintenance_date": None, "maintenance_type": None, "maintenance_description": None},
    ])
    events = reconstruct_segment_maintenance_events(catalog)
    key = ("D", 100.0, 200.0)
    assert key in events
    dates = [event["date"] for event in events[key]]
    assert dates == sorted(dates)
    assert len(dates) == 2
    assert dates[0] == pd.Timestamp("2023-01-01")


def test_reconstruct_segment_maintenance_events_returns_no_key_for_all_nan_segment():
    catalog = _trend_catalog_frame([
        {"dataset_id": "U_1-2_001", "direction": "U", "bin_start_m": 1.0, "bin_end_m": 2.0,
         "cutoff_maintenance_date": None, "maintenance_type": None, "maintenance_description": None},
    ])
    events = reconstruct_segment_maintenance_events(catalog)
    assert ("U", 1.0, 2.0) not in events


# --- months_since_maintenance / classify_maintenance_relation ------------------------------

def test_months_since_maintenance_picks_the_most_recent_past_event():
    events = [
        {"date": pd.Timestamp("2023-01-01")},
        {"date": pd.Timestamp("2023-06-01")},
        {"date": pd.Timestamp("2023-12-01")},  # 未来のイベント、無視されるべき
    ]
    months = months_since_maintenance(pd.Timestamp("2023-08-01"), events)
    expected = (pd.Timestamp("2023-08-01") - pd.Timestamp("2023-06-01")).days / 30.44
    assert months == pytest.approx(expected)


def test_months_since_maintenance_returns_none_when_no_past_event():
    events = [{"date": pd.Timestamp("2024-01-01")}]
    assert months_since_maintenance(pd.Timestamp("2023-01-01"), events) is None


@pytest.mark.parametrize("months,expected", [
    (2.0, "immediately_after"),
    (1.0, "immediately_after"),
    (6.0, "elapsed"),
    (10.0, "elapsed"),
    (4.0, None),
    (None, None),
])
def test_classify_maintenance_relation_boundaries(months, expected):
    assert classify_maintenance_relation(months, immediately_after_months=2.0, elapsed_months=6.0) == expected


# --- select_record --------------------------------------------------------------------------

def _metadata_frame(rows):
    return pd.DataFrame(rows)


def test_select_record_with_explicit_trend_id_bypasses_bucketing():
    metadata = _metadata_frame([
        {"trend_id": "t1", "direction": "D", "bin_start_m": 0.0, "bin_end_m": 100.0, "anchor_date": "2023-01-01"},
    ])
    catalog = _trend_catalog_frame([])
    result = select_record(metadata, catalog, trend_id="t1", available_trend_ids=set())
    assert result == {"trend_id": "t1", "maintenance_relation": None, "months_since_maintenance": None,
                      "selection_pool_size": None}


def test_select_record_raises_for_unknown_explicit_trend_id():
    metadata = _metadata_frame([{"trend_id": "t1", "direction": "D", "bin_start_m": 0.0, "bin_end_m": 100.0,
                                 "anchor_date": "2023-01-01"}])
    with pytest.raises(ValueError):
        select_record(metadata, _trend_catalog_frame([]), trend_id="unknown")


def _segment_metadata_and_catalog():
    # 1区間、施工イベント1件(2023-01-01)。
    # t_after=直後(10日後)、t_elapsed=経過(200日後)、t_gap=中間(80日後、どちらにも分類されない)。
    metadata = _metadata_frame([
        {"trend_id": "t_after", "direction": "D", "bin_start_m": 0.0, "bin_end_m": 100.0,
         "anchor_date": "2023-01-11"},
        {"trend_id": "t_elapsed", "direction": "D", "bin_start_m": 0.0, "bin_end_m": 100.0,
         "anchor_date": "2023-07-20"},
        {"trend_id": "t_gap", "direction": "D", "bin_start_m": 0.0, "bin_end_m": 100.0,
         "anchor_date": "2023-03-22"},
    ])
    catalog = _trend_catalog_frame([
        {"dataset_id": "seg_001", "direction": "D", "bin_start_m": 0.0, "bin_end_m": 100.0,
         "cutoff_maintenance_date": "2023-01-01", "maintenance_type": "real", "maintenance_description": "工事"},
    ])
    return metadata, catalog


def test_select_record_auto_selection_pool_respects_maintenance_relation():
    metadata, catalog = _segment_metadata_and_catalog()
    result = select_record(metadata, catalog, maintenance_relation="immediately_after", seed=0)
    assert result["trend_id"] == "t_after"
    assert result["selection_pool_size"] == 1

    result = select_record(metadata, catalog, maintenance_relation="elapsed", seed=0)
    assert result["trend_id"] == "t_elapsed"
    assert result["selection_pool_size"] == 1


def test_select_record_any_pools_across_both_categories():
    metadata, catalog = _segment_metadata_and_catalog()
    result = select_record(metadata, catalog, maintenance_relation="any", seed=0)
    assert result["trend_id"] in {"t_after", "t_elapsed"}
    assert result["selection_pool_size"] == 2


def test_select_record_is_deterministic_for_a_fixed_seed():
    metadata, catalog = _segment_metadata_and_catalog()
    first = select_record(metadata, catalog, maintenance_relation="any", seed=7)
    second = select_record(metadata, catalog, maintenance_relation="any", seed=7)
    assert first == second


def test_select_record_raises_with_diagnostic_message_when_pool_is_empty():
    metadata, catalog = _segment_metadata_and_catalog()
    with pytest.raises(ValueError, match="合致するレコードがありません"):
        select_record(metadata, catalog, maintenance_relation="immediately_after", immediately_after_months=0.01)


def test_select_record_filters_by_available_trend_ids():
    metadata, catalog = _segment_metadata_and_catalog()
    result = select_record(metadata, catalog, maintenance_relation="any", available_trend_ids={"t_elapsed"}, seed=0)
    assert result["trend_id"] == "t_elapsed"
    assert result["selection_pool_size"] == 1


def test_select_record_custom_thresholds_shift_classification():
    metadata, catalog = _segment_metadata_and_catalog()
    result = select_record(metadata, catalog, maintenance_relation="immediately_after",
                           immediately_after_months=3.0, seed=0)
    assert result["trend_id"] in {"t_after", "t_gap"}
    assert result["selection_pool_size"] == 2


# --- select_records(複数選択版) --------------------------------------------------------------

def _segment_metadata_and_catalog_many():
    # 1区間、施工イベント1件(2023-01-01)。immediately_after候補2件・elapsed候補3件。
    rows = [
        {"trend_id": "a1", "anchor_date": "2023-01-05"},   # 4日後、immediately_after
        {"trend_id": "a2", "anchor_date": "2023-01-20"},   # 19日後、immediately_after
        {"trend_id": "e1", "anchor_date": "2023-07-10"},   # elapsed
        {"trend_id": "e2", "anchor_date": "2023-08-10"},   # elapsed
        {"trend_id": "e3", "anchor_date": "2023-09-10"},   # elapsed
    ]
    for row in rows:
        row.update({"direction": "D", "bin_start_m": 0.0, "bin_end_m": 100.0})
    metadata = _metadata_frame(rows)
    catalog = _trend_catalog_frame([
        {"dataset_id": "seg_001", "direction": "D", "bin_start_m": 0.0, "bin_end_m": 100.0,
         "cutoff_maintenance_date": "2023-01-01", "maintenance_type": "real", "maintenance_description": "工事"},
    ])
    return metadata, catalog


def test_select_records_picks_up_to_count_per_category():
    metadata, catalog = _segment_metadata_and_catalog_many()
    result = select_records(metadata, catalog, count_per_category=2, seed=0)
    assert len(result["immediately_after"]) == 2
    assert set(result["immediately_after"]) == {"a1", "a2"}
    assert len(result["elapsed"]) == 2
    assert set(result["elapsed"]) <= {"e1", "e2", "e3"}
    assert result["pool_sizes"] == {"immediately_after": 2, "elapsed": 3}


def test_select_records_uses_all_available_when_pool_smaller_than_requested():
    metadata, catalog = _segment_metadata_and_catalog_many()
    result = select_records(metadata, catalog, count_per_category=10, seed=0)
    assert set(result["immediately_after"]) == {"a1", "a2"}
    assert set(result["elapsed"]) == {"e1", "e2", "e3"}


def test_select_records_filters_by_available_trend_ids():
    metadata, catalog = _segment_metadata_and_catalog_many()
    result = select_records(metadata, catalog, count_per_category=10,
                            available_trend_ids={"a1", "e1", "e2"}, seed=0)
    assert set(result["immediately_after"]) == {"a1"}
    assert set(result["elapsed"]) == {"e1", "e2"}


def test_select_records_is_deterministic_for_a_fixed_seed():
    metadata, catalog = _segment_metadata_and_catalog_many()
    first = select_records(metadata, catalog, count_per_category=1, seed=3)
    second = select_records(metadata, catalog, count_per_category=1, seed=3)
    assert first == second


# --- compute_fixed_bins -----------------------------------------------------------------------

def test_compute_fixed_bins_covers_the_range_with_bin_width_increments():
    edges = compute_fixed_bins(bin_width=0.1, y_bounds=(0.1, 6.0))
    assert edges[0] == pytest.approx(0.1)
    assert edges[-1] == pytest.approx(6.0)
    assert len(edges) == 60
    np.testing.assert_allclose(np.diff(edges), 0.1, atol=1e-9)


# --- render_generation_histogram / render_quartile_band ---------------------------------------

def test_render_generation_histogram_all_same_value_gives_one_full_length_bar_per_month():
    fig, ax = plt.subplots()
    dates = pd.date_range("2023-01-01", periods=2, freq="MS")
    samples = np.full((100, 2), 1.55)
    bin_edges = compute_fixed_bins(bin_width=0.1, y_bounds=(0.1, 6.0))
    render_generation_histogram(ax, dates, samples, bin_edges, max_half_width_days=6.0)
    widths = [patch.get_width() for patch in ax.patches]
    assert len(widths) == 2
    assert all(width == pytest.approx(12.0) for width in widths)
    plt.close(fig)


def test_render_generation_histogram_splits_evenly_across_two_bins_gives_equal_length_bars():
    fig, ax = plt.subplots()
    dates = pd.date_range("2023-01-01", periods=1, freq="MS")
    samples = np.concatenate([np.full((50, 1), 1.05), np.full((50, 1), 2.05)], axis=0)
    bin_edges = compute_fixed_bins(bin_width=0.1, y_bounds=(0.1, 6.0))
    render_generation_histogram(ax, dates, samples, bin_edges, max_half_width_days=6.0)
    widths = [patch.get_width() for patch in ax.patches]
    assert len(widths) == 2
    assert widths[0] == pytest.approx(widths[1])
    plt.close(fig)


def test_render_quartile_band_matches_numpy_percentile():
    fig, ax = plt.subplots()
    dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(100, 3))
    render_quartile_band(ax, dates, samples)
    expected_q1 = np.percentile(samples, 25, axis=0)
    expected_q3 = np.percentile(samples, 75, axis=0)
    assert len(ax.lines) == 2
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_q1)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), expected_q3)
    plt.close(fig)


# --- load_segment_measurement_history / render_measurement_history -----------------------------

def _write_tiny_vector_database(path, rows):
    """rowsは(trend_id, measurement_date, current_acc_z_max, direction, bin_start_m,
    bin_end_m, velocity_or_None)のタプル列。velocityがNoneの行はwaveform_recordsに
    何も入れない(=速度欠損として扱われるべきレコード)。
    """
    from acceleration_forecasting_v2.retrieval.database import SCHEMA

    connection = sqlite3.connect(str(path))
    connection.executescript(SCHEMA)
    seen_trend_ids = set()
    for index, (trend_id, date, acc, direction, bin_start_m, bin_end_m, velocity) in enumerate(rows):
        if trend_id not in seen_trend_ids:
            seen_trend_ids.add(trend_id)
            connection.execute(
                "INSERT INTO trends (trend_id, dataset_id, model_split, measurement_date, direction, "
                "bin_start_m, bin_end_m, current_acc_z_max, future_values, future_mask, selected_dates, "
                "guide_available_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (trend_id, "d0", "model_train", date, direction, bin_start_m, bin_end_m, acc,
                 "[]", "[]", "[]", "2099-01-01"),
            )
        if velocity is not None:
            connection.execute(
                "INSERT INTO waveform_records (record_id, measurement_id, measurement_date, direction, "
                "bin_start_m, bin_end_m, mean_velocity_kmh, source_csv_path, waveform_sha256, dataset_id, "
                "trend_id, embedding, embedding_dim) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"r{index}", f"m{index}", date, direction, bin_start_m, bin_end_m, velocity,
                 "x.csv", "sha", "d0", trend_id, b"\x00\x00\x00\x00", 1),
            )
    connection.commit()
    connection.close()


def test_load_segment_measurement_history_averages_velocity_and_filters_by_segment(tmp_path):
    db_path = tmp_path / "vector_database.sqlite"
    _write_tiny_vector_database(db_path, [
        ("t0", "2023-01-01", 1.0, "D", 0.0, 100.0, 60.0),
        ("t0", "2023-01-01", 1.0, "D", 0.0, 100.0, 80.0),  # t0と同じ日に2本→平均70.0
        ("t1", "2023-02-01", 1.5, "D", 0.0, 100.0, None),  # 速度欠損
        ("t2", "2023-03-01", 2.0, "U", 0.0, 100.0, 50.0),  # 別区間(direction違い)、対象外
    ])
    history = load_segment_measurement_history(db_path, "D", 0.0)
    assert set(history["trend_id"]) == {"t0", "t1"}
    row_t0 = history.loc[history["trend_id"] == "t0"].iloc[0]
    assert row_t0["velocity"] == pytest.approx(70.0)
    row_t1 = history.loc[history["trend_id"] == "t1"].iloc[0]
    assert pd.isna(row_t1["velocity"])


def test_render_measurement_history_draws_colored_and_gray_scatter_separately():
    fig, ax = plt.subplots()
    history = pd.DataFrame({
        "trend_id": ["a", "b"], "measurement_date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
        "current_acc_z_max": [1.0, 2.0], "velocity": [60.0, np.nan],
    })
    scatter = render_measurement_history(ax, history)
    assert scatter is not None  # 速度ありの点が1件以上あるので、色分け散布図のartistが返る
    assert len(ax.collections) == 2  # 色分け分1つ + グレー分1つ
    plt.close(fig)


def test_plot_prediction_includes_measurement_history_when_database_path_is_given(tmp_path):
    dataset_dir, artifact_dir, prediction_dir, metadata = _write_visualize_fixture(tmp_path)
    row = metadata.iloc[0]
    _write_tiny_vector_database(artifact_dir / "vector_database.sqlite", [
        (str(row["trend_id"]), str(row["anchor_date"]), 1.0, "D", 0.0, 100.0, 60.0),
    ])
    result = plot_prediction(dataset_dir, prediction_dir, artifact_dir, tmp_path / "out_hist",
                             split="inference", trend_id=str(row["trend_id"]))
    output_path = Path(result["output_path"])
    assert output_path.is_file() and output_path.stat().st_size > 0


# --- plot_prediction(スモークテスト、モデル・学習・DDIMは一切使わない) ------------------------

def _write_visualize_fixture(tmp_path, *, n_inference=3):
    dataset_dir = tmp_path / "dataset"
    inference_raw = _raw_split(n_inference, seed=5)
    anchor_dates = ["2023-01-11", "2023-07-20", "2023-03-22"][:n_inference]
    metadata = inference_raw.metadata.copy()
    metadata["direction"] = "D"
    metadata["bin_start_m"] = 0.0
    metadata["bin_end_m"] = 100.0
    metadata["dataset_id"] = "D_0-100_001"
    metadata["anchor_date"] = anchor_dates
    metadata["current_acc_z_max"] = inference_raw.arrays["history_values"][:, -1]
    inference_raw = replace(inference_raw, metadata=metadata)

    write_dataset({
        "model_train": _raw_split(16, seed=1),
        "model_validation": _raw_split(8, seed=2),
        "inference": inference_raw,
    }, dataset_dir)

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, trend_id in enumerate(metadata["trend_id"]):
        anchor = pd.Timestamp(anchor_dates[index])
        forecast_dates = pd.date_range(anchor + pd.DateOffset(months=1), periods=12, freq="MS")
        rows.append({
            "trend_id": trend_id, "dataset_id": "D_0-100_001", "direction": "D",
            "bin_start_m": 0.0, "bin_end_m": 100.0,
            "selected_dates": json.dumps([date.strftime("%Y-%m-%d") for date in forecast_dates]),
            "cutoff_maintenance_date": "2023-01-01", "maintenance_type": "real",
            "maintenance_description": "軌道整備",
        })
    pd.DataFrame(rows).to_csv(artifact_dir / "trend_catalog.csv", index=False, encoding="utf-8-sig")

    prediction_dir = tmp_path / "pred"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=1.5, scale=0.2, size=(n_inference, 10, 12)).astype(np.float32)
    np.savez_compressed(
        prediction_dir / "samples.npz",
        trend_ids=metadata["trend_id"].astype(str).to_numpy().astype("U"), samples=samples,
    )
    return dataset_dir, artifact_dir, prediction_dir, metadata


def test_plot_prediction_renders_a_png_with_explicit_trend_id(tmp_path):
    dataset_dir, artifact_dir, prediction_dir, metadata = _write_visualize_fixture(tmp_path)
    chosen = str(metadata.iloc[0]["trend_id"])
    result = plot_prediction(dataset_dir, prediction_dir, artifact_dir, tmp_path / "out",
                             split="inference", trend_id=chosen)
    output_path = Path(result["output_path"])
    assert output_path.is_file() and output_path.stat().st_size > 0
    assert result["trend_id"] == chosen


def test_plot_prediction_renders_a_png_with_maintenance_relation_elapsed(tmp_path):
    dataset_dir, artifact_dir, prediction_dir, metadata = _write_visualize_fixture(tmp_path)
    result = plot_prediction(dataset_dir, prediction_dir, artifact_dir, tmp_path / "out2",
                             split="inference", maintenance_relation="elapsed", seed=0)
    output_path = Path(result["output_path"])
    assert output_path.is_file() and output_path.stat().st_size > 0
    assert result["maintenance_relation"] == "elapsed"


# --- plot_prediction_batch(スモークテスト) ----------------------------------------------------

def _write_extra_prediction_dir(tmp_path, metadata, *, name, n_inference, seed):
    prediction_dir = tmp_path / name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=1.5, scale=0.2, size=(n_inference, 10, 12)).astype(np.float32)
    np.savez_compressed(
        prediction_dir / "samples.npz",
        trend_ids=metadata["trend_id"].astype(str).to_numpy().astype("U"), samples=samples,
    )
    return prediction_dir


def test_plot_prediction_batch_creates_one_folder_per_record_with_one_png_per_method(tmp_path):
    dataset_dir, artifact_dir, prediction_dir_a, metadata = _write_visualize_fixture(tmp_path, n_inference=3)
    prediction_dir_b = _write_extra_prediction_dir(tmp_path, metadata, name="method_b", n_inference=3, seed=1)

    result = plot_prediction_batch(
        dataset_dir, artifact_dir, tmp_path / "batch_out",
        methods={"method_a": str(prediction_dir_a), "method_b": str(prediction_dir_b)},
        split="inference", count_per_category=1, seed=0,
    )

    # fixtureのanchor_dates(2023-01-11/2023-07-20/2023-03-22)はそれぞれ
    # immediately_after 1件・elapsed 1件・どちらでもない1件、になる設計(他のテストと同じ前提)。
    assert result["record_count"] == 2
    assert set(result["methods"]) == {"method_a", "method_b"}
    for record in result["records"]:
        record_dir = tmp_path / "batch_out" / record["trend_id"]
        assert record_dir.is_dir()
        for method_name in ("method_a", "method_b"):
            output_path = Path(record["outputs"][method_name])
            assert output_path.parent == record_dir
            assert output_path.is_file() and output_path.stat().st_size > 0
