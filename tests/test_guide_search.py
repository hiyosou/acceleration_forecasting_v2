"""retrieval.search.GuideIndex の単体テスト(SPEC.md 2.3のリーク対策チェック項目2〜4)。"""

import json
import sqlite3

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.retrieval.database import (
    initialize_database,
    insert_trends,
    insert_waveform_record,
)
from acceleration_forecasting_v2.retrieval.search import GuideIndex, SearchConfig


def _unit_vector(seed, dim=8):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _trend_row(trend_id, dataset_id, model_split, date, *, bin_start_m=2000.0, guide_available_date="2020-01-01"):
    future_values = [1.0] * 8 + [None] * 4
    future_mask = [1] * 8 + [0] * 4
    return {
        "trend_id": trend_id, "dataset_id": dataset_id, "model_split": model_split,
        "measurement_date": date, "direction": "U", "bin_start_m": bin_start_m, "bin_end_m": bin_start_m + 100.0,
        "current_acc_z_max": 1.0,
        "future_values": json.dumps(future_values),
        "future_mask": json.dumps(future_mask),
        "selected_dates": json.dumps([None] * 12),
        "guide_available_date": guide_available_date, "cutoff_maintenance_date": "",
        "maintenance_type": "", "maintenance_description": "",
    }


def _build_index(tmp_path, rows_and_embeddings):
    connection = initialize_database(tmp_path / "guide.sqlite")
    trend_frame = pd.DataFrame([row for row, _ in rows_and_embeddings])
    insert_trends(connection, trend_frame)
    for row, embedding in rows_and_embeddings:
        insert_waveform_record(connection, {
            "record_id": f"rec-{row['trend_id']}", "measurement_id": f"meas-{row['trend_id']}",
            "measurement_date": row["measurement_date"], "direction": row["direction"],
            "bin_start_m": row["bin_start_m"], "bin_end_m": row["bin_end_m"],
            "mean_velocity_kmh": 60.0, "source_csv_path": "dummy.csv", "waveform_sha256": "abc",
            "dataset_id": row["dataset_id"], "trend_id": row["trend_id"],
        }, embedding)
    connection.commit()
    connection.close()
    read_only = sqlite3.connect(str(tmp_path / "guide.sqlite"))
    return GuideIndex(read_only)


def test_search_excludes_same_dataset_id(tmp_path):
    query_embedding = _unit_vector(1)
    same_dataset = _trend_row("t-same", "seg-query", "model_train", "2023-01-01")
    other_dataset = _trend_row("t-other", "seg-guide", "model_train", "2023-02-01")
    index = _build_index(tmp_path, [(same_dataset, query_embedding), (other_dataset, _unit_vector(2))])

    results = index.search(
        query_embedding, query_date="2023-06-01", query_dataset_id="seg-query", query_current=1.0,
        query_bin_start_m=2000.0, allowed_splits={"model_train"}, config=SearchConfig(min_valid_months=1),
    )
    returned_datasets = {item["dataset_id"] for item in results}
    assert "seg-query" not in returned_datasets
    assert returned_datasets == {"seg-guide"}


def test_search_respects_allowed_splits(tmp_path):
    train_row = _trend_row("t-train", "seg-a", "model_train", "2023-01-01")
    validation_row = _trend_row("t-valid", "seg-b", "model_validation", "2023-01-01")
    index = _build_index(tmp_path, [(train_row, _unit_vector(3)), (validation_row, _unit_vector(4))])

    results = index.search(
        _unit_vector(3), query_date="2023-06-01", query_dataset_id="seg-query", query_current=1.0,
        query_bin_start_m=2000.0, allowed_splits={"model_train"}, config=SearchConfig(min_valid_months=1),
    )
    assert all(item["model_split"] == "model_train" for item in results)


def test_search_near_candidate_requires_earlier_availability(tmp_path):
    # 空間的に近い(距離差0)候補は available_date が query_date より前でなければならない。
    future_near = _trend_row("t-future-near", "seg-a", "model_train", "2023-01-01", bin_start_m=2000.0, guide_available_date="2030-01-01")
    past_far = _trend_row("t-past-far", "seg-b", "model_train", "2023-01-01", bin_start_m=9000.0, guide_available_date="2030-01-01")
    index = _build_index(tmp_path, [(future_near, _unit_vector(5)), (past_far, _unit_vector(6))])

    results = index.search(
        _unit_vector(5), query_date="2024-01-01", query_dataset_id="seg-query", query_current=1.0,
        query_bin_start_m=2000.0, allowed_splits={"model_train"}, config=SearchConfig(min_valid_months=1),
    )
    returned_ids = {item["trend_id"] for item in results}
    # near(距離0)なのに guide_available_date(2030) が query_date(2024) より後 -> 除外される。
    assert "t-future-near" not in returned_ids
    # far(距離7000m > near_distance_m)は時間制約を受けないため候補に残る。
    assert "t-past-far" in returned_ids
