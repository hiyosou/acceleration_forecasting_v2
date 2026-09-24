"""retrieval.database の単体テスト(SPEC.md 2.1のスキーマとの一致を確認)。"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from acceleration_forecasting_v2.retrieval.database import (
    TREND_COLUMNS,
    initialize_database,
    insert_trends,
    insert_waveform_record,
    store_metadata,
)


EXPECTED_TREND_COLUMNS = {
    "trend_id", "dataset_id", "model_split", "measurement_date", "direction",
    "bin_start_m", "bin_end_m", "current_acc_z_max", "future_values", "future_mask",
    "selected_dates", "guide_available_date", "cutoff_maintenance_date",
    "maintenance_type", "maintenance_description",
}
EXPECTED_WAVEFORM_COLUMNS = {
    "record_id", "measurement_id", "measurement_date", "direction", "bin_start_m",
    "bin_end_m", "mean_velocity_kmh", "source_csv_path", "waveform_sha256",
    "dataset_id", "trend_id", "embedding", "embedding_dim",
}


def _sample_trend_frame():
    return pd.DataFrame([{
        "trend_id": "trend-1", "dataset_id": "seg-1", "model_split": "model_train",
        "measurement_date": "2023-01-15", "direction": "U", "bin_start_m": 2000.0,
        "bin_end_m": 2100.0, "current_acc_z_max": 1.23,
        "future_values": "[1.0,null]", "future_mask": "[1,0]", "selected_dates": "[\"2023-02-01\",null]",
        "guide_available_date": "2024-02-01", "cutoff_maintenance_date": "",
        "maintenance_type": "", "maintenance_description": "",
    }])


def test_schema_table_columns_match_spec(tmp_path):
    connection = initialize_database(tmp_path / "db.sqlite")
    trend_columns = {row[1] for row in connection.execute("PRAGMA table_info(trends)").fetchall()}
    waveform_columns = {row[1] for row in connection.execute("PRAGMA table_info(waveform_records)").fetchall()}
    connection.close()
    assert trend_columns == EXPECTED_TREND_COLUMNS
    assert waveform_columns == EXPECTED_WAVEFORM_COLUMNS


def test_trend_columns_constant_matches_schema_order(tmp_path):
    assert set(TREND_COLUMNS) == EXPECTED_TREND_COLUMNS


def test_insert_and_query_roundtrip(tmp_path):
    connection = initialize_database(tmp_path / "db.sqlite")
    insert_trends(connection, _sample_trend_frame())
    embedding = np.arange(8, dtype=np.float32)
    insert_waveform_record(connection, {
        "record_id": "rec-1", "measurement_id": "meas-1", "measurement_date": "2023-01-15",
        "direction": "U", "bin_start_m": 2000.0, "bin_end_m": 2100.0, "mean_velocity_kmh": 60.0,
        "source_csv_path": "dummy.csv", "waveform_sha256": "abc", "dataset_id": "seg-1", "trend_id": "trend-1",
    }, embedding)
    store_metadata(connection, {"embedding_dim": 8, "note": "test"})
    connection.commit()

    trend_row = connection.execute("SELECT model_split, current_acc_z_max FROM trends WHERE trend_id='trend-1'").fetchone()
    waveform_row = connection.execute("SELECT embedding, embedding_dim FROM waveform_records WHERE record_id='rec-1'").fetchone()
    connection.close()

    assert trend_row == ("model_train", pytest.approx(1.23))
    restored = np.frombuffer(waveform_row[0], dtype=np.float32, count=waveform_row[1])
    np.testing.assert_array_equal(restored, embedding)


def test_waveform_record_requires_matching_trend_id(tmp_path):
    connection = initialize_database(tmp_path / "db.sqlite")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        insert_waveform_record(connection, {
            "record_id": "rec-orphan", "measurement_id": "meas-1", "measurement_date": "2023-01-15",
            "direction": "U", "bin_start_m": 2000.0, "bin_end_m": 2100.0, "mean_velocity_kmh": 60.0,
            "source_csv_path": "dummy.csv", "waveform_sha256": "abc", "dataset_id": "seg-1",
            "trend_id": "does-not-exist",
        }, np.zeros(4, dtype=np.float32))
        connection.commit()
    connection.close()
