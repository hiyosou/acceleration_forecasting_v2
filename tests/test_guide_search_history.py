"""retrieval.search.GuideIndex.search_by_history の単体テスト(2026-10-08、ユーザー判断)。

raw波形embeddingではなく6か月履歴ベクトルの類似度でguideを検索する代替パスの検証。
allowed_splits・同一dataset_id除外・空間/時間リーク制約は`search`と共通(`_eligibility`)
なので、ここでは履歴ベクトルの構築と類似度ランキング・重なり除外だけをテストする。
"""

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


def _trend_row(trend_id, dataset_id, model_split, date, *, current_acc_z_max=1.0,
               bin_start_m=2000.0, guide_available_date="2020-01-01"):
    future_values = [1.0] * 8 + [None] * 4
    future_mask = [1] * 8 + [0] * 4
    return {
        "trend_id": trend_id, "dataset_id": dataset_id, "model_split": model_split,
        "measurement_date": date, "direction": "U", "bin_start_m": bin_start_m, "bin_end_m": bin_start_m + 100.0,
        "current_acc_z_max": current_acc_z_max,
        "future_values": json.dumps(future_values),
        "future_mask": json.dumps(future_mask),
        "selected_dates": json.dumps([None] * 12),
        "guide_available_date": guide_available_date, "cutoff_maintenance_date": "",
        "maintenance_type": "", "maintenance_description": "",
    }


def _build_index(tmp_path, rows):
    connection = initialize_database(tmp_path / "guide.sqlite")
    insert_trends(connection, pd.DataFrame(rows))
    for row in rows:
        insert_waveform_record(connection, {
            "record_id": f"rec-{row['trend_id']}", "measurement_id": f"meas-{row['trend_id']}",
            "measurement_date": row["measurement_date"], "direction": row["direction"],
            "bin_start_m": row["bin_start_m"], "bin_end_m": row["bin_end_m"],
            "mean_velocity_kmh": 60.0, "source_csv_path": "dummy.csv", "waveform_sha256": "abc",
            "dataset_id": row["dataset_id"], "trend_id": row["trend_id"],
        }, np.ones(8, dtype=np.float32))
    connection.commit()
    connection.close()
    read_only = sqlite3.connect(str(tmp_path / "guide.sqlite"))
    return GuideIndex(read_only)


def _monthly_dataset(dataset_id, model_split, trend_id_prefix, start_date, values):
    """同一dataset_id内に、月1点ずつ(昇順)current_acc_z_maxを積んだ行を作る。"""
    dates = pd.date_range(start_date, periods=len(values), freq="MS")
    return [
        _trend_row(f"{trend_id_prefix}-{index}", dataset_id, model_split, date.strftime("%Y-%m-%d"),
                  current_acc_z_max=value)
        for index, (date, value) in enumerate(zip(dates, values))
    ]


def test_build_history_vectors_matches_build_history_window(tmp_path):
    from acceleration_forecasting_v2.datasets.history import build_history_window

    rows = _monthly_dataset("seg-a", "model_train", "t", "2023-01-01", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    index = _build_index(tmp_path, rows)

    # 最後の行(2023-06-01)の履歴は、anchors.pyが同じ入力に対して計算するものと一致するはず。
    section_history = pd.DataFrame({
        "measurement_date": pd.to_datetime([row["measurement_date"] for row in rows]),
        "acc_z_max": [row["current_acc_z_max"] for row in rows],
    })
    expected_values, expected_mask, _ = build_history_window(section_history, pd.Timestamp("2023-06-01"), months=6)

    last_row_position = list(index.trend_ids).index("t-5")
    np.testing.assert_allclose(index.history_values[last_row_position], expected_values)
    np.testing.assert_allclose(index.history_masks[last_row_position], expected_mask)


def test_build_history_vectors_caches_by_trend_id(tmp_path):
    # 同一trend_id(同日複数走行)を共有する複数のwaveform_recordは、同じ履歴ベクトルを
    # 持つはず(キャッシュ最適化が計算結果を変えていないことの確認)。
    connection = initialize_database(tmp_path / "guide.sqlite")
    rows = _monthly_dataset("seg-a", "model_train", "t", "2023-01-01", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    insert_trends(connection, pd.DataFrame(rows))
    last_row = rows[-1]
    for suffix in ("run1", "run2", "run3"):
        insert_waveform_record(connection, {
            "record_id": f"rec-{last_row['trend_id']}-{suffix}", "measurement_id": f"meas-{suffix}",
            "measurement_date": last_row["measurement_date"], "direction": last_row["direction"],
            "bin_start_m": last_row["bin_start_m"], "bin_end_m": last_row["bin_end_m"],
            "mean_velocity_kmh": 60.0, "source_csv_path": "dummy.csv", "waveform_sha256": "abc",
            "dataset_id": last_row["dataset_id"], "trend_id": last_row["trend_id"],
        }, np.ones(8, dtype=np.float32))
    connection.commit()
    connection.close()
    index = GuideIndex(sqlite3.connect(str(tmp_path / "guide.sqlite")))

    assert len(index.record_ids) == 3
    np.testing.assert_allclose(index.history_values[0], index.history_values[1])
    np.testing.assert_allclose(index.history_values[1], index.history_values[2])
    np.testing.assert_allclose(index.history_values[0], [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])


def test_search_by_history_ranks_closest_trajectory_first(tmp_path):
    # close/farは別dataset_idだが、`_select`は計測日の重複を除外するため、日付が被らないよう
    # 開始日をずらす(でないと片方がdate重複として捨てられてしまう)。
    query_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    close_rows = _monthly_dataset("seg-close", "model_train", "close", "2023-01-01",
                                  [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    far_rows = _monthly_dataset("seg-far", "model_train", "far", "2024-01-01",
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    index = _build_index(tmp_path, close_rows + far_rows)

    results = index.search_by_history(
        np.asarray(query_values, dtype=np.float32), np.ones(6, dtype=np.float32),
        query_date="2023-07-01", query_dataset_id="seg-query", query_current=2.0,
        query_bin_start_m=2000.0, allowed_splits={"model_train"},
        config=SearchConfig(min_valid_months=1, max_current_difference=None, min_history_overlap_months=6),
    )
    assert results[0]["dataset_id"] == "seg-close"
    assert results[0]["similarity"] > results[-1]["similarity"]
    # 完全一致なので負のRMSE=0。
    assert results[0]["similarity"] == 0.0


def test_search_by_history_excludes_same_dataset_id(tmp_path):
    query_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    same_rows = _monthly_dataset("seg-query", "model_train", "same", "2023-01-01", query_values)
    other_rows = _monthly_dataset("seg-other", "model_train", "other", "2023-01-01", query_values)
    index = _build_index(tmp_path, same_rows + other_rows)

    results = index.search_by_history(
        np.asarray(query_values, dtype=np.float32), np.ones(6, dtype=np.float32),
        query_date="2023-07-01", query_dataset_id="seg-query", query_current=2.0,
        query_bin_start_m=2000.0, allowed_splits={"model_train"},
        config=SearchConfig(min_valid_months=1, max_current_difference=None, min_history_overlap_months=6),
    )
    returned_datasets = {item["dataset_id"] for item in results}
    assert "seg-query" not in returned_datasets
    assert returned_datasets == {"seg-other"}


def test_search_by_history_excludes_insufficient_overlap(tmp_path):
    # 候補の履歴は先頭2か月しか有効でない(残りmask=0)ため、重なりがmin_history_overlap_months未満。
    rows = [
        _trend_row("t-0", "seg-a", "model_train", "2023-01-01", current_acc_z_max=1.0),
        _trend_row("t-1", "seg-a", "model_train", "2023-02-01", current_acc_z_max=1.0),
    ]
    index = _build_index(tmp_path, rows)

    query_values = np.full(6, 1.0, dtype=np.float32)
    query_mask = np.ones(6, dtype=np.float32)
    results = index.search_by_history(
        query_values, query_mask, query_date="2023-07-01", query_dataset_id="seg-query", query_current=1.0,
        query_bin_start_m=2000.0, allowed_splits={"model_train"},
        config=SearchConfig(min_valid_months=1, max_current_difference=None, min_history_overlap_months=3),
    )
    assert results == []


def test_search_by_history_respects_allowed_splits(tmp_path):
    query_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    train_rows = _monthly_dataset("seg-a", "model_train", "train", "2023-01-01", query_values)
    validation_rows = _monthly_dataset("seg-b", "model_validation", "valid", "2023-01-01", query_values)
    index = _build_index(tmp_path, train_rows + validation_rows)

    results = index.search_by_history(
        np.asarray(query_values, dtype=np.float32), np.ones(6, dtype=np.float32),
        query_date="2023-07-01", query_dataset_id="seg-query", query_current=2.0,
        query_bin_start_m=2000.0, allowed_splits={"model_train"},
        config=SearchConfig(min_valid_months=1, max_current_difference=None, min_history_overlap_months=6),
    )
    assert all(item["model_split"] == "model_train" for item in results)


def test_search_by_history_rejects_all_masked_query(tmp_path):
    import pytest

    rows = _monthly_dataset("seg-a", "model_train", "t", "2023-01-01", [1.0] * 6)
    index = _build_index(tmp_path, rows)

    # query_history_mask全てmask=0は検索クエリとして使えない。
    with pytest.raises(ValueError):
        index.search_by_history(
            np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32),
            query_date="2023-07-01", query_dataset_id="seg-query", query_current=1.0,
            query_bin_start_m=2000.0, allowed_splits={"model_train"},
        )
