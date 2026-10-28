"""retrieval.waveform_history(6か月履歴の波形埋め込みによるguide検索、2026-10-20)の検証。"""

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.datasets.history import build_history_window
from acceleration_forecasting_v2.datasets.history_search_dataset import build_history_search_dataset
from acceleration_forecasting_v2.datasets.prepare import GUIDE_SOURCE_SPLITS, prepare_datasets
from acceleration_forecasting_v2.retrieval.database import connect_read_only
from acceleration_forecasting_v2.retrieval.search import GuideIndex, SearchConfig
from acceleration_forecasting_v2.retrieval.waveform_history import (
    build_waveform_history_index, history_trend_windows, trend_embeddings,
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("waveform_history")
    artifact_dir, dataset_dir = root / "artifacts", root / "dataset"
    build_artifacts(artifact_dir, n_datasets=12, months=22, seed=0)
    prepare_datasets(artifact_dir, dataset_dir, device="cpu")
    guide_index = GuideIndex(connect_read_only(artifact_dir / "vector_database.sqlite"))
    return artifact_dir, dataset_dir, build_waveform_history_index(guide_index, artifact_dir, device="cpu")


def test_trend_embedding_is_the_normalized_mean_of_its_runs():
    runs = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ids, embeddings = trend_embeddings(["b", "a", "b"], runs)
    assert list(ids) == ["a", "b"]
    np.testing.assert_allclose(embeddings[0], [0.0, 1.0])
    expected = np.array([1.0, 0.0]) + np.array([0.6, 0.8])
    np.testing.assert_allclose(embeddings[1], expected / np.linalg.norm(expected), rtol=1e-6)


def test_history_windows_follow_build_history_window(prepared):
    artifact_dir, _, _ = prepared
    catalog = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")
    windows = history_trend_windows(catalog)
    group = catalog.loc[catalog["dataset_id"] == catalog["dataset_id"].iloc[0]].copy()
    group["measurement_date"] = pd.to_datetime(group["measurement_date"])
    group = group.sort_values("measurement_date").rename(columns={"current_acc_z_max": "acc_z_max"})
    row = group.iloc[-1]
    _, mask, dates = build_history_window(group, row["measurement_date"])
    by_date = dict(zip(group["measurement_date"].dt.strftime("%Y-%m-%d"), group["trend_id"].astype(str)))
    expected = [by_date[date] if mask[k] > 0 else "" for k, date in enumerate(dates)]
    assert windows[str(row["trend_id"])] == expected
    assert windows[str(row["trend_id"])][-1] == str(row["trend_id"])  # 今月は自分自身


def test_every_trend_has_a_window_and_inference_trends_have_embeddings(prepared):
    artifact_dir, _, index = prepared
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig")
    inference_trend = str(manifest.loc[manifest["model_split"] == "inference", "trend_id"].iloc[0])
    _, mask = index.window(inference_trend)
    assert mask[-1]  # inferenceの記録もその場で埋め込まれている


def test_similarity_is_the_mean_cosine_over_shared_months(prepared):
    _, _, index = prepared
    query = str(index.guide_index.trend_ids[0])
    values, mask = index.window(query)
    rows = np.arange(len(index.guide_index.trend_ids))[:50]
    similarity, enough = index.similarities(query, rows, min_overlap=1)
    for position, row in enumerate(rows):
        other_values, other_mask = index.window(index.guide_index.trend_ids[row])
        both = mask & other_mask
        if both.any():
            expected = float((values[both] * other_values[both]).sum(1).mean())
            assert similarity[position] == pytest.approx(expected, abs=1e-5)


def test_search_respects_eligibility_and_ranks_by_similarity(prepared):
    _, dataset_dir, index = prepared
    metadata = pd.read_csv(dataset_dir / "inference" / "metadata.csv", encoding="utf-8-sig")
    row = metadata.iloc[0]
    guides = index.search(str(row["trend_id"]), query_date=str(row["anchor_date"]),
                          query_dataset_id=str(row["dataset_id"]), query_current=float(row["current_acc_z_max"]),
                          query_bin_start_m=float(row["bin_start_m"]),
                          allowed_splits=GUIDE_SOURCE_SPLITS["inference"], config=SearchConfig())
    assert 0 < len(guides) <= 3
    similarities = [guide["similarity"] for guide in guides]
    assert similarities == sorted(similarities, reverse=True)
    for guide in guides:
        assert guide["dataset_id"] != row["dataset_id"]
        assert guide["model_split"] in GUIDE_SOURCE_SPLITS["inference"]
        assert -1.0 - 1e-6 <= guide["similarity"] <= 1.0 + 1e-6


def test_dataset_builder_supports_the_waveform_history_method(prepared, tmp_path):
    artifact_dir, dataset_dir, _ = prepared
    output = tmp_path / "waveform_history"
    summary = build_history_search_dataset(dataset_dir, artifact_dir, output, search_method="waveform_history_cosine",
                                           device="cpu")
    assert summary["guide_search_method"] == "waveform_history_cosine"
    metadata = pd.read_csv(output / "inference" / "metadata.csv", encoding="utf-8-sig")
    assert (metadata["guide_search_method"] == "waveform_history_cosine").all()
    assert (metadata["guide_count"] > 0).all()
    guide_dataset_ids = metadata["guide_dataset_ids"].map(json.loads)
    assert all(row_id not in ids for row_id, ids in zip(metadata["dataset_id"], guide_dataset_ids))
    np.testing.assert_array_equal(np.load(output / "inference" / "target_values.npy"),
                                  np.load(dataset_dir / "inference" / "target_values.npy"))


def test_unknown_search_method_is_rejected(prepared, tmp_path):
    artifact_dir, dataset_dir, _ = prepared
    with pytest.raises(ValueError, match="search_method"):
        build_history_search_dataset(dataset_dir, artifact_dir, tmp_path / "x", search_method="cosine_today")
