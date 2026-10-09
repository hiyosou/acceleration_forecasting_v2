"""datasets.history_search_dataset の検証(guide検索方法を6か月履歴ベースに差し替える、
2026-10-08、ユーザー判断による検証用データセット構築)。`test_self_reference.py`と同じ
「既存データセットの一部だけ差し替えて書き出す」系の結合テストの構成を踏襲する。
"""

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.datasets.build import ARRAY_NAMES
from acceleration_forecasting_v2.datasets.history_search_dataset import (
    SPLITS,
    build_history_search_dataset,
    build_history_search_inference_dataset,
)
from acceleration_forecasting_v2.datasets.normalization import Normalization
from acceleration_forecasting_v2.datasets.prepare import prepare_datasets
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from acceleration_forecasting_v2.datasets.verify import verify_dataset_leakage
from acceleration_forecasting_v2.retrieval.search import SearchConfig


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("history_search")
    artifact_dir, dataset_dir = root / "artifacts", root / "dataset"
    build_artifacts(artifact_dir, n_datasets=12, months=22, seed=0)
    prepare_datasets(artifact_dir, dataset_dir, device="cpu")
    return artifact_dir, dataset_dir


def test_model_train_and_model_validation_are_byte_identical(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search"
    build_history_search_inference_dataset(dataset_dir, artifact_dir, output)

    for split in ("model_train", "model_validation"):
        for name in ARRAY_NAMES:
            source = np.load(dataset_dir / split / f"{name}.npy")
            copy = np.load(output / split / f"{name}.npy")
            np.testing.assert_array_equal(source, copy)
        assert (output / split / "metadata.csv").read_text(encoding="utf-8") == \
            (dataset_dir / split / "metadata.csv").read_text(encoding="utf-8")


def test_normalization_files_are_reused_unchanged(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search"
    build_history_search_inference_dataset(dataset_dir, artifact_dir, output)

    for name in ("condition_normalization.json", "target_normalization.json"):
        assert (output / name).read_text(encoding="utf-8") == (dataset_dir / name).read_text(encoding="utf-8")


def test_inference_history_and_target_are_unchanged_only_guides_differ(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search"
    build_history_search_inference_dataset(dataset_dir, artifact_dir, output)

    for name in ("history_values", "history_masks", "segment_weights", "target_values", "target_masks"):
        source = np.load(dataset_dir / "inference" / f"{name}.npy")
        copy = np.load(output / "inference" / f"{name}.npy")
        np.testing.assert_array_equal(source, copy)

    source_guides = np.load(dataset_dir / "inference" / "guide_values.npy")
    new_guides = np.load(output / "inference" / "guide_values.npy")
    assert source_guides.shape == new_guides.shape
    for name in ARRAY_NAMES:
        assert (output / "inference" / f"{name}.npy").is_file()


def test_metadata_records_the_new_search_method_and_guide_provenance(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search"
    build_history_search_inference_dataset(dataset_dir, artifact_dir, output)

    frame = pd.read_csv(output / "inference" / "metadata.csv", encoding="utf-8-sig")
    assert (frame["guide_search_method"] == "history_rmse").all()
    for _, row in frame.iterrows():
        assert len(json.loads(row["guide_trend_ids"])) == row["guide_count"]
        # inferenceはmodel_train/model_validationからしかguideを取らない(自分自身のsplitは見ない)。
        assert set(json.loads(row["guide_model_splits"])) <= {"model_train", "model_validation"}


def test_guide_values_stay_within_search_config_top_k(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search"
    build_history_search_inference_dataset(
        dataset_dir, artifact_dir, output, search_config=SearchConfig(top_k=2, min_valid_months=1)
    )
    frame = pd.read_csv(output / "inference" / "metadata.csv", encoding="utf-8-sig")
    assert (frame["guide_count"] <= 2).all()


def test_forecast_dataset_can_load_the_output(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search"
    build_history_search_inference_dataset(dataset_dir, artifact_dir, output)

    dataset = ForecastDatasetV2(output / "inference", output, include_targets=False)
    item = dataset[0]
    assert item["guide_values"].shape == (3, 12)
    assert item["history_values"].shape == (6,)


def test_output_directory_must_differ_from_source(prepared):
    artifact_dir, dataset_dir = prepared
    with pytest.raises(ValueError, match="別ディレクトリ"):
        build_history_search_inference_dataset(dataset_dir, artifact_dir, dataset_dir)


def test_splits_constant_matches_prepare_splits():
    from acceleration_forecasting_v2.datasets.prepare import SPLITS as PREPARE_SPLITS

    assert set(SPLITS) == set(PREPARE_SPLITS)


# --- build_history_search_dataset(全split版、P3: 2026-10-09) ---


def test_all_splits_rebuild_changes_model_train_and_model_validation_guides(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search_all"
    build_history_search_dataset(dataset_dir, artifact_dir, output)

    for split in ("model_train", "model_validation", "inference"):
        frame = pd.read_csv(output / split / "metadata.csv", encoding="utf-8-sig")
        assert (frame["guide_search_method"] == "history_rmse").all()
        # model_train/model_validationはinferenceと違い自分自身のsplitからもguideを
        # 取ってよい(GUIDE_SOURCE_SPLITS)。
        allowed = {"model_train"} if split != "inference" else {"model_train", "model_validation"}
        for _, row in frame.iterrows():
            assert set(json.loads(row["guide_model_splits"])) <= allowed

    # guide_valuesの中身が元(embeddingベース検索)から実際に変わっていること
    # (全件一致だと検索方法が反映されていないバグの疑いがある)。
    source_guides = np.load(dataset_dir / "model_train" / "guide_values.npy")
    new_guides = np.load(output / "model_train" / "guide_values.npy")
    assert source_guides.shape == new_guides.shape
    finite_source = np.nan_to_num(source_guides)
    finite_new = np.nan_to_num(new_guides)
    assert not np.array_equal(finite_source, finite_new)


def test_all_splits_rebuild_refits_condition_normalization(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search_all"
    build_history_search_dataset(dataset_dir, artifact_dir, output)

    source_norm = Normalization.load(dataset_dir / "condition_normalization.json")
    new_norm = Normalization.load(output / "condition_normalization.json")
    # model_trainのguide分布が変わるため、再fitした統計は(偶然の一致を除き)元とは
    # 異なる値になる。target_normalizationはguideに依存しないため不変のはず。
    assert (new_norm.mean, new_norm.std, new_norm.count) != (source_norm.mean, source_norm.std, source_norm.count)
    source_target_norm = Normalization.load(dataset_dir / "target_normalization.json")
    new_target_norm = Normalization.load(output / "target_normalization.json")
    assert new_target_norm == source_target_norm


def test_inference_only_rebuild_keeps_normalization_byte_identical_to_source(prepared, tmp_path):
    """後方互換性の回帰確認: splits_to_rebuild=("inference",)(=
    build_history_search_inference_datasetの内部実装)はmodel_trainを変更しないため、
    正規化統計は元のデータセットと数値的に一致する。"""
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search_inference_only"
    build_history_search_dataset(dataset_dir, artifact_dir, output, splits_to_rebuild=("inference",))

    for name in ("condition_normalization.json", "target_normalization.json"):
        assert (output / name).read_text(encoding="utf-8") == (dataset_dir / name).read_text(encoding="utf-8")


def test_all_splits_rebuild_passes_leakage_verification(prepared, tmp_path):
    artifact_dir, dataset_dir = prepared
    output = tmp_path / "history_search_all"
    build_history_search_dataset(dataset_dir, artifact_dir, output)

    violations = verify_dataset_leakage(output)
    assert violations == []
