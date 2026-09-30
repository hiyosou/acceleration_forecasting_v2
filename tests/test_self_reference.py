"""datasets.self_reference の検証(自己参照guideデータセット構築、生成モジュールの上限性能測定用)。"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.datasets.build import ARRAY_NAMES
from acceleration_forecasting_v2.datasets.prepare import SPLITS, prepare_datasets
from acceleration_forecasting_v2.datasets.self_reference import (
    SPLITS as SELF_REFERENCE_SPLITS,
    _self_reference_arrays,
    build_self_reference_dataset,
    build_self_target_batch,
)
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from acceleration_forecasting_v2.datasets.verify import verify_dataset_leakage


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("self_reference")
    artifact_dir, dataset_dir = root / "artifacts", root / "dataset"
    build_artifacts(artifact_dir, n_datasets=12, months=22, seed=0)
    prepare_datasets(artifact_dir, dataset_dir, device="cpu")
    return dataset_dir


# --- _self_reference_arrays(単体) --------------------------------------------------

def test_only_slot_zero_is_filled_the_rest_stay_empty():
    target_values = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    target_masks = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)  # 3か月目は欠測
    current = np.array([0.5], dtype=np.float32)

    arrays = _self_reference_arrays(target_values, target_masks, current, top_k=3)

    np.testing.assert_allclose(arrays["guide_values"][0, 0], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(arrays["guide_masks"][0, 0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(arrays["guide_deltas"][0, 0], [0.5, 1.5, 0.0])
    assert arrays["guide_similarities"][0, 0] == 1.0
    assert arrays["retrieval_masks"][0, 0] == 1.0
    # スロット1・2は完全に空(無効)。
    for slot in (1, 2):
        np.testing.assert_array_equal(arrays["guide_values"][0, slot], np.zeros(3))
        np.testing.assert_array_equal(arrays["guide_masks"][0, slot], np.zeros(3))
        assert arrays["guide_similarities"][0, slot] == 0.0
        assert arrays["retrieval_masks"][0, slot] == 0.0


def test_guide_baseline_falls_back_to_current_where_target_is_invalid():
    target_values = np.array([[np.nan, 4.0]], dtype=np.float32)
    target_masks = np.array([[0.0, 1.0]], dtype=np.float32)
    current = np.array([2.0], dtype=np.float32)

    arrays = _self_reference_arrays(target_values, target_masks, current, top_k=3)
    np.testing.assert_allclose(arrays["guide_baselines"][0], [2.0, 4.0])


# --- build_self_reference_dataset(結合) --------------------------------------------

def test_output_reuses_history_and_target_unchanged(prepared, tmp_path):
    output = tmp_path / "self_reference"
    summary = build_self_reference_dataset(prepared, output)

    assert set(summary["counts"]) == set(SPLITS)
    for split in SPLITS:
        for name in ("history_values", "history_masks", "segment_weights", "target_values", "target_masks"):
            source = np.load(prepared / split / f"{name}.npy")
            copy = np.load(output / split / f"{name}.npy")
            np.testing.assert_array_equal(source, copy)
        for name in ARRAY_NAMES:
            assert (output / split / f"{name}.npy").is_file()

    for name in ("condition_normalization.json", "target_normalization.json"):
        assert (output / name).read_text(encoding="utf-8") == (prepared / name).read_text(encoding="utf-8")


def test_guide_values_equal_the_records_own_target(prepared, tmp_path):
    output = tmp_path / "self_reference"
    build_self_reference_dataset(prepared, output)

    for split in SPLITS:
        target_values = np.load(prepared / split / "target_values.npy")
        target_masks = np.load(prepared / split / "target_masks.npy")
        guide_values = np.load(output / split / "guide_values.npy")
        guide_masks = np.load(output / split / "guide_masks.npy")
        np.testing.assert_array_equal(guide_masks[:, 0], target_masks)
        np.testing.assert_array_equal(guide_values[:, 0][target_masks > 0], target_values[target_masks > 0])
        # スロット1・2は全レコードで無効。
        assert guide_masks[:, 1:].sum() == 0
        retrieval_masks = np.load(output / split / "retrieval_masks.npy")
        np.testing.assert_array_equal(retrieval_masks[:, 0], np.ones(len(retrieval_masks)))
        np.testing.assert_array_equal(retrieval_masks[:, 1:], np.zeros((len(retrieval_masks), 2)))


def test_metadata_reflects_self_as_the_guide_source(prepared, tmp_path):
    output = tmp_path / "self_reference"
    build_self_reference_dataset(prepared, output)

    frame = pd.read_csv(output / "model_validation" / "metadata.csv", encoding="utf-8-sig")
    for _, row in frame.iterrows():
        assert json.loads(row["guide_trend_ids"]) == [str(row["trend_id"])]
        assert json.loads(row["guide_dataset_ids"]) == [str(row["dataset_id"])]
    assert (frame["diagnostic_mode"] == "self_target_single_guide").all()
    assert (frame["guide_count"] == 1).all()


def test_verify_dataset_leakage_correctly_flags_the_intentional_self_reference(prepared, tmp_path):
    # このデータセットは意図的に「guideが自分自身のdataset_idから来ている」状態を作る。
    # verify_dataset_leakageは通常運用のデータセットには適用するチェックだが、
    # このデータセットに適用すればちゃんと違反として検出できることを確認しておく
    # (=メタデータの書き換えが正しく、検査自体が壊れていないことの裏付け)。
    output = tmp_path / "self_reference"
    build_self_reference_dataset(prepared, output)
    violations = verify_dataset_leakage(output)
    assert any("同一dataset_id" in text for text in violations)


def test_forecast_dataset_can_load_the_self_reference_output(prepared, tmp_path):
    output = tmp_path / "self_reference"
    build_self_reference_dataset(prepared, output)
    dataset = ForecastDatasetV2(output / "model_validation", output, include_targets=True)
    item = dataset[0]
    assert item["guide_values"].shape == (3, 12)
    assert item["guide_mask"][0].sum() > 0  # スロット0は有効
    assert item["guide_mask"][1:].sum() == 0  # スロット1・2は無効
    assert item["retrieval_mask"][0] == 1.0
    assert item["retrieval_mask"][1:].sum() == 0.0


def test_output_directory_must_differ_from_source(prepared):
    with pytest.raises(ValueError, match="別ディレクトリ"):
        build_self_reference_dataset(prepared, prepared)


def test_self_reference_splits_constant_matches_prepare_splits():
    assert set(SELF_REFERENCE_SPLITS) == set(SPLITS)


def test_build_self_target_batch_fills_only_slot_zero_with_condition_normalized_target(prepared):
    # inference/self_guide_diagnostics.pyの_build_self_target_batchが元々持っていたテストと
    # 同じアサーションを、公開移設先(datasets.self_reference.build_self_target_batch)に
    # 対して行う(公開APIそのものに直接カバレッジを持たせる)。
    dataset = ForecastDatasetV2(prepared / "model_validation", prepared, include_targets=True)
    batch = {key: (value.unsqueeze(0) if torch.is_tensor(value) else value) for key, value in dataset[0].items()}
    current_values = np.array([1.0], dtype=np.float32)

    self_target = build_self_target_batch(dataset, batch, current_values)
    expected_guide_values = dataset.condition_norm.normalize(dataset.denormalize_target(batch["target"].numpy()))
    np.testing.assert_allclose(self_target["guide_values"][:, 0].numpy(), expected_guide_values, atol=1e-4)
    np.testing.assert_array_equal(self_target["guide_mask"][:, 0].numpy(), batch["target_mask"].numpy())
    assert self_target["guide_mask"][:, 1:].sum() == 0
    assert self_target["retrieval_mask"][0, 0] == 1.0
    assert self_target["retrieval_mask"][0, 1:].sum() == 0
    assert self_target["guide_similarities"][0, 0] == 1.0
    assert torch.equal(self_target["history_values"], batch["history_values"])
