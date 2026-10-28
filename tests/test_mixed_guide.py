"""datasets.mixed_guide の検証(実guideと正解をλで混ぜたguideデータセット、手順0の基準曲線用)。"""

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.datasets.baseline import softmax_guide_baseline
from acceleration_forecasting_v2.datasets.build import ARRAY_NAMES
from acceleration_forecasting_v2.datasets.mixed_guide import _mixed_guide_arrays, build_mixed_guide_dataset
from acceleration_forecasting_v2.datasets.prepare import SPLITS, prepare_datasets
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed_guide")
    artifact_dir, dataset_dir = root / "artifacts", root / "dataset"
    build_artifacts(artifact_dir, n_datasets=12, months=22, seed=0)
    prepare_datasets(artifact_dir, dataset_dir, device="cpu")
    return dataset_dir


def _one_record():
    """1レコード・3スロット・4か月。スロット0: current=1.0の実guide(3か月目は欠測)、
    スロット1: 検索で見つからなかった空きスロット、スロット2: current=2.0の実guide。
    正解は4か月目が欠測(補修で打ち切り)。"""
    nan = np.nan
    guide_values = np.array([[[1.2, 1.4, nan, 1.8], [0.0] * 4, [2.0, 2.2, 2.4, 2.6]]], np.float32)
    guide_masks = np.array([[[1, 1, 0, 1], [0] * 4, [1, 1, 1, 1]]], np.float32)
    guide_deltas = np.array([[[0.2, 0.4, nan, 0.8], [0.0] * 4, [0.0, 0.2, 0.4, 0.6]]], np.float32)
    similarities = np.array([[0.9, 0.0, 0.5]], np.float32)
    retrieval_masks = np.array([[1, 0, 1]], np.float32)
    target_values = np.array([[3.0, 3.2, 3.4, nan]], np.float32)
    target_masks = np.array([[1, 1, 1, 0]], np.float32)
    current = np.array([2.8], np.float32)
    return (guide_values, guide_masks, guide_deltas, similarities, retrieval_masks,
            target_values, target_masks, current)


# --- _mixed_guide_arrays(単体) ------------------------------------------------------

def test_lambda_zero_returns_the_original_guide_unchanged():
    inputs = _one_record()
    arrays = _mixed_guide_arrays(*inputs, 0.0)
    assert set(arrays) == {"guide_values", "guide_masks", "guide_deltas"}
    np.testing.assert_array_equal(arrays["guide_values"], inputs[0])
    np.testing.assert_array_equal(arrays["guide_masks"], inputs[1])
    np.testing.assert_array_equal(arrays["guide_deltas"], inputs[2])


def test_lambda_half_mixes_values_and_current_by_hand():
    arrays = _mixed_guide_arrays(*_one_record(), 0.5)
    # スロット0: 有効月 = guide有効(1,2,4か月目) ∧ 正解有効(1〜3か月目) = 1,2か月目。
    np.testing.assert_allclose(arrays["guide_masks"][0, 0], [1, 1, 0, 0])
    np.testing.assert_allclose(arrays["guide_values"][0, 0, :2], [2.1, 2.3], rtol=1e-6)
    assert np.isnan(arrays["guide_values"][0, 0, 2:]).all()
    # 混ぜた後のcurrent = 0.5*2.8 + 0.5*1.0 = 1.9 → deltas = values − 1.9。
    np.testing.assert_allclose(arrays["guide_deltas"][0, 0, :2], [0.2, 0.4], rtol=1e-5)
    assert np.isnan(arrays["guide_deltas"][0, 0, 2:]).all()
    # スロット2: 1〜3か月目が有効。current = 0.5*2.8 + 0.5*2.0 = 2.4。
    np.testing.assert_allclose(arrays["guide_masks"][0, 2], [1, 1, 1, 0])
    np.testing.assert_allclose(arrays["guide_values"][0, 2, :3], [2.5, 2.7, 2.9], rtol=1e-6)
    np.testing.assert_allclose(arrays["guide_deltas"][0, 2, :3], [0.1, 0.3, 0.5], atol=1e-6)


def test_lambda_one_gives_the_target_and_deltas_from_the_query_current():
    inputs = _one_record()
    arrays = _mixed_guide_arrays(*inputs, 1.0)
    target, current = inputs[5], inputs[7]
    for slot in (0, 2):
        valid = arrays["guide_masks"][0, slot] > 0
        np.testing.assert_allclose(arrays["guide_values"][0, slot][valid], target[0][valid], rtol=1e-6)
        np.testing.assert_allclose(arrays["guide_deltas"][0, slot][valid], target[0][valid] - current[0], atol=1e-6)


def test_empty_retrieval_slot_is_left_untouched():
    inputs = _one_record()
    arrays = _mixed_guide_arrays(*inputs, 0.5)
    np.testing.assert_array_equal(arrays["guide_values"][0, 1], inputs[0][0, 1])
    np.testing.assert_array_equal(arrays["guide_masks"][0, 1], inputs[1][0, 1])
    np.testing.assert_array_equal(arrays["guide_deltas"][0, 1], inputs[2][0, 1])


def test_guide_baselines_are_recomputed_from_the_mixed_guide():
    inputs = _one_record()
    arrays = _mixed_guide_arrays(*inputs, 0.75)
    expected, weights = softmax_guide_baseline(arrays["guide_values"][0], arrays["guide_masks"][0], inputs[3][0],
                                               inputs[4][0], inputs[7][0], 0.1)
    np.testing.assert_allclose(arrays["guide_baselines"][0], expected)
    np.testing.assert_allclose(arrays["guide_softmax_weights"][0], weights)


def test_guide_mask_mode_keeps_the_guide_mask_and_leaves_target_missing_months_unmixed():
    inputs = _one_record()
    arrays = _mixed_guide_arrays(*inputs, 1.0, mask_mode="guide")
    np.testing.assert_array_equal(arrays["guide_masks"], inputs[1])
    # 4か月目は正解が欠測なので、λ=1でもguideの値・deltasのまま。
    np.testing.assert_allclose(arrays["guide_values"][0, 0, [0, 1, 3]], [3.0, 3.2, 1.8], rtol=1e-6)
    np.testing.assert_allclose(arrays["guide_deltas"][0, 0, [0, 1, 3]], [0.2, 0.4, 0.8], atol=1e-6)
    np.testing.assert_allclose(arrays["guide_values"][0, 2], [3.0, 3.2, 3.4, 2.6], rtol=1e-6)
    assert np.isnan(arrays["guide_values"][0, 0, 2])


def test_unknown_mask_mode_is_rejected():
    with pytest.raises(ValueError, match="mask_mode"):
        _mixed_guide_arrays(*_one_record(), 0.5, mask_mode="union")


def test_lambda_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="lam"):
        _mixed_guide_arrays(*_one_record(), 1.5)


# --- build_mixed_guide_dataset(結合) ------------------------------------------------

def test_lambda_zero_dataset_is_identical_to_the_source(prepared, tmp_path):
    output = tmp_path / "lam000"
    build_mixed_guide_dataset(prepared, output, 0.0)
    for split in SPLITS:
        for name in ARRAY_NAMES:
            np.testing.assert_array_equal(np.load(prepared / split / f"{name}.npy"),
                                          np.load(output / split / f"{name}.npy"))


def test_non_guide_arrays_and_normalization_are_copied(prepared, tmp_path):
    output = tmp_path / "lam050"
    summary = build_mixed_guide_dataset(prepared, output, 0.5)
    assert set(summary["counts"]) == set(SPLITS)
    for split in SPLITS:
        for name in ("history_values", "history_masks", "segment_weights", "target_values", "target_masks",
                     "guide_similarities", "retrieval_masks"):
            np.testing.assert_array_equal(np.load(prepared / split / f"{name}.npy"),
                                          np.load(output / split / f"{name}.npy"))
    for name in ("condition_normalization.json", "target_normalization.json"):
        assert (output / name).read_text(encoding="utf-8") == (prepared / name).read_text(encoding="utf-8")
    frame = pd.read_csv(output / "inference" / "metadata.csv", encoding="utf-8-sig")
    assert (frame["diagnostic_mode"] == "mixed_guide").all()
    assert (frame["guide_mix_lambda"] == 0.5).all()
    assert json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))["guide_mix_lambda"] == 0.5


def test_guide_error_shrinks_linearly_with_lambda(prepared, tmp_path):
    errors = {}
    for lam in (0.0, 0.5, 1.0):
        output = tmp_path / f"lam{lam}"
        build_mixed_guide_dataset(prepared, output, lam)
        split_dir = output / "inference"
        values, masks = np.load(split_dir / "guide_values.npy"), np.load(split_dir / "guide_masks.npy")
        target, target_masks = np.load(split_dir / "target_values.npy"), np.load(split_dir / "target_masks.npy")
        valid = (masks > 0) & (target_masks[:, None, :] > 0)
        errors[lam] = float(np.abs(values - target[:, None, :])[valid].mean())
    assert errors[1.0] == pytest.approx(0.0, abs=1e-6)
    assert errors[0.5] == pytest.approx(errors[0.0] / 2, rel=1e-4)


def test_forecast_dataset_can_load_the_mixed_output(prepared, tmp_path):
    output = tmp_path / "lam025"
    build_mixed_guide_dataset(prepared, output, 0.25)
    item = ForecastDatasetV2(output / "inference", output, include_targets=True)[0]
    for key in ("guide_values", "guide_deltas", "guide_mask"):
        assert item[key].shape == (3, 12)
        assert bool(item[key].isfinite().all())


def test_output_directory_must_differ_from_source(prepared):
    with pytest.raises(ValueError, match="別ディレクトリ"):
        build_mixed_guide_dataset(prepared, prepared, 0.5)
