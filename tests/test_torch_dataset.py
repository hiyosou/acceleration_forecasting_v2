"""datasets.torch_dataset.ForecastDatasetV2 の単体テスト(SPEC.md 1.6・2.1対応)。"""

import numpy as np
import pytest
import torch

from acceleration_forecasting_v2.datasets.build import ARRAY_NAMES
from acceleration_forecasting_v2.datasets.normalization import Normalization
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2


def _write_split(tmp_path, *, n=3, forecast_months=12, history_months=6, top_k=3):
    split_dir = tmp_path / "model_train"
    split_dir.mkdir()
    rng = np.random.default_rng(0)
    arrays = {
        "history_values": rng.normal(size=(n, history_months)).astype(np.float32),
        "history_masks": np.ones((n, history_months), dtype=np.float32),
        "guide_values": rng.normal(size=(n, top_k, forecast_months)).astype(np.float32),
        "guide_masks": np.ones((n, top_k, forecast_months), dtype=np.float32),
        "guide_deltas": rng.normal(size=(n, top_k, forecast_months)).astype(np.float32),
        "guide_similarities": rng.uniform(size=(n, top_k)).astype(np.float32),
        "retrieval_masks": np.ones((n, top_k), dtype=np.float32),
        "guide_baselines": rng.normal(size=(n, forecast_months)).astype(np.float32),
        "guide_softmax_weights": rng.uniform(size=(n, top_k, forecast_months)).astype(np.float32),
        "segment_weights": np.linspace(0.2, 1.0, n).astype(np.float32),
        "target_values": rng.normal(loc=2.0, size=(n, forecast_months)).astype(np.float32),
        "target_masks": np.ones((n, forecast_months), dtype=np.float32),
    }
    assert set(arrays.keys()) == set(ARRAY_NAMES)
    for name, values in arrays.items():
        np.save(split_dir / f"{name}.npy", values)

    condition_norm = Normalization(mean=0.0, std=1.0, count=n * history_months, source="test")
    target_norm = Normalization(mean=2.0, std=1.0, count=n * forecast_months, source="test")
    condition_norm.save(tmp_path / "condition_normalization.json")
    target_norm.save(tmp_path / "target_normalization.json")
    return split_dir, tmp_path, arrays


def test_dataset_length_matches_record_count(tmp_path):
    split_dir, root, arrays = _write_split(tmp_path, n=5)
    dataset = ForecastDatasetV2(split_dir, root)
    assert len(dataset) == 5


def test_dataset_item_keys_and_shapes(tmp_path):
    split_dir, root, arrays = _write_split(tmp_path, n=3)
    dataset = ForecastDatasetV2(split_dir, root)
    item = dataset[0]
    assert set(item.keys()) == {
        "history_values", "history_masks", "guide_values", "guide_deltas", "guide_mask",
        "guide_similarities", "retrieval_mask", "segment_weight", "index", "target", "target_mask",
    }
    assert item["history_values"].shape == (6,)
    assert item["guide_values"].shape == (3, 12)
    assert item["target"].shape == (12,)
    assert item["segment_weight"].dtype == torch.float32
    assert item["segment_weight"].ndim == 0


def test_dataset_segment_weight_matches_saved_array(tmp_path):
    split_dir, root, arrays = _write_split(tmp_path, n=3)
    dataset = ForecastDatasetV2(split_dir, root)
    for index in range(3):
        item = dataset[index]
        assert item["segment_weight"].item() == pytest.approx(float(arrays["segment_weights"][index]))


def test_dataset_no_nan_reaches_tensors_even_with_masked_missing_values(tmp_path):
    split_dir, root, arrays = _write_split(tmp_path, n=2)
    # 履歴の一部をNaN(欠測センチネル、SPEC.md 1.2)にして保存し直す。
    history = np.load(split_dir / "history_values.npy")
    history[0, 0] = np.nan
    np.save(split_dir / "history_values.npy", history)
    mask = np.load(split_dir / "history_masks.npy")
    mask[0, 0] = 0.0
    np.save(split_dir / "history_masks.npy", mask)

    dataset = ForecastDatasetV2(split_dir, root)
    item = dataset[0]
    assert torch.isfinite(item["history_values"]).all()
    assert item["history_masks"][0].item() == 0.0


def test_dataset_normalizes_history_and_target(tmp_path):
    split_dir, root, arrays = _write_split(tmp_path, n=1)
    dataset = ForecastDatasetV2(split_dir, root)
    item = dataset[0]
    # condition_norm(mean=0.0, std=1.0) -> historyの正規化値は生値とほぼ一致するはず。
    np.testing.assert_allclose(item["history_values"].numpy(), arrays["history_values"][0], atol=1e-5)
    # target_norm(mean=2.0, std=1.0) -> (生値-2.0)/1.0 になっているはず。
    expected_target = (arrays["target_values"][0] - 2.0) / 1.0
    np.testing.assert_allclose(item["target"].numpy(), expected_target, atol=1e-5)


def test_physical_prediction_does_not_add_guide_baseline(tmp_path):
    # target_mode=absoluteの決定(SPEC.md 1.6): guide_baselineを加算しない。
    split_dir, root, arrays = _write_split(tmp_path, n=1)
    dataset = ForecastDatasetV2(split_dir, root)
    normalized_zero = np.zeros(12, dtype=np.float32)
    physical = dataset.physical_prediction(normalized_zero)
    # target_norm(mean=2.0, std=1.0)でdenormalizeすると2.0になるはず(guide_baseline不加算)。
    np.testing.assert_allclose(physical, 2.0, atol=1e-5)


def test_physical_prediction_applies_clipping_bounds(tmp_path):
    split_dir, root, arrays = _write_split(tmp_path, n=1)
    dataset = ForecastDatasetV2(split_dir, root)
    normalized_large = np.full(12, 100.0, dtype=np.float32)
    physical = dataset.physical_prediction(normalized_large, bounds=(0.1, 6.0))
    np.testing.assert_allclose(physical, 6.0)


def test_include_targets_false_omits_target_keys(tmp_path):
    split_dir, root, arrays = _write_split(tmp_path, n=1)
    dataset = ForecastDatasetV2(split_dir, root, include_targets=False)
    item = dataset[0]
    assert "target" not in item
    assert "target_mask" not in item
    with pytest.raises(ValueError):
        dataset.physical_target(0)
