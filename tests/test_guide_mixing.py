"""training.guide_mixing(学習時の混合guide・残差方式、手順2)と、train/predictへの組み込みの検証。"""

import numpy as np
import pandas as pd
import pytest
import torch

from synthetic_artifacts import build_artifacts
from test_training import _raw_split, _train_kwargs

from acceleration_forecasting_v2.datasets.build import write_dataset
from acceleration_forecasting_v2.datasets.mixed_guide import build_mixed_guide_dataset
from acceleration_forecasting_v2.datasets.normalization import Normalization
from acceleration_forecasting_v2.datasets.prepare import prepare_datasets
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from acceleration_forecasting_v2.inference.predict import predict
from acceleration_forecasting_v2.training.guide_mixing import (
    from_residual, mix_guide_batch, residual_base, residual_clip_bounds, sample_lambdas, to_residual,
)
from acceleration_forecasting_v2.training.train import train


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("guide_mixing")
    artifact_dir, dataset_dir = root / "artifacts", root / "dataset"
    build_artifacts(artifact_dir, n_datasets=12, months=22, seed=0)
    prepare_datasets(artifact_dir, dataset_dir, device="cpu")
    return dataset_dir


def _whole_split(dataset_dir, split="model_validation"):
    dataset = ForecastDatasetV2(dataset_dir / split, dataset_dir, include_targets=True)
    items = [dataset[index] for index in range(len(dataset))]
    return dataset, {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _norms(dataset):
    return {"condition_norm": dataset.condition_norm, "target_norm": dataset.target_norm}


# --- バッチ版とデータセット版(mask_mode="guide")の整合性 ---------------------------

def test_history_current_matches_metadata_current(prepared):
    # バッチ版はcurrentを履歴の最後の有効値から、データセット版はmetadataのcurrent_acc_z_maxから取る。
    dataset, batch = _whole_split(prepared)
    metadata = pd.read_csv(prepared / "model_validation" / "metadata.csv", encoding="utf-8-sig")
    history = np.load(prepared / "model_validation" / "history_values.npy")
    masks = np.load(prepared / "model_validation" / "history_masks.npy")
    last = np.array([row[mask > 0][-1] for row, mask in zip(history, masks)])
    np.testing.assert_allclose(last, metadata["current_acc_z_max"].to_numpy(), rtol=1e-6)


@pytest.mark.parametrize("lam", [0.3, 1.0])
def test_batch_mixing_matches_the_mixed_dataset(prepared, tmp_path, lam):
    output = tmp_path / "mixed"
    build_mixed_guide_dataset(prepared, output, lam, mask_mode="guide")
    dataset, batch = _whole_split(prepared)
    _, expected = _whole_split(output)
    mixed = mix_guide_batch(batch, torch.full((len(dataset),), lam), **_norms(dataset))
    torch.testing.assert_close(mixed["guide_mask"], expected["guide_mask"])
    valid = expected["guide_mask"] > 0
    torch.testing.assert_close(mixed["guide_values"][valid], expected["guide_values"][valid], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mixed["guide_deltas"][valid], expected["guide_deltas"][valid], atol=1e-4, rtol=1e-4)


# --- mix_guide_batch(単体) ---------------------------------------------------------

def _tiny_batch():
    """1記録・3スロット・12か月。正規化は恒等(mean=0, std=1)。スロット2は検索で見つからなかった空き。
    正解は12か月目が欠測、スロット0は3か月目が欠測。"""
    guide_values = torch.zeros(1, 3, 12)
    guide_values[0, 0] = torch.linspace(1.0, 2.1, 12)
    guide_values[0, 1] = torch.linspace(2.0, 3.1, 12)
    guide_mask = torch.ones(1, 3, 12)
    guide_mask[0, 0, 2] = 0
    guide_mask[0, 2] = 0
    guide_values[0, 0, 2] = 0
    guide_deltas = guide_values - torch.tensor([0.9, 1.9, 0.0])[None, :, None]
    guide_deltas = guide_deltas * guide_mask
    target = torch.linspace(3.0, 4.1, 12)[None]
    target_mask = torch.ones(1, 12)
    target_mask[0, 11] = 0
    target[0, 11] = 0
    history = torch.tensor([[2.0, 2.2, 2.4, 2.6, 2.8, 0.0]])
    history_masks = torch.tensor([[1.0, 1, 1, 1, 1, 0]])  # current = 2.8(最後の有効月)
    return {"guide_values": guide_values, "guide_deltas": guide_deltas, "guide_mask": guide_mask,
            "retrieval_mask": torch.tensor([[1.0, 1.0, 0.0]]), "guide_similarities": torch.zeros(1, 3),
            "target": target, "target_mask": target_mask, "history_values": history,
            "history_masks": history_masks}


IDENTITY = Normalization(0.0, 1.0, 2, "test")
NORMS = {"condition_norm": IDENTITY, "target_norm": IDENTITY}


def test_lambda_zero_leaves_the_batch_unchanged():
    batch = _tiny_batch()
    mixed = mix_guide_batch(batch, torch.zeros(1), **NORMS)
    for key in ("guide_values", "guide_deltas", "guide_mask"):
        torch.testing.assert_close(mixed[key], batch[key])


def test_lambda_one_gives_the_target_where_both_are_valid_and_keeps_the_rest():
    batch = _tiny_batch()
    mixed = mix_guide_batch(batch, torch.ones(1), **NORMS)
    both = torch.ones(12, dtype=torch.bool)
    both[[2, 11]] = False  # スロット0: 3か月目はguide欠測、12か月目は正解欠測
    torch.testing.assert_close(mixed["guide_values"][0, 0, both], batch["target"][0, both])
    torch.testing.assert_close(mixed["guide_deltas"][0, 0, both], batch["target"][0, both] - 2.8)
    # 正解欠測の月はguideのまま、マスクは元のまま、空きスロットは不変。
    assert mixed["guide_values"][0, 0, 11] == batch["guide_values"][0, 0, 11]
    assert mixed["guide_deltas"][0, 1, 11] == batch["guide_deltas"][0, 1, 11]
    torch.testing.assert_close(mixed["guide_mask"], batch["guide_mask"])
    torch.testing.assert_close(mixed["guide_values"][0, 2], batch["guide_values"][0, 2])


def test_sample_lambdas_has_the_requested_share_of_zeros():
    lam = sample_lambdas(20000, generator=torch.Generator().manual_seed(0), p_zero=0.5)
    assert 0.48 < float((lam == 0).float().mean()) < 0.52
    assert float(lam.max()) <= 1.0 and float(lam.min()) >= 0.0


# --- residual_base / to_residual / from_residual ------------------------------------

def test_residual_base_is_current_plus_mean_valid_delta():
    batch = _tiny_batch()
    base = residual_base(batch, **{"condition_norm": IDENTITY})
    deltas = batch["guide_deltas"][0]
    # 3か月目はスロット1だけ有効、それ以外はスロット0と1の平均(スロット2は空き)。
    torch.testing.assert_close(base[0, 2], 2.8 + deltas[1, 2])
    torch.testing.assert_close(base[0, 0], 2.8 + (deltas[0, 0] + deltas[1, 0]) / 2)


def test_residual_base_falls_back_to_record_mean_and_to_persistence():
    batch = _tiny_batch()
    batch["guide_mask"][0, :, 5] = 0  # 6か月目は全スロット無効
    base = residual_base(batch, condition_norm=IDENTITY)
    valid = (batch["guide_mask"][0] > 0) & (batch["retrieval_mask"][0][:, None] > 0)
    record_mean = batch["guide_deltas"][0][valid].mean()
    torch.testing.assert_close(base[0, 5], 2.8 + record_mean)
    batch["retrieval_mask"][:] = 0  # guideが1本も無い記録は現状維持
    torch.testing.assert_close(residual_base(batch, condition_norm=IDENTITY)[0], torch.full((12,), 2.8))


def test_residual_clip_bounds_map_to_the_physical_bounds():
    batch = _tiny_batch()
    residual = {"mean": 0.1, "std": 0.5}
    low, high = residual_clip_bounds(batch, residual, (0.1, 6.0), condition_norm=IDENTITY)
    torch.testing.assert_close(from_residual(low, batch, residual, condition_norm=IDENTITY), torch.full((1, 12), 0.1))
    torch.testing.assert_close(from_residual(high, batch, residual, condition_norm=IDENTITY), torch.full((1, 12), 6.0))


def test_residual_round_trip_recovers_the_target():
    batch = _tiny_batch()
    residual = {"mean": 0.1, "std": 0.5}
    converted = to_residual(batch, residual, **NORMS)
    restored = from_residual(converted["target"], batch, residual, condition_norm=IDENTITY)
    valid = batch["target_mask"] > 0
    torch.testing.assert_close(restored[valid], batch["target"][valid])
    assert float(converted["target"][0, 11]) == 0.0


# --- train / predict への組み込み ---------------------------------------------------

def _small_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_dataset({"model_train": _raw_split(32, seed=1), "model_validation": _raw_split(16, seed=2),
                   "inference": _raw_split(5, seed=3)}, dataset_dir)
    return dataset_dir


@pytest.mark.parametrize("target_mode", ["absolute", "residual"])
def test_train_with_mixing_records_config_and_lambda0_validation(tmp_path, target_mode):
    dataset_dir = _small_dataset(tmp_path)
    result = train(dataset_dir, tmp_path / "run", **_train_kwargs(
        epochs=2, guide_mix_probability_zero=0.5, target_mode=target_mode))
    config = result["model_config"]
    assert config["guide_mix_probability_zero"] == 0.5 and config["target_mode"] == target_mode
    assert ("residual_target" in config) == (target_mode == "residual")
    history = pd.read_csv(tmp_path / "run" / "training_history.csv")
    assert "validation_loss_lambda0" in history.columns and history["validation_loss_lambda0"].notna().all()


def test_default_train_config_has_no_new_keys(tmp_path):
    result = train(_small_dataset(tmp_path), tmp_path / "run", **_train_kwargs(epochs=1))
    assert "guide_mix_probability_zero" not in result["model_config"]
    assert "residual_target" not in result["model_config"]


def test_mixing_cannot_be_combined_with_self_target_loss(tmp_path):
    with pytest.raises(ValueError, match="self_target_loss_weight"):
        train(_small_dataset(tmp_path), tmp_path / "run", **_train_kwargs(
            epochs=1, guide_mix_probability_zero=0.5, self_target_loss_weight=1.0))


def test_predict_with_a_residual_checkpoint_returns_bounded_physical_values(tmp_path):
    dataset_dir = _small_dataset(tmp_path)
    train(dataset_dir, tmp_path / "run", **_train_kwargs(epochs=2, target_mode="residual"))
    summary = predict(dataset_dir, tmp_path / "run" / "best_model.pt", tmp_path / "pred", device="cpu",
                      num_samples=4, sampling_steps=5, start_timestep=900)
    assert summary["target_mode"] == "residual"
    samples = np.load(tmp_path / "pred" / "samples.npz")["samples"]
    assert samples.shape == (5, 4, 12)
    assert np.isfinite(samples).all() and samples.min() >= 0.1 and samples.max() <= 6.0
