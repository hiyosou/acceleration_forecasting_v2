"""合成月の損失の重み(synthetic_month_weight、2026-11)のテスト。"""

import numpy as np
import pytest
import torch
from torch import nn

from acceleration_forecasting_v2.diffusion.process import DiffusionProcess
from acceleration_forecasting_v2.training.train import train
from test_training import _write_small_dataset


class _FixedModel(nn.Module):
    def __init__(self, predicted):
        super().__init__()
        self.predicted = predicted
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, noisy, timesteps, batch):
        return self.predicted * self.weight


def _setup():
    rng = np.random.default_rng(1)
    n = 3
    mask = torch.tensor([[1] * 12, [1] * 8 + [0] * 4, [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)
    synth = torch.tensor([[0] * 6 + [1] * 6, [0] * 8 + [0] * 4, [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)
    target = torch.tensor(rng.normal(size=(n, 12)), dtype=torch.float32)
    noise = torch.tensor(rng.normal(size=(n, 12)), dtype=torch.float32)
    predicted = torch.tensor(rng.normal(size=(n, 12)), dtype=torch.float32)
    process = DiffusionProcess(_FixedModel(predicted), steps=1000, prediction_type="epsilon")
    batch = {"target": target, "target_mask": mask, "target_synthetic_mask": synth}
    return process, batch, noise, predicted, mask, synth, torch.tensor([10, 500, 990])


def test_weight_one_is_identical_to_default():
    process, batch, noise, _, _, _, t = _setup()
    base = process.per_record_loss(batch, noise=noise, timesteps=t)
    same = process.per_record_loss(batch, noise=noise, timesteps=t, synthetic_weight=1.0)
    assert torch.equal(base, same)


def test_weight_zero_equals_real_months_only_mse():
    process, batch, noise, predicted, mask, synth, t = _setup()
    out = process.per_record_loss(batch, noise=noise, timesteps=t, synthetic_weight=0.0)
    real = mask * (1 - synth)
    err = (predicted - noise).square()
    expected = (err * real).sum(1) / real.sum(1).clamp_min(1e-6)
    assert torch.allclose(out, expected, atol=1e-6)


def test_intermediate_weight_is_weighted_mean():
    process, batch, noise, predicted, mask, synth, t = _setup()
    w = 0.3
    out = process.per_record_loss(batch, noise=noise, timesteps=t, synthetic_weight=w)
    mw = mask * (1 - (1 - w) * synth)
    err = (predicted - noise).square()
    assert torch.allclose(out, (err * mw).sum(1) / mw.sum(1), atol=1e-6)


def test_no_synthetic_mask_in_batch_ignores_weight():
    process, batch, noise, _, _, _, t = _setup()
    batch = {k: v for k, v in batch.items() if k != "target_synthetic_mask"}
    base = process.per_record_loss(batch, noise=noise, timesteps=t)
    assert torch.equal(base, process.per_record_loss(batch, noise=noise, timesteps=t, synthetic_weight=0.3))


def test_train_rejects_out_of_range_weight(tmp_path):
    dataset = _write_small_dataset(tmp_path)
    with pytest.raises(ValueError):
        train(dataset, tmp_path / "out", epochs=1, device="cpu", batch_size=8, progress=False, synthetic_month_weight=1.5)


def test_train_records_weight_in_config_and_default_adds_nothing(tmp_path):
    dataset = _write_small_dataset(tmp_path)
    a = train(dataset, tmp_path / "a", device="cpu", epochs=1, batch_size=8, progress=False)
    b = train(dataset, tmp_path / "b", device="cpu", epochs=1, batch_size=8, progress=False, synthetic_month_weight=0.3)
    assert "synthetic_month_weight" not in a["model_config"]
    assert b["model_config"]["synthetic_month_weight"] == 0.3
