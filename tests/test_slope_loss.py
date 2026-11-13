"""損失への変化量・傾きの補助項(slope_loss_weight、2026-11 案A)のテスト。"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from acceleration_forecasting_v2.cli import main
from acceleration_forecasting_v2.diffusion.process import DiffusionProcess
from acceleration_forecasting_v2.training.train import train
from test_training import _write_small_dataset


class _ScaledNoiseModel(nn.Module):
    """ε̂ = 固定のテンソル(batchごとの行)を返す。手計算で補助項を確かめるためのモデル。"""

    def __init__(self, predicted):
        super().__init__()
        self.predicted = predicted
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, noisy, timesteps, batch):
        return self.predicted * self.weight


def _batch(target, mask):
    return {"target": target, "target_mask": mask}


def _reference(predicted, noise, mask, alpha_bar):
    """numpyでの手計算: x̂0 − x0 = −sqrt(1−ᾱ)/sqrt(ᾱ)·(ε̂−ε) を実際に作り、ᾱ·(…)で評価する。"""
    out = []
    for p, n, m, a in zip(predicted, noise, mask, alpha_bar):
        e = -np.sqrt(1 - a) / np.sqrt(a) * (p - n)  # x̂0 − x0
        valid = np.flatnonzero(m > 0)
        last3 = valid[-3:]
        level = e[last3].mean().__pow__(2) if len(last3) else 0.0
        pairs = [i for i in range(1, len(m)) if m[i] > 0 and m[i - 1] > 0]
        shape = np.mean([(e[i] - e[i - 1]) ** 2 for i in pairs]) if pairs else 0.0
        out.append(a * (level + shape))
    return np.array(out)


def _process_with(predicted, steps=1000):
    return DiffusionProcess(_ScaledNoiseModel(predicted), steps=steps, prediction_type="epsilon")


@pytest.mark.parametrize("mask_rows", [
    [[1] * 12, [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # 穴あり・1か月だけ
    [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],  # 3か月未満・隣接ペアなし
])
def test_slope_term_matches_the_numpy_reference(mask_rows):
    rng = np.random.default_rng(0)
    n = len(mask_rows)
    mask = torch.tensor(mask_rows, dtype=torch.float32)
    target = torch.tensor(rng.normal(size=(n, 12)), dtype=torch.float32)
    noise = torch.tensor(rng.normal(size=(n, 12)), dtype=torch.float32)
    predicted = torch.tensor(rng.normal(size=(n, 12)), dtype=torch.float32)
    timesteps = torch.tensor([10, 500, 990][:n])
    process = _process_with(predicted)
    mse, slope = process.per_record_loss_with_slope(_batch(target, mask), noise=noise, timesteps=timesteps)
    alpha = process.alpha_bars[timesteps].numpy()
    expected = _reference(predicted.numpy(), noise.numpy(), mask.numpy(), alpha)
    np.testing.assert_allclose(slope.detach().numpy(), expected, rtol=1e-4, atol=1e-7)
    # MSEは従来のper_record_lossと同じ
    np.testing.assert_array_equal(mse.detach().numpy(),
                                  process.per_record_loss(_batch(target, mask), noise=noise, timesteps=timesteps).detach().numpy())


def test_slope_term_is_finite_and_grows_with_noise_level():
    n = 12
    predicted = torch.ones(1, n)
    noise = torch.zeros(1, n)
    mask = torch.ones(1, n)
    target = torch.zeros(1, n)
    values = []
    for t in (0, 100, 500, 999):
        process = _process_with(predicted)
        _, slope = process.per_record_loss_with_slope(_batch(target, mask), noise=noise, timesteps=torch.tensor([t]))
        assert torch.isfinite(slope).all()
        values.append(float(slope))
    # 一定の誤差のとき、変化量の項は (1−ᾱ)·誤差² で、tが大きいほど大きい。傾きの項は0。
    assert values == sorted(values)
    assert values[0] < 1e-3 < values[-1]


def test_per_record_loss_is_unchanged_by_the_refactor():
    rng = np.random.default_rng(1)
    predicted = torch.tensor(rng.normal(size=(4, 12)), dtype=torch.float32)
    noise = torch.tensor(rng.normal(size=(4, 12)), dtype=torch.float32)
    target = torch.tensor(rng.normal(size=(4, 12)), dtype=torch.float32)
    mask = torch.tensor(rng.random((4, 12)) > 0.3, dtype=torch.float32)
    timesteps = torch.tensor([3, 300, 600, 900])
    process = _process_with(predicted)
    alpha = process.alpha_bars[timesteps][:, None]
    expected = ((predicted - noise).square() * mask).sum(1) / mask.sum(1).clamp_min(1)
    got = process.per_record_loss(_batch(target, mask), noise=noise, timesteps=timesteps)
    assert torch.equal(got, expected)
    assert alpha.shape == (4, 1)


def test_slope_term_rejects_non_epsilon_prediction():
    process = DiffusionProcess(_ScaledNoiseModel(torch.zeros(1, 12)), steps=1000, prediction_type="v_prediction")
    with pytest.raises(ValueError, match="epsilon"):
        process.per_record_loss_with_slope(_batch(torch.zeros(1, 12), torch.ones(1, 12)))


def _history(root):
    return pd.read_csv(root / "training_history.csv", encoding="utf-8-sig")


def test_zero_weight_training_matches_unspecified_and_adds_no_column(tmp_path):
    dataset = _write_small_dataset(tmp_path)
    a = train(dataset, tmp_path / "a", device="cpu", epochs=2, batch_size=8, progress=False)
    b = train(dataset, tmp_path / "b", device="cpu", epochs=2, batch_size=8, progress=False, slope_loss_weight=0.0)
    for key in a["model_config"]:
        assert a["model_config"][key] == b["model_config"][key]
    assert "slope_loss_weight" not in b["model_config"]
    assert "train_slope_loss" not in _history(tmp_path / "b").columns
    ca = torch.load(tmp_path / "a" / "best_model.pt", map_location="cpu", weights_only=False)
    cb = torch.load(tmp_path / "b" / "best_model.pt", map_location="cpu", weights_only=False)
    for key, value in ca["ema_state_dict"].items():
        assert torch.equal(value, cb["ema_state_dict"][key])


def test_slope_training_records_config_column_and_loss_goes_down(tmp_path):
    dataset = _write_small_dataset(tmp_path, n_train=64)
    # tとノイズが毎エポック違うので、補助項は1エポックごとには単調でない。最初の3エポックと最後の3エポックの平均で比べる。
    result = train(dataset, tmp_path / "s", device="cpu", epochs=12, batch_size=8, progress=False, patience=99,
                   slope_loss_weight=0.3)
    assert result["model_config"]["slope_loss_weight"] == 0.3
    history = _history(tmp_path / "s")
    assert np.isfinite(history["train_slope_loss"]).all()
    assert np.isfinite(history["validation_loss"]).all()
    assert history["train_slope_loss"].iloc[-3:].mean() < history["train_slope_loss"].iloc[:3].mean()
    # 検証損失は補助項を含まない(εのMSEだけ)
    plain = train(dataset, tmp_path / "p", device="cpu", epochs=1, batch_size=8, progress=False)
    assert plain["model_config"].get("slope_loss_weight") is None


def test_slope_training_resumes_and_rejects_a_changed_weight(tmp_path):
    dataset = _write_small_dataset(tmp_path)
    out = tmp_path / "r"
    train(dataset, out, device="cpu", epochs=1, batch_size=8, progress=False, slope_loss_weight=0.3)
    train(dataset, out, device="cpu", epochs=2, batch_size=8, progress=False, slope_loss_weight=0.3)
    assert len(_history(out)) == 2
    with pytest.raises(Exception):
        train(dataset, out, device="cpu", epochs=3, batch_size=8, progress=False, slope_loss_weight=1.0)


@pytest.mark.parametrize("kwargs", [
    {"guide_mix_probability_zero": 0.5},
    {"target_mode": "residual"},
    {"self_target_loss_weight": 0.1},
    {"prediction_type": "v_prediction"},
    {"slope_loss_weight_negative": True},
])
def test_slope_training_rejects_incompatible_options(tmp_path, kwargs):
    dataset = _write_small_dataset(tmp_path)
    weight = -0.1 if kwargs.pop("slope_loss_weight_negative", False) else 0.3
    with pytest.raises(ValueError):
        train(dataset, tmp_path / "x", device="cpu", epochs=1, batch_size=8, progress=False,
              slope_loss_weight=weight, **kwargs)


def test_cli_train_with_slope_loss_weight(tmp_path, capsys):
    dataset = _write_small_dataset(tmp_path)
    assert main(["train", "--dataset-dir", str(dataset), "--output-dir", str(tmp_path / "cli"), "--epochs", "3",
                 "--batch-size", "8", "--device", "cpu", "--no-progress", "--slope-loss-weight", "0.3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model_config"]["slope_loss_weight"] == 0.3
    assert "train_slope_loss" in _history(tmp_path / "cli").columns
