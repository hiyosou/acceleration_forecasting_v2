"""energy scoreでの微調整(training/energy.py、DiffusionProcess.ddim_differentiable、2026-11 案B)のテスト。"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from acceleration_forecasting_v2.cli import main
from acceleration_forecasting_v2.diffusion.process import DiffusionProcess
from acceleration_forecasting_v2.inference.predict import load_process, predict
from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2
from acceleration_forecasting_v2.training.energy import energy_score, finetune_energy
from acceleration_forecasting_v2.training.train import train
from test_diffusion_process import _dummy_batch
from test_training import _write_small_dataset


def _reference_energy(samples, target, mask):
    out = []
    for x, y, m in zip(samples, target, mask):
        n = max(m.sum(), 1.0)

        def dist(a, b):
            return np.sqrt(((a - b) ** 2 * m).sum() / n + 1e-8)

        k = len(x)
        first = np.mean([dist(x[i], y) for i in range(k)])
        pairs = [dist(x[i], x[j]) for i in range(k) for j in range(k) if i != j]
        second = np.mean(pairs) if pairs else 0.0
        out.append((first - 0.5 * second) if m.sum() > 0 else 0.0)
    return np.array(out)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_energy_score_matches_the_numpy_reference(k):
    rng = np.random.default_rng(k)
    samples = rng.normal(size=(4, k, 12)).astype(np.float32)
    target = rng.normal(size=(4, 12)).astype(np.float32)
    mask = np.array([[1] * 12, [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1], [1] + [0] * 11, [0] * 12], dtype=np.float32)
    got = energy_score(torch.tensor(samples), torch.tensor(target), torch.tensor(mask)).numpy()
    np.testing.assert_allclose(got, _reference_energy(samples, target, mask), rtol=1e-4, atol=1e-6)


def test_energy_score_masked_months_do_not_matter():
    rng = np.random.default_rng(0)
    samples = torch.tensor(rng.normal(size=(2, 3, 12)), dtype=torch.float32)
    target = torch.tensor(rng.normal(size=(2, 12)), dtype=torch.float32)
    mask = torch.tensor([[1] * 6 + [0] * 6, [0, 1] * 6], dtype=torch.float32)
    changed = samples.clone()
    changed[:, :, mask[0] == 0] += 5.0  # 同じ月を全サンプルでずらしても、無効月なら変わらない(1件目のみ確認)
    a = energy_score(samples, target, mask)
    b = energy_score(changed[:1], target[:1], mask[:1])
    torch.testing.assert_close(a[:1], b)


def test_energy_score_is_zero_for_perfect_samples_and_spread_lowers_it():
    target = torch.linspace(-1, 1, 12)[None, :]
    mask = torch.ones(1, 12)
    perfect = target[:, None, :].repeat(1, 4, 1)
    assert float(energy_score(perfect, target, mask)) < 1e-3
    # 正解からずれた2サンプル: 同じ点に潰した場合より、正解を挟んで広がった場合の方が小さい(広がりの項は引かれる)。
    delta = 0.5 * torch.ones(1, 1, 12)
    collapsed = torch.cat([target[:, None, :] + delta, target[:, None, :] + delta], dim=1)
    spread = torch.cat([target[:, None, :] + delta, target[:, None, :] - delta], dim=1)
    assert float(energy_score(spread, target, mask)) < float(energy_score(collapsed, target, mask)) - 0.1
    # 遠いサンプルは、近いサンプルより悪い。
    far = target[:, None, :] + 3 * delta
    assert float(energy_score(far, target, mask)) > float(energy_score(collapsed, target, mask))


def test_energy_score_has_finite_gradient_even_for_identical_samples():
    target = torch.zeros(2, 12)
    samples = torch.zeros(2, 4, 12, requires_grad=True)
    energy_score(samples, target, torch.ones(2, 12)).sum().backward()
    assert torch.isfinite(samples.grad).all()


def test_energy_score_rejects_bad_shapes():
    with pytest.raises(ValueError):
        energy_score(torch.zeros(2, 12), torch.zeros(2, 12), torch.ones(2, 12))


def test_ddim_differentiable_matches_ddim_with_the_same_initial_noise():
    torch.manual_seed(0)
    model = ReferenceModulatedUNetV2().eval()
    process = DiffusionProcess(model, steps=1000, prediction_type="epsilon")
    batch = {key: torch.randn_like(value) if value.dtype.is_floating_point and "mask" not in key else value
             for key, value in _dummy_batch(3).items()}
    for clip in (None, (-1.5, 1.5)):
        generator = torch.Generator().manual_seed(7)
        expected = process.ddim(batch, shape=(3, 12), sampling_steps=5, normalized_clip=clip, generator=generator,
                                start_timestep=958)
        noise = torch.randn((3, 12), generator=torch.Generator().manual_seed(7))
        got = process.ddim_differentiable(batch, shape=(3, 12), sampling_steps=5, normalized_clip=clip,
                                          start_timestep=958, noise=noise)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


def test_ddim_differentiable_passes_gradients_to_the_model_and_validates_arguments():
    model = ReferenceModulatedUNetV2().eval()
    process = DiffusionProcess(model, steps=1000, prediction_type="epsilon")
    batch = _dummy_batch(2)
    noise = torch.randn(2, 12)
    values = process.ddim_differentiable(batch, shape=(2, 12), sampling_steps=4, normalized_clip=(-3.0, 3.0),
                                         noise=noise)
    values.square().sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(float(g.abs().sum()) > 0 for g in grads)
    with pytest.raises(ValueError, match="noise"):
        process.ddim_differentiable(batch, shape=(2, 12), sampling_steps=4)
    with pytest.raises(ValueError, match="sampling_steps"):
        process.ddim_differentiable(batch, shape=(2, 12), sampling_steps=0, noise=noise)
    with pytest.raises(ValueError, match="start_timestep"):
        process.ddim_differentiable(batch, shape=(2, 12), sampling_steps=4, start_timestep=1000, noise=noise)


@pytest.fixture(scope="module")
def initial(tmp_path_factory):
    root = tmp_path_factory.mktemp("energy")
    dataset = _write_small_dataset(root, n_train=32, n_valid=16)
    train(dataset, root / "init", device="cpu", epochs=1, batch_size=8, progress=False, accumulation_steps=1)
    return {"root": root, "dataset": dataset, "checkpoint": root / "init" / "best_model.pt"}


def _small(**overrides):
    kwargs = dict(device="cpu", progress=False, batch_size=8, accumulation_steps=2, energy_batch=4,
                  energy_samples=2, energy_steps=2, validation_records=8, validation_samples=2, max_epochs=2,
                  patience=5, learning_rate=1e-4)
    kwargs.update(overrides)
    return kwargs


def _history(path):
    return pd.read_csv(path / "training_history.csv", encoding="utf-8-sig")


def test_finetune_energy_writes_history_checkpoints_and_config(initial):
    out = initial["root"] / "ft"
    result = finetune_energy(initial["dataset"], out, init_checkpoint=initial["checkpoint"], energy_weight=1.0,
                             **_small())
    history = _history(out)
    assert list(history.columns) == ["epoch", "train_loss_eps", "train_energy", "validation_loss",
                                     "validation_energy", "best_validation_energy", "elapsed_seconds"]
    assert history["epoch"].tolist() == [0, 1, 2]
    assert np.isfinite(history[["validation_loss", "validation_energy"]].to_numpy()).all()
    assert np.isfinite(history.loc[1:, ["train_loss_eps", "train_energy", "best_validation_energy"]].to_numpy()).all()
    for name in ("best_model.pt", "last_model.pt", "resolved_config.json"):
        assert (out / name).is_file()
    config = json.loads((out / "resolved_config.json").read_text(encoding="utf-8"))
    assert config["model_config"]["energy_finetune"]["weight"] == 1.0
    assert config["model_config"]["energy_finetune"]["steps"] == 2
    assert result["best_epoch"] in (1, 2) and result["epochs_completed"] == 2
    # 微調整後のcheckpointをpredictがそのまま読める
    process, loaded = load_process(out / "best_model.pt", torch.device("cpu"))
    assert loaded["prediction_type"] == "epsilon" and "energy_finetune" in loaded
    summary = predict(initial["dataset"], out / "best_model.pt", out / "pred", split="model_validation",
                      device="cpu", num_samples=2, sampling_steps=3, batch_size=4)
    assert summary["record_count"] == 16
    # 初期値のEMAと、微調整後のEMAは違う(重みが動いた)
    init = torch.load(initial["checkpoint"], map_location="cpu", weights_only=False)["ema_state_dict"]
    tuned = torch.load(out / "last_model.pt", map_location="cpu", weights_only=False)["ema_state_dict"]
    assert any(not torch.equal(init[key], tuned[key]) for key in init)


def test_finetune_energy_starts_from_the_init_ema_weights(initial):
    out = initial["root"] / "ft_zero_epochs"
    finetune_energy(initial["dataset"], out, init_checkpoint=initial["checkpoint"], energy_weight=1.0,
                    **_small(max_epochs=0))
    init = torch.load(initial["checkpoint"], map_location="cpu", weights_only=False)["ema_state_dict"]
    for name in ("last_model.pt", "best_model.pt"):
        checkpoint = torch.load(out / name, map_location="cpu", weights_only=False)
        for key, value in init.items():
            assert torch.equal(checkpoint["ema_state_dict"][key], value)
            assert torch.equal(checkpoint["model_state_dict"][key], value)
    assert _history(out)["epoch"].tolist() == [0]


def test_finetune_energy_resumes_and_matches_an_uninterrupted_run(initial):
    full = initial["root"] / "ft_full"
    finetune_energy(initial["dataset"], full, init_checkpoint=initial["checkpoint"], energy_weight=0.5,
                    **_small(max_epochs=2))
    part = initial["root"] / "ft_part"
    finetune_energy(initial["dataset"], part, init_checkpoint=initial["checkpoint"], energy_weight=0.5,
                    **_small(max_epochs=1))
    assert len(_history(part)) == 2
    finetune_energy(initial["dataset"], part, init_checkpoint=initial["checkpoint"], energy_weight=0.5,
                    **_small(max_epochs=2))
    history = _history(part)
    assert history["epoch"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(history["validation_energy"], _history(full)["validation_energy"], rtol=1e-5)
    # 設定が変わったときは再開を拒否する
    with pytest.raises(ValueError):
        finetune_energy(initial["dataset"], part, init_checkpoint=initial["checkpoint"], energy_weight=1.0,
                        **_small(max_epochs=3))


def test_finetune_energy_zero_weight_is_the_control(initial):
    out = initial["root"] / "ft_control"
    result = finetune_energy(initial["dataset"], out, init_checkpoint=initial["checkpoint"], energy_weight=0.0,
                             **_small())
    history = _history(out)
    assert history["train_energy"].isna().all()
    assert np.isfinite(history["train_loss_eps"].iloc[1:]).all()
    assert np.isfinite(history["validation_energy"]).all()
    assert result["model_config"]["energy_finetune"]["weight"] == 0.0


def test_finetune_energy_early_stops_with_patience(initial):
    out = initial["root"] / "ft_patience"
    # 学習率0なら重みが動かず、検証ESはエポック1から改善しない(同値)→ patience 1でエポック2で止まる
    finetune_energy(initial["dataset"], out, init_checkpoint=initial["checkpoint"], energy_weight=1.0,
                    **_small(max_epochs=10, patience=1, learning_rate=0.0))
    assert _history(out)["epoch"].tolist() == [0, 1, 2]


def test_finetune_energy_rejects_bad_arguments(initial):
    with pytest.raises(ValueError):
        finetune_energy(initial["dataset"], initial["root"] / "bad", init_checkpoint=initial["checkpoint"],
                        energy_weight=-1.0, **_small())
    with pytest.raises(ValueError):
        finetune_energy(initial["dataset"], initial["root"] / "bad", init_checkpoint=initial["checkpoint"],
                        energy_weight=1.0, **_small(energy_samples=0))


def test_cli_finetune_energy_smoke(initial, capsys):
    out = initial["root"] / "ft_cli"
    assert main(["finetune-energy", "--dataset-dir", str(initial["dataset"]), "--output-dir", str(out),
                 "--init-checkpoint", str(initial["checkpoint"]), "--energy-weight", "0.3", "--energy-batch", "4",
                 "--energy-samples", "2", "--energy-steps", "2", "--max-epochs", "1", "--validation-records", "8",
                 "--validation-samples", "2", "--device", "cpu", "--no-progress"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_config"]["energy_finetune"]["weight"] == 0.3
    assert (out / "best_model.pt").is_file()
