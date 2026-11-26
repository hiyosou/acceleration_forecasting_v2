"""モデル構造の比較用オプション(line_channels / reference_source / delta_head、2026-11)のテスト。

既定値では従来と同じモデル(重みのキー・形が同じ)で、各オプションは学習・保存・復元・予測まで通る。
"""

import json

import numpy as np
import pytest
import torch

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.cli import main
from acceleration_forecasting_v2.datasets.cycle_features import FEATURE_DIM, add_cycle_features
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from acceleration_forecasting_v2.inference.predict import load_process
from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2


def _batch(n=2, with_cycle=True):
    batch = {"history_values": torch.randn(n, 6), "history_masks": torch.ones(n, 6), "guide_values": torch.randn(n, 3, 12),
             "guide_deltas": torch.randn(n, 3, 12), "guide_similarities": torch.rand(n, 3),
             "guide_mask": torch.ones(n, 3, 12), "retrieval_mask": torch.ones(n, 3)}
    if with_cycle:
        batch["cycle_features"] = torch.randn(n, FEATURE_DIM)
        batch["cycle_lines"] = torch.randn(n, 4, 12)
    return batch


def test_default_model_has_no_new_parameters():
    model = ReferenceModulatedUNetV2(cycle_feature_dim=FEATURE_DIM)
    keys = list(model.state_dict())
    assert not any(k.startswith(("token_encoder", "delta_head")) for k in keys)
    assert model.input.weight.shape == (64, 1, 3)
    assert hasattr(model, "reference_encoder")
    assert 4_900_000 < sum(p.numel() for p in model.parameters()) < 5_200_000  # 約504万(従来)


@pytest.mark.parametrize("kwargs", [
    {"line_channels": 4}, {"reference_source": "condition_tokens"}, {"reference_source": "none"},
    {"base_channels": 32, "reference_source": "none"}, {"delta_head": True},
])
def test_variants_forward_and_shapes(kwargs):
    torch.manual_seed(0)
    model = ReferenceModulatedUNetV2(cycle_feature_dim=FEATURE_DIM, **kwargs).eval()
    batch = _batch()
    out = model(torch.randn(2, 12), torch.tensor([10, 500]), batch)
    assert out.shape == (2, 12) and torch.isfinite(out).all()
    if kwargs.get("delta_head"):
        assert model.predict_delta(batch).shape == (2,)
    else:
        with pytest.raises(RuntimeError):
            model.predict_delta(batch)


def test_condition_tokens_and_none_ignore_the_guides_and_lines_need_cycle_lines():
    torch.manual_seed(0)
    noisy, steps = torch.randn(2, 12), torch.tensor([10, 500])
    for source in ("condition_tokens", "none"):
        model = ReferenceModulatedUNetV2(cycle_feature_dim=FEATURE_DIM, reference_source=source).eval()
        batch = _batch()
        other = {**batch, "guide_values": batch["guide_values"] + 5.0, "guide_deltas": batch["guide_deltas"] - 3.0}
        torch.testing.assert_close(model(noisy, steps, batch), model(noisy, steps, other))
    lines = ReferenceModulatedUNetV2(cycle_feature_dim=FEATURE_DIM, line_channels=4).eval()
    with pytest.raises(KeyError):
        lines(noisy, steps, {k: v for k, v in _batch().items() if k != "cycle_lines"})


def test_condition_tokens_depends_on_history_and_d4():
    torch.manual_seed(0)
    model = ReferenceModulatedUNetV2(cycle_feature_dim=FEATURE_DIM, reference_source="condition_tokens").eval()
    noisy, steps, batch = torch.randn(2, 12), torch.tensor([10, 500]), _batch()
    base = model(noisy, steps, batch)
    changed = {**batch, "cycle_features": batch["cycle_features"] + 1.0}
    assert (model(noisy, steps, changed) - base).abs().max() > 1e-6


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("model_structure")
    artifacts, dataset, featured = root / "artifacts", root / "dataset", root / "featured"
    build_artifacts(artifacts, n_datasets=12, months=22, seed=0)
    assert main(["prepare", "--artifact-dir", str(artifacts), "--output-dir", str(dataset), "--device", "cpu",
                 "--inference-max-per-segment", "10"]) == 0
    add_cycle_features(dataset, featured, artifacts / "trend_catalog.csv")
    return {"root": root, "featured": featured}


def test_dataset_items_have_lines_and_delta_target(prepared):
    dataset = ForecastDatasetV2(prepared["featured"] / "model_train", prepared["featured"])
    item = dataset[0]
    assert item["cycle_lines"].shape == (4, 12) and torch.isfinite(item["cycle_lines"]).all()
    # 水準が同じで傾きが0なら直線は一定、傾きがあれば月とともに動く
    raw = np.array(dataset.cycle_features[0], dtype=float)[:8].reshape(4, 2)
    expected = np.clip(raw[:, :1] + raw[:, 1:] * np.arange(1, 13)[None], 0.1, 6.0)
    np.testing.assert_allclose(dataset.target_norm.denormalize(item["cycle_lines"].numpy()), expected, rtol=1e-4, atol=1e-4)
    # Δ = 最後の有効3か月の平均 − 起点の値 を目標の標準偏差で割った値
    targets, masks = np.asarray(dataset.targets[0], float), np.asarray(dataset.target_masks[0]) > 0
    history, hmask = np.asarray(dataset.arrays["history_values"][0], float), np.asarray(dataset.arrays["history_masks"][0]) > 0
    last = np.flatnonzero(masks & np.isfinite(targets))[-3:]
    current = history[np.flatnonzero(hmask & np.isfinite(history))[-1]]
    assert item["delta_mask"].item() == 1.0
    assert item["delta_target"].item() == pytest.approx((targets[last].mean() - current) / dataset.target_norm.std, rel=1e-5)
    no_target = ForecastDatasetV2(prepared["featured"] / "inference", prepared["featured"], include_targets=False)[0]
    assert "delta_target" not in no_target


@pytest.mark.parametrize("name,flags,expected", [
    ("lines", ["--line-channels", "4"], {"line_channels": 4}),
    ("tokens", ["--reference-source", "condition_tokens"], {"reference_source": "condition_tokens"}),
    ("none32", ["--reference-source", "none", "--base-channels", "32"], {"reference_source": "none", "base_channels": 32}),
    ("aux", ["--delta-aux-weight", "0.5"], {"delta_aux_weight": 0.5}),
])
def test_train_save_load_predict_each_option(prepared, capsys, name, flags, expected):
    run = prepared["root"] / name
    assert main(["train", "--dataset-dir", str(prepared["featured"]), "--output-dir", str(run / "model"), "--epochs", "1",
                 "--batch-size", "32", "--device", "cpu", "--no-progress", "--cycle-features", *flags]) == 0
    config = json.loads(capsys.readouterr().out)["model_config"]
    for key, value in expected.items():
        assert config[key] == value
    process, loaded = load_process(run / "model" / "best_model.pt", torch.device("cpu"))
    assert process.model.line_channels == expected.get("line_channels", 0)
    assert main(["predict", "--dataset-dir", str(prepared["featured"]), "--checkpoint", str(run / "model" / "best_model.pt"),
                 "--output-dir", str(run / "pred"), "--num-samples", "3", "--sampling-steps", "3", "--device", "cpu"]) == 0
    capsys.readouterr()
    assert np.isfinite(np.load(run / "pred" / "samples.npz")["samples"]).all()


def test_default_training_config_has_no_new_keys_and_bad_combinations_fail(prepared, capsys):
    run = prepared["root"] / "plain"
    assert main(["train", "--dataset-dir", str(prepared["featured"]), "--output-dir", str(run), "--epochs", "1",
                 "--batch-size", "32", "--device", "cpu", "--no-progress", "--cycle-features"]) == 0
    config = json.loads(capsys.readouterr().out)["model_config"]
    assert not {"line_channels", "reference_source", "base_channels", "delta_aux_weight"} & set(config)
    with pytest.raises(ValueError):
        main(["train", "--dataset-dir", str(prepared["featured"]), "--output-dir", str(prepared["root"] / "bad1"),
              "--epochs", "1", "--device", "cpu", "--no-progress", "--line-channels", "4"])  # --cycle-features なし
    with pytest.raises(ValueError):
        main(["train", "--dataset-dir", str(prepared["featured"]), "--output-dir", str(prepared["root"] / "bad2"),
              "--epochs", "1", "--device", "cpu", "--no-progress", "--cycle-features", "--delta-aux-weight", "-1"])
