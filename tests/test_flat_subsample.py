"""横ばいの件を減らしたデータセット(datasets.flat_subsample)と、副の検証・予測の再利用のテスト(合成データ、CPU)。"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from test_training import _raw_split, _train_kwargs

from acceleration_forecasting_v2.cli import main
from acceleration_forecasting_v2.datasets.build import RawSplit, write_dataset
from acceleration_forecasting_v2.datasets.flat_subsample import (
    VALIDATION_FLATCUT_SPLIT, categorize, change_amounts, flat_keep_mask, subsample_flat,
)
from acceleration_forecasting_v2.inference.predict import predict, reuse_prediction
from acceleration_forecasting_v2.training.train import train

# 変化量の区分(下降・横ばい・小さく上昇・大きく上昇)を行ごとに回す
DELTAS = (-0.5, 0.0, 0.3, 0.8)


def _split(n_segments, per_segment, *, seed):
    """行iの変化量は DELTAS[i % 4]。奇数行は最後の4か月を無効にし、その月には極端な値を入れる(使われないことの確認用)。"""
    n = n_segments * per_segment
    raw = _raw_split(n, seed=seed)
    arrays = {key: value.copy() for key, value in raw.arrays.items()}
    current = np.full(n, 1.0, np.float32)
    target = np.zeros((n, 12), np.float32)
    masks = np.ones((n, 12), np.float32)
    for i in range(n):
        target[i, :] = 1.0 + DELTAS[i % 4]
        if i % 2 == 1:
            masks[i, 8:] = 0.0
            target[i, 8:] = 99.0
    arrays["target_values"], arrays["target_masks"] = target, masks
    arrays["target_synthetic_masks"] = (masks * (np.arange(12) >= 6)).astype(np.float32)
    arrays["segment_weights"] = np.repeat(1.0 / per_segment * np.arange(1, n_segments + 1), per_segment).astype(np.float32)
    metadata = pd.DataFrame({"trend_id": [f"t{seed}_{i}" for i in range(n)],
                             "dataset_id": [f"s{i // per_segment}" for i in range(n)], "current_acc_z_max": current})
    return RawSplit(arrays=arrays, metadata=metadata)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("flat")
    dataset = root / "dataset"
    write_dataset({"model_train": _split(3, 40, seed=1), "model_validation": _split(2, 20, seed=2),
                   "inference": _split(2, 8, seed=3)}, dataset)
    return dataset


def _tree_bytes(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


def test_change_amount_uses_the_last_three_valid_months_including_synthetic(source):
    delta = change_amounts(source / "model_train")
    np.testing.assert_allclose(delta, [DELTAS[i % 4] for i in range(len(delta))], atol=1e-6)
    assert categorize(delta).tolist()[:4] == [0, 1, 2, 3]
    assert categorize([np.nan, 0.19, 0.2, 0.5, -0.2]).tolist() == [-1, 1, 2, 3, 0]


def test_keep_mask_keeps_all_non_flat_and_the_quota_of_flat_per_segment(source):
    metadata = pd.read_csv(source / "model_train" / "metadata.csv", encoding="utf-8-sig")
    flat = categorize(change_amounts(source / "model_train")) == 1
    for ratio, quota in ((1.0, 10), (0.5, 5), (0.25, 3), (0.1, 1), (0.0, 0)):
        keep = flat_keep_mask(metadata["dataset_id"], flat, ratio, seed=42)
        assert keep[~flat].all()
        per_segment = pd.Series(keep & flat).groupby(metadata["dataset_id"].to_numpy()).sum()
        assert per_segment.tolist() == [quota] * 3
    with pytest.raises(ValueError):
        flat_keep_mask(metadata["dataset_id"], flat, 1.5)


def test_kept_rows_are_nested_and_deterministic(source):
    metadata = pd.read_csv(source / "model_train" / "metadata.csv", encoding="utf-8-sig")
    flat = categorize(change_amounts(source / "model_train")) == 1
    masks = {r: flat_keep_mask(metadata["dataset_id"], flat, r, seed=42) for r in (0.1, 0.2, 0.5)}
    assert (masks[0.1] <= masks[0.2]).all() and (masks[0.2] <= masks[0.5]).all()
    assert (masks[0.5] == flat_keep_mask(metadata["dataset_id"], flat, 0.5, seed=42)).all()
    assert (masks[0.5] != flat_keep_mask(metadata["dataset_id"], flat, 0.5, seed=7)).any()


def test_subsample_rows_stay_aligned_and_weights_and_normalization_are_untouched(source, tmp_path):
    before = _tree_bytes(source)
    output = tmp_path / "flat"
    summary = subsample_flat(source, output, 0.5)
    assert _tree_bytes(source) == before  # 元のデータセットは変更されない
    assert not (output / "inference").exists()
    for name in ("condition_normalization.json", "target_normalization.json"):
        assert (output / name).read_bytes() == (source / name).read_bytes()
    train_src, train_out = source / "model_train", output / "model_train"
    metadata_src = pd.read_csv(train_src / "metadata.csv", encoding="utf-8-sig")
    metadata_out = pd.read_csv(train_out / "metadata.csv", encoding="utf-8-sig")
    rows = metadata_src.reset_index().merge(metadata_out[["trend_id"]], on="trend_id")["index"].to_numpy()
    assert len(rows) == len(metadata_out) and (np.diff(rows) > 0).all()  # 元の順序のまま
    for path in train_src.glob("*.npy"):
        np.testing.assert_array_equal(np.load(train_out / path.name), np.load(path)[rows])  # 重みも元の値のまま
    # 検証a(元のまま)と検証b(削る)
    assert (output / "model_validation" / "metadata.csv").read_bytes() == (source / "model_validation" / "metadata.csv").read_bytes()
    assert len(pd.read_csv(output / VALIDATION_FLATCUT_SPLIT / "metadata.csv")) == 2 * (15 + 3)
    assert summary["splits"]["model_train"]["records_after"] == 3 * (30 + 5)
    assert summary["splits"]["model_train"]["categories_after"]["横ばい"] == 3 * 5
    assert json.loads((output / "flat_subsample_summary.json").read_text(encoding="utf-8")) == summary
    with pytest.raises(FileExistsError):
        subsample_flat(source, output, 0.5)


def test_subsample_flat_cli(source, tmp_path, capsys):
    assert main(["subsample-flat", "--source-dir", str(source), "--output-dir", str(tmp_path / "o"), "--keep-ratio", "0.2"]) == 0
    assert json.loads(capsys.readouterr().out)["keep_ratio"] == 0.2


# --- 副の検証 ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small(tmp_path_factory):
    root = tmp_path_factory.mktemp("secondary")
    dataset = root / "dataset"
    write_dataset({"model_train": _raw_split(32, seed=1), "model_validation": _raw_split(16, seed=2),
                   VALIDATION_FLATCUT_SPLIT: _raw_split(8, seed=4), "inference": _raw_split(5, seed=3)}, dataset)
    return root, dataset


def test_secondary_validation_writes_outputs_and_keeps_the_primary_model_identical(small):
    root, dataset = small
    train(dataset, root / "plain", **_train_kwargs(epochs=3))
    result = train(dataset, root / "both", **_train_kwargs(epochs=3), secondary_validation_split=VALIDATION_FLATCUT_SPLIT)
    history = pd.read_csv(root / "both" / "training_history.csv")
    assert {"validation_loss_secondary", "best_validation_loss_secondary"} <= set(history.columns)
    assert (root / "both" / "best_model_secondary.pt").is_file() and not (root / "plain" / "best_model_secondary.pt").exists()
    assert result["secondary_validation"]["split"] == VALIDATION_FLATCUT_SPLIT and result["best_epoch"] >= 1
    plain = torch.load(root / "plain" / "best_model.pt", weights_only=False)
    both = torch.load(root / "both" / "best_model.pt", weights_only=False)
    assert all(torch.equal(plain["ema_state_dict"][k], both["ema_state_dict"][k]) for k in plain["ema_state_dict"])
    assert plain["model_config"] == both["model_config"]
    plain_history = pd.read_csv(root / "plain" / "training_history.csv")
    np.testing.assert_array_equal(plain_history["validation_loss"], history["validation_loss"])


def test_secondary_validation_keeps_training_until_both_tracks_stop(small, monkeypatch):
    root, dataset = small
    scripted = {"primary": [1.0, 1.1, 1.2, 1.3, 0.5, 0.4], "secondary": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]}
    calls = {"n": 0}

    def fake_validation_loss(process, loader, device, seed=42, prepare=None, compute_real_loss=True):
        index, track = divmod(calls["n"], 2)
        calls["n"] += 1
        return {"loss": scripted["secondary" if track else "primary"][index], "record_loss": 0.0}

    monkeypatch.setattr("acceleration_forecasting_v2.training.train.validation_loss", fake_validation_loss)
    result = train(dataset, root / "scripted", **_train_kwargs(epochs=6, patience=2),
                   secondary_validation_split=VALIDATION_FLATCUT_SPLIT)
    assert result["epochs_completed"] == 6                      # 主はepoch3で止まるが、副が改善を続けるので続く
    assert result["best_epoch"] == 1 and result["best_validation_loss"] == 1.0   # 主はpatienceに達した後は更新しない
    assert result["secondary_validation"]["best_epoch"] == 6


def test_secondary_validation_resumes_and_rejects_a_run_without_it(small):
    root, dataset = small
    train(dataset, root / "resume", **_train_kwargs(epochs=2), secondary_validation_split=VALIDATION_FLATCUT_SPLIT)
    result = train(dataset, root / "resume", **_train_kwargs(epochs=4), secondary_validation_split=VALIDATION_FLATCUT_SPLIT)
    assert result["epochs_completed"] == 4
    train(dataset, root / "plain_first", **_train_kwargs(epochs=1))
    with pytest.raises(ValueError):
        train(dataset, root / "plain_first", **_train_kwargs(epochs=3), secondary_validation_split=VALIDATION_FLATCUT_SPLIT)


def test_train_cli_accepts_secondary_validation_split(small, capsys):
    root, dataset = small
    assert main(["train", "--dataset-dir", str(dataset), "--output-dir", str(root / "cli"), "--epochs", "1",
                 "--batch-size", "16", "--device", "cpu", "--no-progress",
                 "--secondary-validation-split", VALIDATION_FLATCUT_SPLIT]) == 0
    assert json.loads(capsys.readouterr().out)["secondary_validation"]["split"] == VALIDATION_FLATCUT_SPLIT


# --- 予測の再利用 ------------------------------------------------------------------------------------
PREDICT = dict(device="cpu", num_samples=6, sampling_steps=4, batch_size=2, seed=0, start_timestep=958)


def _plain_checkpoint(small):
    root, dataset = small
    checkpoint = root / "plain" / "best_model.pt"
    if not checkpoint.exists():
        train(dataset, root / "plain", **_train_kwargs(epochs=3))
    return checkpoint


def _same_checkpoint_copy(checkpoint, path):
    torch.save(torch.load(checkpoint, weights_only=False), path)
    return path


def _changed_checkpoint(checkpoint, path):
    changed = torch.load(checkpoint, weights_only=False)
    first = next(iter(changed["ema_state_dict"]))
    changed["ema_state_dict"][first] = changed["ema_state_dict"][first] + 1e-3
    torch.save(changed, path)
    return path


def test_reuse_prediction_is_byte_identical_to_predicting_again(small, tmp_path):
    _, dataset = small
    checkpoint = _plain_checkpoint(small)
    copy = _same_checkpoint_copy(checkpoint, tmp_path / "copy.pt")
    predict(dataset, checkpoint, tmp_path / "a", **PREDICT)
    expected = predict(dataset, copy, tmp_path / "base", **PREDICT)  # 基準:通常の予測
    reused = reuse_prediction(tmp_path / "a", copy, tmp_path / "b", dataset_size=5, **PREDICT)
    assert reused == expected
    assert _tree_bytes(tmp_path / "b") == _tree_bytes(tmp_path / "base")


def test_reuse_prediction_returns_none_when_weights_or_settings_differ(small, tmp_path):
    _, dataset = small
    checkpoint = _plain_checkpoint(small)
    predict(dataset, checkpoint, tmp_path / "a", **PREDICT)
    changed = _changed_checkpoint(checkpoint, tmp_path / "changed.pt")
    assert reuse_prediction(tmp_path / "a", changed, tmp_path / "b", dataset_size=5, **PREDICT) is None
    copy = _same_checkpoint_copy(checkpoint, tmp_path / "copy.pt")
    assert reuse_prediction(tmp_path / "a", copy, tmp_path / "b", dataset_size=5, **{**PREDICT, "seed": 1}) is None
    assert reuse_prediction(tmp_path / "a", copy, tmp_path / "b", dataset_size=6, **PREDICT) is None
    assert reuse_prediction(tmp_path / "a", copy, tmp_path / "b", dataset_size=5, **{**PREDICT, "start_timestep": None}) is None
    assert reuse_prediction(tmp_path / "missing", copy, tmp_path / "b", dataset_size=5, **PREDICT) is None
    assert not (tmp_path / "b").exists()


def test_predict_cli_reuse_from_matches_a_plain_cli_run(small, tmp_path, capsys):
    _, dataset = small
    checkpoint = _plain_checkpoint(small)
    copy = _same_checkpoint_copy(checkpoint, tmp_path / "copy.pt")

    def run(output, checkpoint_path, *extra):
        argv = ["predict", "--dataset-dir", str(dataset), "--output-dir", str(output), "--num-samples", "6",
                "--sampling-steps", "4", "--batch-size", "2", "--seed", "0", "--device", "cpu",
                "--start-timestep", "958", "--checkpoint", str(checkpoint_path), *extra]
        assert main(argv) == 0
        return capsys.readouterr().out

    run(tmp_path / "a", checkpoint)
    plain = run(tmp_path / "base", copy)
    reused = run(tmp_path / "b", copy, "--reuse-from", str(tmp_path / "a"))
    assert reused == plain
    assert _tree_bytes(tmp_path / "b") == _tree_bytes(tmp_path / "base")
    # 重みが違えば通常の予測に切り替わる
    changed = _changed_checkpoint(checkpoint, tmp_path / "changed.pt")
    fallback = run(tmp_path / "c", changed, "--reuse-from", str(tmp_path / "a"))
    assert json.loads(fallback)["checkpoint"].endswith("changed.pt")
    assert not np.array_equal(np.load(tmp_path / "c" / "samples.npz")["samples"],
                              np.load(tmp_path / "a" / "samples.npz")["samples"])


def test_validation_loss_without_real_loss_keeps_the_other_values_identical(source, tmp_path):
    from torch.utils.data import DataLoader

    from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
    from acceleration_forecasting_v2.diffusion.process import DiffusionProcess
    from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2
    from acceleration_forecasting_v2.training.train import validation_loss

    torch.manual_seed(0)
    process = DiffusionProcess(ReferenceModulatedUNetV2(), 1000, prediction_type="epsilon")
    loader = DataLoader(ForecastDatasetV2(source / "model_validation", source), batch_size=16, shuffle=False)
    with_real = validation_loss(process, loader, torch.device("cpu"), 42)
    without_real = validation_loss(process, loader, torch.device("cpu"), 42, compute_real_loss=False)
    assert "real_loss" in with_real and "real_loss" not in without_real
    assert with_real["loss"] == without_real["loss"] and with_real["record_loss"] == without_real["record_loss"]
