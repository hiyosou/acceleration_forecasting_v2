"""datasets.build.write_dataset と training.train の単体テスト(SPEC.md 1.1・1.5・実装順序案ステップ10)。"""

import numpy as np
import pandas as pd
import pytest
import torch

from acceleration_forecasting_v2.datasets.build import RawSplit, write_dataset
from acceleration_forecasting_v2.training.train import train, validation_loss


def _raw_split(n, *, seed=0, target_offset=0.0, weights=None):
    rng = np.random.default_rng(seed)
    history = rng.normal(loc=1.0, scale=0.3, size=(n, 6)).astype(np.float32)
    # 学習可能な規則性: 12か月分の正解 = 履歴平均 + 月ごとの固定の傾き(+offset)。
    slope = np.linspace(0.0, 0.5, 12, dtype=np.float32)
    target = history.mean(axis=1, keepdims=True) + slope[None, :] + target_offset
    arrays = {
        "history_values": history,
        "history_masks": np.ones((n, 6), np.float32),
        "guide_values": (target[:, None, :] + rng.normal(scale=0.05, size=(n, 3, 12))).astype(np.float32),
        "guide_masks": np.ones((n, 3, 12), np.float32),
        "guide_deltas": rng.normal(scale=0.05, size=(n, 3, 12)).astype(np.float32),
        "guide_similarities": rng.uniform(size=(n, 3)).astype(np.float32),
        "retrieval_masks": np.ones((n, 3), np.float32),
        "guide_baselines": target.astype(np.float32),
        "guide_softmax_weights": np.full((n, 3, 12), 1 / 3, np.float32),
        "segment_weights": np.ones(n, np.float32) if weights is None else np.asarray(weights, np.float32),
        "target_values": target.astype(np.float32),
        "target_masks": np.ones((n, 12), np.float32),
    }
    return RawSplit(arrays=arrays, metadata=pd.DataFrame({"trend_id": [f"t{i}" for i in range(n)]}))


def _write_small_dataset(tmp_path, *, n_train=32, n_valid=16):
    dataset_dir = tmp_path / "dataset"
    write_dataset({
        "model_train": _raw_split(n_train, seed=1),
        "model_validation": _raw_split(n_valid, seed=2),
    }, dataset_dir)
    return dataset_dir


def _train_kwargs(**overrides):
    kwargs = dict(device="cpu", epochs=6, batch_size=16, accumulation_steps=1, learning_rate=1e-3,
                  patience=100, progress=False, seed=0)
    kwargs.update(overrides)
    return kwargs


# --- write_dataset -----------------------------------------------------------------

def test_write_dataset_writes_all_arrays_and_normalizations(tmp_path):
    dataset_dir = _write_small_dataset(tmp_path)
    for split in ("model_train", "model_validation"):
        assert (dataset_dir / split / "target_values.npy").is_file()
        assert (dataset_dir / split / "segment_weights.npy").is_file()
        assert (dataset_dir / split / "metadata.csv").is_file()
    assert (dataset_dir / "condition_normalization.json").is_file()
    assert (dataset_dir / "target_normalization.json").is_file()


def test_write_dataset_fits_normalization_only_from_model_train(tmp_path):
    # validationの正解値を極端に大きくずらしても、正規化統計(平均)は変わらない
    # (SPEC.md 2.3 チェック項目5: model_trainのみでfit)。
    train_raw = _raw_split(32, seed=1)
    stats = write_dataset({
        "model_train": train_raw,
        "model_validation": _raw_split(16, seed=2, target_offset=1000.0),
    }, tmp_path / "dataset")
    assert stats["target_norm"].mean == pytest.approx(float(train_raw.arrays["target_values"].mean()), rel=1e-5)
    assert stats["target_norm"].mean < 10.0


def test_write_dataset_requires_train_and_validation(tmp_path):
    with pytest.raises(ValueError, match="model_validation"):
        write_dataset({"model_train": _raw_split(8)}, tmp_path / "dataset")


# --- validation_loss ---------------------------------------------------------------

class _FakeProcess:
    steps = 10

    def __init__(self):
        self.model = torch.nn.Identity()

    def per_record_loss(self, batch, noise=None, timesteps=None):
        return batch["target"][:, 0]


def test_validation_loss_is_segment_weighted_over_whole_set_not_per_batch():
    # SPEC 1.1の例: セグメントA 10件(loss 0.2, 重み0.1)、セグメントB 1件(loss 0.5, 重み1.0)。
    # バッチサイズ3で分割しても、検証セット全体で (0.2+0.5)/2 = 0.35 になる。
    losses = [0.2] * 10 + [0.5]
    weights = [0.1] * 10 + [1.0]
    items = [{"target": torch.tensor([loss, 0.0]), "segment_weight": torch.tensor(weight)}
             for loss, weight in zip(losses, weights)]
    loader = torch.utils.data.DataLoader(items, batch_size=3, shuffle=False)
    result = validation_loss(_FakeProcess(), loader, torch.device("cpu"))
    assert result["loss"] == pytest.approx(0.35, abs=1e-6)
    # 参考値のレコード単位単純平均は(10*0.2+0.5)/11 ≈ 0.227 と異なる。
    assert result["record_loss"] == pytest.approx(2.5 / 11, abs=1e-6)
    assert result["loss"] != pytest.approx(result["record_loss"], abs=1e-3)


# --- train -------------------------------------------------------------------------

def test_train_writes_outputs_and_history_has_one_row_per_epoch(tmp_path):
    dataset_dir = _write_small_dataset(tmp_path)
    output_dir = tmp_path / "run"
    result = train(dataset_dir, output_dir, **_train_kwargs())
    assert result["epochs_completed"] == 6
    for name in ("best_model.pt", "last_model.pt", "training_history.csv", "resolved_config.json"):
        assert (output_dir / name).is_file()
    history = pd.read_csv(output_dir / "training_history.csv")
    assert len(history) == 6
    assert {"train_loss", "validation_loss", "validation_record_loss"} <= set(history.columns)
    assert np.isfinite(history[["train_loss", "validation_loss"]].to_numpy()).all()


def test_train_loss_decreases_on_learnable_synthetic_data(tmp_path):
    dataset_dir = _write_small_dataset(tmp_path, n_train=64)
    history_path = tmp_path / "run" / "training_history.csv"
    train(dataset_dir, tmp_path / "run", **_train_kwargs(epochs=15, batch_size=16))
    history = pd.read_csv(history_path)
    early = history["train_loss"].iloc[:3].mean()
    late = history["train_loss"].iloc[-3:].mean()
    assert late < early


def test_train_supports_v_prediction(tmp_path):
    dataset_dir = _write_small_dataset(tmp_path)
    result = train(dataset_dir, tmp_path / "run", **_train_kwargs(epochs=2, prediction_type="v_prediction"))
    assert result["model_config"]["prediction_type"] == "v_prediction"
    assert np.isfinite(result["best_validation_loss"])


def test_train_rejects_unsupported_prediction_type(tmp_path):
    dataset_dir = _write_small_dataset(tmp_path)
    with pytest.raises(ValueError, match="prediction_type"):
        train(dataset_dir, tmp_path / "run", **_train_kwargs(prediction_type="x0_prediction"))


def test_train_early_stopping_stops_after_patience(tmp_path):
    dataset_dir = _write_small_dataset(tmp_path)
    # min_deltaを巨大にすると2エポック目以降は「改善なし」となり、patience=1で停止する。
    result = train(dataset_dir, tmp_path / "run", **_train_kwargs(epochs=10, patience=1, min_delta=1e9))
    assert result["epochs_completed"] == 2


def test_train_resume_continues_from_last_epoch(tmp_path):
    dataset_dir = _write_small_dataset(tmp_path)
    output_dir = tmp_path / "run"
    train(dataset_dir, output_dir, **_train_kwargs(epochs=2))
    result = train(dataset_dir, output_dir, **_train_kwargs(epochs=4))
    assert result["epochs_completed"] == 4
    history = pd.read_csv(output_dir / "training_history.csv")
    assert history["epoch"].tolist() == [1, 2, 3, 4]


def test_train_resume_rejects_mismatched_configuration(tmp_path):
    dataset_dir = _write_small_dataset(tmp_path)
    output_dir = tmp_path / "run"
    train(dataset_dir, output_dir, **_train_kwargs(epochs=1))
    with pytest.raises(ValueError, match="does not match"):
        train(dataset_dir, output_dir, **_train_kwargs(epochs=2, prediction_type="v_prediction"))


def test_train_resume_rejects_different_dataset(tmp_path):
    output_dir = tmp_path / "run"
    train(_write_small_dataset(tmp_path), output_dir, **_train_kwargs(epochs=1))
    other = tmp_path / "other"
    write_dataset({"model_train": _raw_split(24, seed=5), "model_validation": _raw_split(8, seed=6)}, other)
    with pytest.raises(ValueError, match="does not match"):
        train(other, output_dir, **_train_kwargs(epochs=2))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_train_runs_on_cuda_with_autocast_and_grad_scaler(tmp_path):
    # CPUテストでは通らないGPU固有の経路(autocast/GradScaler/bf16)を実際に実行する。
    dataset_dir = _write_small_dataset(tmp_path)
    result = train(dataset_dir, tmp_path / "run", **_train_kwargs(device="cuda", epochs=3, accumulation_steps=2))
    assert result["device"].startswith("cuda")
    history = pd.read_csv(tmp_path / "run" / "training_history.csv")
    assert np.isfinite(history[["train_loss", "validation_loss"]].to_numpy()).all()
