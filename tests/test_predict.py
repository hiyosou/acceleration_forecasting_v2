"""inference.predict の単体テスト(SPEC.md 実装順序案ステップ11)。"""

import numpy as np
import pandas as pd
import pytest

from test_training import _raw_split, _train_kwargs

from acceleration_forecasting_v2.datasets.build import write_dataset
from acceleration_forecasting_v2.inference.predict import predict
from acceleration_forecasting_v2.training.train import train


N_INFERENCE = 5


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("predict")
    dataset_dir = root / "dataset"
    write_dataset({
        "model_train": _raw_split(32, seed=1),
        "model_validation": _raw_split(16, seed=2),
        "inference": _raw_split(N_INFERENCE, seed=3),
    }, dataset_dir)
    train(dataset_dir, root / "run", **_train_kwargs(epochs=2))
    return dataset_dir, root / "run" / "best_model.pt"


def _predict(trained, output_dir, **overrides):
    dataset_dir, checkpoint = trained
    kwargs = dict(device="cpu", num_samples=6, sampling_steps=4, batch_size=2, seed=0)
    kwargs.update(overrides)
    return predict(dataset_dir, checkpoint, output_dir, **kwargs)


def test_predictions_have_one_row_per_record_and_month(trained, tmp_path):
    summary = _predict(trained, tmp_path / "out")
    frame = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert summary["record_count"] == N_INFERENCE
    assert len(frame) == N_INFERENCE * 12
    assert sorted(frame["month_index"].unique()) == list(range(1, 13))
    assert {"trend_id", "prediction_median", "prediction_p10", "prediction_p90", "prediction_std"} <= set(frame.columns)


def test_quantiles_are_ordered_and_within_physical_bounds(trained, tmp_path):
    _predict(trained, tmp_path / "out")
    frame = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert (frame["prediction_p10"] <= frame["prediction_median"] + 1e-9).all()
    assert (frame["prediction_median"] <= frame["prediction_p90"] + 1e-9).all()
    for column in ("prediction_p10", "prediction_median", "prediction_p90"):
        assert frame[column].between(0.1, 6.0).all()


def test_custom_bounds_clip_all_samples(trained, tmp_path):
    _predict(trained, tmp_path / "out", bounds=(1.0, 1.5))
    samples = np.load(tmp_path / "out" / "samples.npz")["samples"]
    assert samples.min() >= 1.0 - 1e-6
    assert samples.max() <= 1.5 + 1e-6


def test_samples_file_has_requested_number_of_samples(trained, tmp_path):
    _predict(trained, tmp_path / "out", num_samples=7)
    data = np.load(tmp_path / "out" / "samples.npz")
    assert data["samples"].shape == (N_INFERENCE, 7, 12)
    assert len(data["trend_ids"]) == N_INFERENCE


def test_same_seed_is_deterministic_and_different_seed_differs(trained, tmp_path):
    _predict(trained, tmp_path / "a", seed=1)
    _predict(trained, tmp_path / "b", seed=1)
    _predict(trained, tmp_path / "c", seed=2)
    a = np.load(tmp_path / "a" / "samples.npz")["samples"]
    b = np.load(tmp_path / "b" / "samples.npz")["samples"]
    c = np.load(tmp_path / "c" / "samples.npz")["samples"]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_prediction_does_not_read_target_files(trained, tmp_path):
    # 正解ファイルが存在しなくても予測できる(=未来の実測値が予測に使われない)。
    dataset_dir, checkpoint = trained
    copy_dir = tmp_path / "dataset_copy"
    import shutil
    shutil.copytree(dataset_dir, copy_dir)
    (copy_dir / "inference" / "target_values.npy").unlink()
    (copy_dir / "inference" / "target_masks.npy").unlink()
    summary = predict(copy_dir, checkpoint, tmp_path / "out", device="cpu", num_samples=3, sampling_steps=3, seed=0)
    assert summary["record_count"] == N_INFERENCE


def test_non_ema_weights_can_be_used(trained, tmp_path):
    summary = _predict(trained, tmp_path / "out", use_ema=False)
    assert summary["use_ema"] is False


def test_metadata_length_mismatch_is_rejected(trained, tmp_path):
    dataset_dir, checkpoint = trained
    import shutil
    copy_dir = tmp_path / "dataset_copy"
    shutil.copytree(dataset_dir, copy_dir)
    metadata_path = copy_dir / "inference" / "metadata.csv"
    pd.read_csv(metadata_path, encoding="utf-8-sig").iloc[:-1].to_csv(metadata_path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="レコード数"):
        predict(copy_dir, checkpoint, tmp_path / "out", device="cpu", num_samples=2, sampling_steps=2)


def test_run_summary_records_configuration(trained, tmp_path):
    summary = _predict(trained, tmp_path / "out", sampling_steps=3, eta=0.5)
    assert summary["sampling_steps"] == 3
    assert summary["eta"] == 0.5
    assert summary["prediction_type"] == "epsilon"
    assert (tmp_path / "out" / "prediction_run.json").is_file()
