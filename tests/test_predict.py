"""inference.predict の単体テスト(SPEC.md 実装順序案ステップ11)。"""

import numpy as np
import pandas as pd
import pytest
import torch

from test_training import _raw_split, _raw_split_with_current_max, _train_kwargs

from acceleration_forecasting_v2.datasets.build import write_dataset
from acceleration_forecasting_v2.inference.predict import load_process, predict
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


@pytest.fixture(scope="module")
def trained_with_current_max(tmp_path_factory):
    # guide_mode="self_target"はcurrent_acc_z_max列(build_self_target_batchが要求)を
    # 必要とするため、別fixtureとして用意する(既定のtrained fixtureには存在しない)。
    root = tmp_path_factory.mktemp("predict_self_target")
    dataset_dir = root / "dataset"
    write_dataset({
        "model_train": _raw_split_with_current_max(32, seed=1),
        "model_validation": _raw_split_with_current_max(16, seed=2),
        "inference": _raw_split_with_current_max(N_INFERENCE, seed=3),
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


def test_mixed_precision_is_off_on_cpu_by_default(trained, tmp_path):
    summary = _predict(trained, tmp_path / "out")
    assert summary["mixed_precision"] is False


def test_guide_mode_rejects_unknown_value(trained, tmp_path):
    dataset_dir, checkpoint = trained
    with pytest.raises(ValueError, match="guide_mode"):
        predict(dataset_dir, checkpoint, tmp_path / "out", device="cpu", num_samples=2, sampling_steps=2,
               guide_mode="bogus")


def test_guide_mode_retrieved_matches_omitting_the_parameter(trained, tmp_path):
    # guide_mode="retrieved"(既定)を明示しても、省略した場合と完全にbit-identical。
    _predict(trained, tmp_path / "a", seed=1)
    _predict(trained, tmp_path / "b", seed=1, guide_mode="retrieved")
    a = np.load(tmp_path / "a" / "samples.npz")["samples"]
    b = np.load(tmp_path / "b" / "samples.npz")["samples"]
    np.testing.assert_array_equal(a, b)


def test_guide_mode_self_target_requires_target_files(trained_with_current_max, tmp_path):
    # guide_mode="retrieved"(既定)は正解ファイルが無くても動く(test_prediction_does_not_
    # read_target_filesの通り)が、guide_mode="self_target"は正解を必要とするため、
    # 正解ファイルが無いと明確なエラーで失敗するべき(サイレントに壊れた結果を返さない)。
    dataset_dir, checkpoint = trained_with_current_max
    import shutil
    copy_dir = tmp_path / "dataset_copy"
    shutil.copytree(dataset_dir, copy_dir)
    (copy_dir / "inference" / "target_values.npy").unlink()
    with pytest.raises(FileNotFoundError):
        predict(copy_dir, checkpoint, tmp_path / "out", device="cpu", num_samples=2, sampling_steps=2,
               guide_mode="self_target")


def test_guide_mode_self_target_records_target_count_and_run_summary(trained_with_current_max, tmp_path):
    dataset_dir, checkpoint = trained_with_current_max
    summary = predict(dataset_dir, checkpoint, tmp_path / "out", device="cpu", num_samples=6, sampling_steps=4,
                      batch_size=2, seed=0, guide_mode="self_target")
    assert summary["guide_mode"] == "self_target"
    assert summary["record_count"] == N_INFERENCE
    frame = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert len(frame) == N_INFERENCE * 12


def test_guide_mode_self_target_differs_from_retrieved(trained_with_current_max, tmp_path):
    # guideの中身を変えているので(検索結果→自分自身の正解)、同一seedでも出力は変わるはず。
    dataset_dir, checkpoint = trained_with_current_max
    kwargs = dict(device="cpu", num_samples=6, sampling_steps=4, batch_size=2, seed=0)
    predict(dataset_dir, checkpoint, tmp_path / "retrieved", guide_mode="retrieved", **kwargs)
    predict(dataset_dir, checkpoint, tmp_path / "self_target", guide_mode="self_target", **kwargs)
    retrieved = pd.read_csv(tmp_path / "retrieved" / "predictions.csv")["prediction_median"].to_numpy()
    self_target = pd.read_csv(tmp_path / "self_target" / "predictions.csv")["prediction_median"].to_numpy()
    assert not np.allclose(retrieved, self_target)


def test_run_summary_records_guide_mode(trained, tmp_path):
    summary = _predict(trained, tmp_path / "out")
    assert summary["guide_mode"] == "retrieved"


def test_load_process_reads_a_checkpoint_saved_without_context_residual(tmp_path):
    # 2026-10-05: context_residual導入(2026-09-30)より前に学習したcheckpointも、
    # 現行コードでそのまま読み込めることの回帰テスト(state_dictのキーから自動判定、
    # models/reference_modulated_unet.pyのcontext_residual_enabledフラグ参照)。
    from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2

    model = ReferenceModulatedUNetV2(context_residual_enabled=False)
    checkpoint_path = tmp_path / "pre_context_residual.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "ema_state_dict": model.state_dict(),
        "model_config": {"dropout": 0.1, "prediction_type": "epsilon", "diffusion_steps": 1000},
    }, checkpoint_path)

    process, config = load_process(checkpoint_path, "cpu")
    assert not any("context_residual" in key for key in process.model.state_dict())
    assert config["prediction_type"] == "epsilon"


@pytest.mark.gpu
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_bf16_sampling_on_cuda_stays_close_to_fp32(trained, tmp_path):
    dataset_dir, checkpoint = trained
    kwargs = dict(device="cuda", num_samples=16, sampling_steps=8, batch_size=5, seed=0)
    fp32 = predict(dataset_dir, checkpoint, tmp_path / "fp32", mixed_precision=False, **kwargs)
    bf16 = predict(dataset_dir, checkpoint, tmp_path / "bf16", mixed_precision=True, **kwargs)
    assert fp32["mixed_precision"] is False and bf16["mixed_precision"] is True
    a = pd.read_csv(tmp_path / "fp32" / "predictions.csv")["prediction_median"].to_numpy()
    b = pd.read_csv(tmp_path / "bf16" / "predictions.csv")["prediction_median"].to_numpy()
    assert np.isfinite(b).all()
    print("mean |median(fp32)-median(bf16)| =", float(np.abs(a - b).mean()))
    assert np.abs(a - b).mean() < 0.15
