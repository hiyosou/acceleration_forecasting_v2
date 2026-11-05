"""条件入力の履歴を18か月(拡張ウィンドウ)にしたときの、モデル・学習・予測の通しのテスト(合成データ、CPU、小規模)。"""

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


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("history18")
    artifacts, dataset, featured = root / "artifacts", root / "dataset", root / "featured"
    build_artifacts(artifacts, n_datasets=12, months=22, seed=0)
    assert main(["prepare", "--artifact-dir", str(artifacts), "--output-dir", str(dataset), "--device", "cpu",
                 "--inference-max-per-segment", "10", "--history-months", "18"]) == 0
    add_cycle_features(dataset, featured, artifacts / "trend_catalog.csv")
    return {"root": root, "dataset": dataset, "featured": featured}


def test_prepare_cli_writes_18_month_history(prepared):
    for split in ("model_train", "model_validation", "inference"):
        values = np.load(prepared["dataset"] / split / "history_values.npy")
        masks = np.load(prepared["dataset"] / split / "history_masks.npy")
        assert values.shape[1] == 18 and masks.shape == values.shape
    item = ForecastDatasetV2(prepared["dataset"] / "inference", prepared["dataset"], include_targets=False)[0]
    assert item["history_values"].shape == (18,) and item["history_masks"].shape == (18,)
    summary = json.loads((prepared["dataset"] / "prepare_summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["history_months"] == 18


def test_model_condition_encoder_width_follows_history_months():
    assert ReferenceModulatedUNetV2(history_months=18).condition_encoder[0].in_features == 36
    assert ReferenceModulatedUNetV2(history_months=18, cycle_feature_dim=FEATURE_DIM).condition_encoder[0].in_features \
        == 36 + FEATURE_DIM
    assert ReferenceModulatedUNetV2().condition_encoder[0].in_features == 12  # 既定は従来どおり


@pytest.mark.parametrize("name,with_d4", [("plain", False), ("d4", True)])
def test_train_and_predict_with_18_month_history(prepared, capsys, name, with_d4):
    dataset = prepared["featured"] if with_d4 else prepared["dataset"]
    run = prepared["root"] / f"run_{name}"
    command = ["train", "--dataset-dir", str(dataset), "--output-dir", str(run / "model"), "--epochs", "1",
               "--batch-size", "32", "--device", "cpu", "--no-progress"] + (["--cycle-features"] if with_d4 else [])
    assert main(command) == 0
    config = json.loads(capsys.readouterr().out)["model_config"]
    assert config["history_months"] == 18
    assert ("cycle_feature_dim" in config) == with_d4
    process, loaded = load_process(run / "model" / "best_model.pt", torch.device("cpu"))
    assert process.model.history_months == 18 and loaded["history_months"] == 18
    assert main(["predict", "--dataset-dir", str(dataset), "--checkpoint", str(run / "model" / "best_model.pt"),
                 "--output-dir", str(run / "pred"), "--num-samples", "4", "--sampling-steps", "3",
                 "--device", "cpu"]) == 0
    capsys.readouterr()
    assert np.load(run / "pred" / "samples.npz")["samples"].shape[1] == 4


def test_six_month_model_config_has_no_history_key_and_loads_as_six(tmp_path, capsys):
    artifacts, dataset = tmp_path / "artifacts", tmp_path / "dataset"
    build_artifacts(artifacts, n_datasets=12, months=22, seed=0)
    assert main(["prepare", "--artifact-dir", str(artifacts), "--output-dir", str(dataset), "--device", "cpu",
                 "--inference-max-per-segment", "10"]) == 0
    capsys.readouterr()
    run = tmp_path / "run"
    assert main(["train", "--dataset-dir", str(dataset), "--output-dir", str(run), "--epochs", "1",
                 "--batch-size", "32", "--device", "cpu", "--no-progress"]) == 0
    assert "history_months" not in json.loads(capsys.readouterr().out)["model_config"]  # 旧checkpointと同じ構成
    process, _ = load_process(run / "best_model.pt", torch.device("cpu"))
    assert process.model.history_months == 6
