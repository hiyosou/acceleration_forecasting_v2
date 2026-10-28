"""忘却因子Dの要約(D4)を拡散モデルの条件入力に足す処理のテスト(合成データ、CPU、小規模)。"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.cli import main
from acceleration_forecasting_v2.datasets.cycle_features import (
    FEATURE_DIM, NORMALIZATION_FILE, add_cycle_features, normalize_cycle_features, split_cycle_features,
)
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from acceleration_forecasting_v2.evaluation.feature_screening import _catalog_groups
from acceleration_forecasting_v2.evaluation.forgetting_factor import D4_LAMBDAS, d4_features
from acceleration_forecasting_v2.inference.predict import load_process
from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("cycle_features")
    artifacts, dataset, featured = root / "artifacts", root / "dataset", root / "featured"
    build_artifacts(artifacts, n_datasets=12, months=22, seed=0)
    assert main(["prepare", "--artifact-dir", str(artifacts), "--output-dir", str(dataset), "--device", "cpu",
                 "--inference-max-per-segment", "10"]) == 0
    summary = add_cycle_features(dataset, featured, artifacts / "trend_catalog.csv")
    return {"root": root, "artifacts": artifacts, "dataset": dataset, "featured": featured, "summary": summary,
            "catalog": artifacts / "trend_catalog.csv"}


def test_features_match_d4_of_measurements_up_to_the_anchor(prepared):
    metadata = pd.read_csv(prepared["featured"] / "inference" / "metadata.csv", encoding="utf-8-sig")
    features = np.load(prepared["featured"] / "inference" / "cycle_features.npy")
    assert features.shape == (len(metadata), FEATURE_DIM) and features.dtype == np.float32
    catalog = pd.read_csv(prepared["catalog"], encoding="utf-8-sig")
    row = metadata.iloc[0]
    rows = catalog.loc[(catalog["dataset_id"] == row["dataset_id"])
                       & (pd.to_datetime(catalog["measurement_date"]) <= pd.Timestamp(row["anchor_date"]))]
    expected = d4_features(pd.to_datetime(rows["measurement_date"]).sort_values(),
                           rows.assign(d=pd.to_datetime(rows["measurement_date"])).sort_values("d")["current_acc_z_max"])
    np.testing.assert_allclose(features[0], expected.reshape(-1), rtol=1e-4, atol=1e-5)


def test_features_ignore_measurements_after_the_anchor(prepared):
    metadata = pd.read_csv(prepared["dataset"] / "inference" / "metadata.csv", encoding="utf-8-sig").head(1)
    row = metadata.iloc[0]
    groups = _catalog_groups(prepared["catalog"])
    before = split_cycle_features(metadata, groups)
    dates, values = groups[str(row["dataset_id"])]
    later = values.copy()
    later[dates > pd.Timestamp(row["anchor_date"])] += 5.0
    assert (dates > pd.Timestamp(row["anchor_date"])).any()
    after = split_cycle_features(metadata, {**groups, str(row["dataset_id"]): (dates, later)})
    np.testing.assert_array_equal(before, after)


def test_source_dataset_is_untouched_and_normalization_matches_the_json(prepared):
    assert not (prepared["dataset"] / "inference" / "cycle_features.npy").exists()
    assert not (prepared["dataset"] / NORMALIZATION_FILE).exists()
    normalization = json.loads((prepared["featured"] / NORMALIZATION_FILE).read_text(encoding="utf-8"))
    assert len(normalization["low"]) == FEATURE_DIM and normalization["lambdas"] == list(D4_LAMBDAS)
    physical = np.load(prepared["featured"] / "model_train" / "cycle_features.npy")
    normalized = normalize_cycle_features(physical, normalization)
    condition = json.loads((prepared["featured"] / "condition_normalization.json").read_text(encoding="utf-8"))
    assert normalization["shift"][0] == pytest.approx(condition["mean"])
    assert normalization["scale"][1] == pytest.approx(condition["std"])
    # 傾きの列は学習データの0.5%〜99.5%点にクリップされる
    clipped = np.clip(physical[:, 1], normalization["low"][1], normalization["high"][1])
    np.testing.assert_allclose(normalized[:, 1], clipped / condition["std"], rtol=1e-5, atol=1e-6)
    assert prepared["summary"]["splits"]["model_train"]["records"] == len(physical)


def test_dataset_item_contains_normalized_cycle_features(prepared):
    dataset = ForecastDatasetV2(prepared["featured"] / "inference", prepared["featured"], include_targets=False)
    item = dataset[0]
    assert item["cycle_features"].shape == (FEATURE_DIM,) and torch.isfinite(item["cycle_features"]).all()
    plain = ForecastDatasetV2(prepared["dataset"] / "inference", prepared["dataset"], include_targets=False)
    assert "cycle_features" not in plain[0]


def test_model_with_dim_zero_is_unchanged_and_with_dim_requires_features():
    torch.manual_seed(0)
    old = ReferenceModulatedUNetV2()
    assert old.condition_encoder[0].in_features == 12
    new = ReferenceModulatedUNetV2(cycle_feature_dim=FEATURE_DIM)
    assert new.condition_encoder[0].in_features == 12 + FEATURE_DIM
    batch = {"history_values": torch.zeros(2, 6), "history_masks": torch.ones(2, 6), "guide_values": torch.zeros(2, 3, 12),
             "guide_deltas": torch.zeros(2, 3, 12), "guide_similarities": torch.zeros(2, 3),
             "guide_mask": torch.ones(2, 3, 12), "retrieval_mask": torch.ones(2, 3)}
    noisy, steps = torch.zeros(2, 12), torch.zeros(2, dtype=torch.long)
    assert old(noisy, steps, batch).shape == (2, 12)
    with pytest.raises(KeyError):
        new(noisy, steps, batch)
    assert new(noisy, steps, {**batch, "cycle_features": torch.zeros(2, FEATURE_DIM)}).shape == (2, 12)


def test_train_with_cycle_features_then_predict(prepared, capsys):
    run = prepared["root"] / "run"
    assert main(["train", "--dataset-dir", str(prepared["featured"]), "--output-dir", str(run / "model"), "--epochs", "1",
                 "--batch-size", "32", "--device", "cpu", "--no-progress", "--cycle-features"]) == 0
    assert json.loads(capsys.readouterr().out)["model_config"]["cycle_feature_dim"] == FEATURE_DIM
    process, config = load_process(run / "model" / "best_model.pt", torch.device("cpu"))
    assert config["cycle_feature_dim"] == FEATURE_DIM
    assert main(["predict", "--dataset-dir", str(prepared["featured"]), "--checkpoint", str(run / "model" / "best_model.pt"),
                 "--output-dir", str(run / "pred"), "--num-samples", "4", "--sampling-steps", "3", "--device", "cpu"]) == 0
    capsys.readouterr()
    assert np.load(run / "pred" / "samples.npz")["samples"].shape[1] == 4


def test_train_without_features_keeps_the_old_model_config_and_flag_requires_the_file(prepared, capsys):
    run = prepared["root"] / "run_plain"
    assert main(["train", "--dataset-dir", str(prepared["dataset"]), "--output-dir", str(run), "--epochs", "1",
                 "--batch-size", "32", "--device", "cpu", "--no-progress"]) == 0
    assert "cycle_feature_dim" not in json.loads(capsys.readouterr().out)["model_config"]
    process, config = load_process(run / "best_model.pt", torch.device("cpu"))
    assert process.model.cycle_feature_dim == 0
    with pytest.raises(ValueError):
        main(["train", "--dataset-dir", str(prepared["dataset"]), "--output-dir", str(prepared["root"] / "run_bad"),
              "--epochs", "1", "--device", "cpu", "--no-progress", "--cycle-features"])


def test_add_cycle_features_cli_and_reused_normalization(prepared, tmp_path, capsys):
    second = tmp_path / "second"
    assert main(["add-cycle-features", "--source-dir", str(prepared["dataset"]), "--output-dir", str(second),
                 "--trend-catalog-path", str(prepared["catalog"]), "--normalization-from", str(prepared["featured"])]) == 0
    capsys.readouterr()
    assert (second / NORMALIZATION_FILE).read_text(encoding="utf-8") == \
        (prepared["featured"] / NORMALIZATION_FILE).read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        add_cycle_features(prepared["dataset"], second, prepared["catalog"])
