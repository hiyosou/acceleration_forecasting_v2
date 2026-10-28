"""入力特徴の候補をridgeでふるう処理(evaluation/feature_screening.py)のテスト(合成データ、CPU、小規模)。"""

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.cli import main
from acceleration_forecasting_v2.evaluation.baselines import fit_history_ridge, predict_history_ridge, _load_history
from acceleration_forecasting_v2.evaluation.feature_screening import (
    CANDIDATES, SPLITS, SplitData, _catalog_groups, candidate_matrix, compute_split_features, design_matrix, fair_crps,
    fit_ridge_candidate, record_features, residual_ensemble, screen_features,
)
from acceleration_forecasting_v2.evaluation.forgetting_factor import (
    DEFAULT_LAMBDAS, forgetting_factor_coefficients, forgetting_factor_line, predict_forgetting_factor,
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("screening")
    artifacts, dataset = root / "artifacts", root / "dataset"
    build_artifacts(artifacts, n_datasets=12, months=22, seed=0)
    assert main(["prepare", "--artifact-dir", str(artifacts), "--output-dir", str(dataset), "--device", "cpu",
                 "--inference-max-per-segment", "10"]) == 0
    return {"root": root, "artifacts": artifacts, "dataset": dataset,
            "catalog": artifacts / "trend_catalog.csv"}


def _series(days=240, step=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.Timestamp("2023-01-05") + pd.to_timedelta(np.arange(0, days, step), unit="D"))
    return dates, 1.0 + 0.004 * np.arange(len(dates)) + rng.normal(0, 0.05, len(dates))


def test_coefficients_match_the_line_at_the_last_measurement_and_one_month_later():
    dates, values = _series()
    for lam in (1.0, 0.95):
        level, slope = forgetting_factor_coefficients(dates, values, lam)
        line = forgetting_factor_line(dates, values, [dates.max(), dates.max() + pd.Timedelta(days=30)], lam)
        np.testing.assert_allclose(line, [level, level + 30 * slope], rtol=1e-9)
    assert forgetting_factor_coefficients(dates[:1], values[:1], 0.9) == (pytest.approx(values[0]), 0.0)


def test_record_features_ignore_measurements_after_the_anchor():
    dates, values = _series()
    anchor = dates[60]
    before = record_features(dates, values, anchor, float(values[60]))
    changed = values.copy()
    changed[61:] += 5.0
    after = record_features(dates, changed, anchor, float(values[60]))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_record_features_shapes_and_current_month_uses_month_start_to_anchor():
    dates, values = _series()
    anchor = dates[60]
    features = record_features(dates, values, anchor, float(values[60]))
    assert features["D_samples"].shape == (len(DEFAULT_LAMBDAS), 12)
    assert features["D_coef"].shape == (4, 2) and features["H"].shape == (6, 7) and features["R30"].shape == (2,)
    month_start = anchor.replace(day=1)
    in_month = values[(dates >= month_start) & (dates <= anchor)]
    median, count, maximum, minimum, std, mean, trimmed = features["H"][-1]
    assert (median, count, maximum, minimum) == (np.median(in_month), len(in_month), in_month.max(), in_month.min())
    assert std == pytest.approx(in_month.std()) and mean == pytest.approx(in_month.mean())
    kept = np.sort(in_month)[int(0.2 * len(in_month)):len(in_month) - int(0.2 * len(in_month))]
    assert trimmed == pytest.approx(kept.mean())


def test_months_without_measurements_use_current_and_count_zero():
    dates = pd.DatetimeIndex(["2023-06-10", "2023-06-20"])
    features = record_features(dates, np.array([1.0, 2.0]), "2023-06-20", 2.0)
    assert features["H"][0].tolist() == [2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0]  # 5か月前は計測なし
    assert features["H"][-1][1] == 2.0


def test_single_measurement_per_month_gives_that_value_as_median():
    dates = pd.DatetimeIndex(pd.date_range("2023-01-10", periods=6, freq="MS") + pd.Timedelta(days=9))
    values = np.arange(6, dtype=float) + 1
    features = record_features(dates, values, dates[-1], float(values[-1]))
    np.testing.assert_array_equal(features["H"][:, 0], values)
    np.testing.assert_array_equal(features["H"][:, 1], np.ones(6))
    assert features["H"][:, 4].tolist() == [0.0] * 6


def test_d_median_matches_predict_forgetting_factor(prepared, tmp_path):
    groups = _catalog_groups(prepared["catalog"])
    features = compute_split_features(prepared["dataset"], "inference", prepared["catalog"], groups=groups)
    predict_forgetting_factor(prepared["dataset"], prepared["catalog"], tmp_path / "ff")
    with np.load(tmp_path / "ff" / "samples.npz") as npz:
        np.testing.assert_allclose(features["D_samples"], npz["samples"], rtol=1e-5, atol=1e-6)
    assert candidate_matrix("D1", features).shape == (len(features["D_samples"]), 12)


def test_candidate_matrix_shapes(prepared):
    features = compute_split_features(prepared["dataset"], "inference", prepared["catalog"])
    n = len(features["D_samples"])
    widths = {name: candidate_matrix(name, features).shape for name in CANDIDATES}
    assert widths == {"D1": (n, 12), "D2": (n, 17), "D3": (n, 132), "D4": (n, 8),
                      "H1": (n, 12), "H2": (n, 24), "H3": (n, 30), "H4": (n, 2),
                      "E1": (n, 1), "E2": (n, 1), "E3": (n, 12)}
    assert candidate_matrix("B0", features).shape == (n, 0)
    assert candidate_matrix("C:D2+H3", features).shape == (n, 47)
    assert candidate_matrix("D4+E1+E2", features).shape == (n, 10)
    assert candidate_matrix("Rmedian", features).shape == (n, 0)  # 置き換えは追加特徴ではない


def test_b0_ridge_matches_the_existing_history_ridge(prepared):
    dataset = prepared["dataset"]
    data = {split: SplitData(dataset, split) for split in ("model_train", "model_validation")}
    features = {split: compute_split_features(dataset, split, prepared["catalog"]) for split in data}
    best = fit_ridge_candidate("B0", data["model_train"], data["model_validation"], features)
    validation = data["model_validation"]
    reference = predict_history_ridge(fit_history_ridge(data["model_train"].dir, ridge_lambda=best["ridge_lambda"]),
                                      _load_history(validation.dir), validation.metadata)
    np.testing.assert_allclose(best["validation_prediction"], reference, rtol=1e-7, atol=1e-9)


def test_fair_crps_and_residual_ensemble():
    rng = np.random.default_rng(0)
    pool = [rng.normal(0, 0.2, 500) for _ in range(12)]
    prediction = np.full((4, 12), 1.5)
    ensemble = residual_ensemble(prediction, pool, size=50)
    assert ensemble.shape == (4, 50, 12)
    target = np.full((4, 12), 1.5)
    crps = fair_crps(ensemble, target)
    assert crps.shape == (4, 12) and np.all(crps > -1e-9) and crps.mean() < 0.2
    assert fair_crps(np.repeat(prediction[:, None, :], 5, axis=1), target).max() == pytest.approx(0.0)


def test_screen_features_writes_tables_and_summary(prepared, tmp_path):
    output = tmp_path / "screening"
    # 合成データでは正解8か月以上の起点が少ないので、MLPは省いて表の形とキャッシュだけ確かめる
    result = screen_features(prepared["dataset"], prepared["catalog"], output, mlp=False)
    validation, inference = result["validation"], result["inference"]
    assert validation["候補"].tolist()[:2] == ["B0", "D1"] and len(validation) == 1 + 4 + 4 + 1
    assert inference["候補"].tolist()[0] == "ridge B0" and len(inference) == 4
    assert {"MAE", "上昇件 AUC", "横ばい件 MAE", "CRPS"} <= set(inference.columns)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["combined"] == f"C:{summary['best_D']}+{summary['best_H']}"
    for split in SPLITS:
        assert (output / f"features_{split}.npz").is_file()
    again = screen_features(prepared["dataset"], prepared["catalog"], output, mlp=False)  # キャッシュを読む
    pd.testing.assert_frame_equal(again["inference"], inference)


def test_cli_screen_features(prepared, tmp_path, capsys):
    code = main(["screen-features", "--dataset-dir", str(prepared["dataset"]), "--trend-catalog-path",
                 str(prepared["catalog"]), "--output-dir", str(tmp_path / "cli"), "--no-mlp"])
    assert code == 0 and json.loads(capsys.readouterr().out)["mlp"] is False
    assert (tmp_path / "cli" / "inference_table.csv").is_file()


def test_screen_features_with_mlp_runs(prepared, tmp_path):
    result = screen_features(prepared["dataset"], prepared["catalog"], tmp_path / "mlp", mlp=True)
    assert result["inference"]["候補"].str.startswith("MLP").sum() == 3  # B0・最良のD・組み合わせ


def test_replacement_uses_monthly_aggregates_in_place_of_the_history_values(prepared):
    data = SplitData(prepared["dataset"], "inference")
    features = compute_split_features(prepared["dataset"], "inference", prepared["catalog"])
    plain = design_matrix("B0", data, features)
    np.testing.assert_array_equal(plain, data.base)
    for name, column in (("Rmedian", 0), ("Rmean", 5), ("Rtrim", 6)):
        replaced = design_matrix(name, data, features)
        assert replaced.shape == data.base.shape and not np.allclose(replaced, plain)
    both = design_matrix("C:Rmedian+D4+E1", data, features)
    assert both.shape[1] == data.base.shape[1] + 9


def test_residual_std_is_zero_for_a_perfect_line_and_positive_with_noise():
    dates = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=30, freq="5D"))
    line = 1.0 + 0.01 * np.arange(30)
    assert record_features(dates, line, dates[-1], float(line[-1]))["resid_std"][0] == pytest.approx(0.0, abs=1e-9)
    noisy = line + np.random.default_rng(0).normal(0, 0.1, 30)
    assert record_features(dates, noisy, dates[-1], float(noisy[-1]))["resid_std"][0] > 0.05


def test_screen_features_stage2_and_cli(prepared, tmp_path, capsys):
    result = screen_features(prepared["dataset"], prepared["catalog"], tmp_path / "stage2", mlp=False, stage=2)
    names = result["validation"]["候補"].tolist()
    assert names[:6] == ["B0", "D4", "D4+E1", "D4+E2", "D4+E1+E2", "E3"] and "Rmedian" in names
    assert result["summary"]["stage"] == 2
    code = main(["screen-features", "--dataset-dir", str(prepared["dataset"]), "--trend-catalog-path",
                 str(prepared["catalog"]), "--output-dir", str(tmp_path / "cli2"), "--no-mlp", "--stage", "2"])
    assert code == 0 and json.loads(capsys.readouterr().out)["stage"] == 2
