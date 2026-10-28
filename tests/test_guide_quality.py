"""evaluation.guide_quality(guideの質 × 生成精度の基準曲線)の検証。"""

import json

import numpy as np
import pandas as pd

from test_training import _raw_split

from acceleration_forecasting_v2.datasets.build import RawSplit, write_dataset
from acceleration_forecasting_v2.evaluation.baselines import baseline_predictions
from acceleration_forecasting_v2.evaluation.guide_quality import (
    fit_history_guide_ridge, guide_quality_row, plot_guide_quality_curve, predict_history_guide_ridge,
)


def _split_with_noisy_target(n, *, seed):
    """正解 = 履歴平均 + レコードごとの傾き(履歴からは分からない)。guide平均は正解そのもの。"""
    raw = _raw_split(n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    arrays = dict(raw.arrays)
    slope = rng.normal(scale=0.05, size=(n, 1)) * np.arange(12)[None, :]
    target = (arrays["target_values"] + slope).astype(np.float32)
    arrays["target_values"] = target
    arrays["guide_baselines"] = target
    metadata = pd.DataFrame({"trend_id": [f"t{i}" for i in range(n)], "dataset_id": [f"d{i % 10}" for i in range(n)],
                             "anchor_date": "2023-05-01"})
    return RawSplit(arrays=arrays, metadata=metadata)


def _write(tmp_path):
    dataset_dir = tmp_path / "dataset_lam100"
    write_dataset({"model_train": _split_with_noisy_target(300, seed=1),
                   "model_validation": _split_with_noisy_target(20, seed=2),
                   "inference": _split_with_noisy_target(40, seed=3)}, dataset_dir)
    return dataset_dir


def test_history_guide_ridge_uses_informative_guides(tmp_path):
    dataset_dir = _write(tmp_path)
    model = fit_history_guide_ridge(dataset_dir / "model_train")
    prediction = predict_history_guide_ridge(model, dataset_dir / "inference")
    target = np.load(dataset_dir / "inference" / "target_values.npy")
    history_only = baseline_predictions(dataset_dir, "inference")["history_ridge"]
    assert prediction.shape == (40, 12)
    assert np.abs(prediction - target).mean() < 0.5 * np.abs(history_only - target).mean()


def test_guide_quality_row_and_plot(tmp_path):
    dataset_dir = _write(tmp_path)
    trend_ids = [f"t{i}" for i in range(40)]
    evaluation_dir = tmp_path / "eval"
    evaluation_dir.mkdir()
    pd.DataFrame({"trend_id": trend_ids, "dataset_id": [f"d{i % 10}" for i in range(40)],
                  "MAE": 0.2, "baseline_MAE": 0.0, "history_ridge_MAE": 0.3, "mean_error": 0.01,
                  "correlation": 0.5}).to_csv(evaluation_dir / "evaluation_per_target.csv", index=False,
                                              encoding="utf-8-sig")
    fidelity_dir = tmp_path / "fidelity"
    fidelity_dir.mkdir()
    (fidelity_dir / "generated_vs_guide_fidelity.json").write_text(json.dumps({
        "guide_vs_target_MAE": {"mean": 0.0}, "generated_vs_guide_MAE": {"mean": 0.2}}), encoding="utf-8")

    row = guide_quality_row(dataset_dir, evaluation_dir, fidelity_dir, lam=1.0, checkpoint_name="best")
    assert row["records"] == 40 and row["segments"] == 10
    assert row["model_MAE"] == 0.2 and row["history_ridge_MAE"] == 0.3
    assert row["history_guide_ridge_MAE"] < 0.1
    assert np.isfinite(row["history_guide_ridge_MAE_segment"])

    output = plot_guide_quality_curve([row, {**row, "lambda": 0.5, "guide_vs_target_MAE": 0.1}],
                                      tmp_path / "curve.png")
    assert output.is_file() and output.stat().st_size > 0
