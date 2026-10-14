"""evaluation.baselines(素朴なベースライン予測)の単体テスト。"""

import shutil

import numpy as np
import pandas as pd
import pytest

from test_training import _raw_split

from acceleration_forecasting_v2.datasets.build import RawSplit, write_dataset
from acceleration_forecasting_v2.evaluation.baselines import (
    baseline_predictions, current_values, history_max_prediction, persistence_prediction,
)


def _history():
    # 2件目は最後の月が欠損(mask=0)、3件目は全月欠損。
    return np.array([
        [1.0, 2.0, 3.0, 2.5, 1.5, 1.2],
        [1.0, 4.0, 2.0, 2.2, 1.8, np.nan],
        [np.nan] * 6,
    ])


def test_current_value_is_last_valid_history_month():
    current = current_values(_history())
    assert current[0] == pytest.approx(1.2)
    assert current[1] == pytest.approx(1.8)
    assert np.isnan(current[2])


def test_persistence_and_history_max_are_flat_over_the_horizon():
    persistence = persistence_prediction(_history(), months=12)
    maximum = history_max_prediction(_history(), months=12)
    assert persistence.shape == maximum.shape == (3, 12)
    np.testing.assert_allclose(persistence[0], 1.2)
    np.testing.assert_allclose(persistence[1], 1.8)
    np.testing.assert_allclose(maximum[0], 3.0)
    np.testing.assert_allclose(maximum[1], 4.0)
    assert np.isnan(maximum[2]).all()


def _write(tmp_path, *, with_train=True, n_train=200):
    inference = RawSplit(arrays=_raw_split(20, seed=3).arrays, metadata=pd.DataFrame({
        "trend_id": [f"t{i}" for i in range(20)], "anchor_date": "2023-05-01"}))
    dataset_dir = tmp_path / "dataset"
    write_dataset({"model_train": _raw_split(n_train, seed=1), "model_validation": _raw_split(16, seed=2),
                   "inference": inference}, dataset_dir)
    if not with_train:
        # write_datasetはmodel_trainを必須にしているため、書き出した後に消す。
        shutil.rmtree(dataset_dir / "model_train")
    return dataset_dir


def test_history_ridge_learns_a_linear_rule(tmp_path):
    # _raw_splitの正解は「履歴平均 + 月ごとの固定の傾き」で、履歴特徴の線形結合で表せる。
    dataset_dir = _write(tmp_path)
    predictions = baseline_predictions(dataset_dir, "inference")
    target = np.load(dataset_dir / "inference" / "target_values.npy")
    assert set(predictions) == {"persistence", "history_max", "history_ridge"}
    assert np.abs(predictions["history_ridge"] - target).mean() < 0.02
    # 学習不要のベースラインは傾きを表せないので、ridgeより明確に悪い。
    assert np.abs(predictions["persistence"] - target).mean() > 0.1


def test_history_ridge_is_skipped_with_warning_without_model_train(tmp_path):
    dataset_dir = _write(tmp_path, with_train=False)
    with pytest.warns(UserWarning, match="history_ridge"):
        predictions = baseline_predictions(dataset_dir, "inference")
    assert set(predictions) == {"persistence", "history_max"}
