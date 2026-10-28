"""拡散モデルの条件入力に足す、サイクルの忘却因子Dの要約(D4、2026-10-27)。

同じ区間・同じ補修サイクルの、サイクル開始から予測起点の日までの全計測に、λ = 1.00・0.98・0.95・0.90 の
忘却因子付き直線を引き、それぞれの「起点での水準」と「30日あたりの傾き」の8個を特徴にする
(`evaluation.forgetting_factor.d4_features`)。起点より後の計測は使わない。

既存のデータセットをコピーし、各splitに`cycle_features.npy`(物理値、(N, 8)、水準・傾きがλごとに交互)を
足すだけで、guide検索のやり直しは要らない。正規化の統計は`cycle_feature_normalization.json`に保存する。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_v2.datasets.normalization import Normalization
from acceleration_forecasting_v2.evaluation.feature_screening import _catalog_groups
from acceleration_forecasting_v2.evaluation.forgetting_factor import D4_LAMBDAS, d4_features

SPLITS = ("model_train", "model_validation", "inference")
CYCLE_FEATURE_NAME = "cycle_features"
NORMALIZATION_FILE = "cycle_feature_normalization.json"
SUMMARY_FILE = "cycle_features_summary.json"
FEATURE_DIM = 2 * len(D4_LAMBDAS)
SLOPE_CLIP_PERCENTILES = (0.5, 99.5)


def split_cycle_features(metadata, groups) -> np.ndarray:
    """metadataの各行(予測起点)の特徴(物理値、float32、shape (N, 8))。

    計測が1点もない起点は、起点の値(`current_acc_z_max`)だけの1点として扱う(傾き0)。
    """
    features = np.zeros((len(metadata), FEATURE_DIM), dtype=np.float32)
    for index, row in enumerate(metadata.itertuples(index=False)):
        dates, values = groups[str(row.dataset_id)]
        anchor = pd.Timestamp(row.anchor_date).normalize()
        stop = dates.searchsorted(anchor, side="right")
        past_dates, past_values = dates[:stop], values[:stop]
        if len(past_values) == 0:
            past_dates, past_values = pd.DatetimeIndex([anchor]), np.array([float(row.current_acc_z_max)])
        features[index] = d4_features(past_dates, past_values).reshape(-1)
    return features


def fit_cycle_feature_normalization(train_features, condition_normalization) -> dict:
    """model_trainの特徴から正規化の統計を作る。

    水準(偶数番目の列)は物理範囲にクリップして履歴と同じ平均・標準偏差で、傾き(奇数番目の列)は
    学習データの0.5%点〜99.5%点にクリップして履歴と同じ標準偏差で割る(平均は引かない)。
    `low`・`high`でクリップし、`shift`を引いて`scale`で割る。
    """
    train_features = np.asarray(train_features, dtype=np.float64)
    low, high, shift = [], [], []
    for column in range(train_features.shape[1]):
        if column % 2 == 0:
            low.append(float(PHYSICAL_MIN))
            high.append(float(PHYSICAL_MAX))
            shift.append(float(condition_normalization.mean))
        else:
            lo, hi = np.percentile(train_features[:, column], SLOPE_CLIP_PERCENTILES)
            low.append(float(lo))
            high.append(float(hi))
            shift.append(0.0)
    return {"low": low, "high": high, "shift": shift, "scale": [float(condition_normalization.std)] * len(low),
            "lambdas": list(D4_LAMBDAS), "source": "model_train_cycle_features"}


def normalize_cycle_features(features, normalization) -> np.ndarray:
    clipped = np.clip(np.asarray(features, dtype=np.float64), normalization["low"], normalization["high"])
    return ((clipped - np.asarray(normalization["shift"])) / np.asarray(normalization["scale"])).astype(np.float32)


def add_cycle_features(source_dir, output_dir, trend_catalog_path, *, normalization_from=None) -> dict:
    """`source_dir`のデータセットを`output_dir`にコピーし、存在する各splitに`cycle_features.npy`を足す。

    Args:
        trend_catalog_path: 全計測を持つ(合成値を含まない)trend_catalog.csv。
        normalization_from: 別のデータセットの`cycle_feature_normalization.json`を使う場合のそのディレクトリ。
            省略時は`source_dir/model_train`の特徴から作る(model_trainが無いとエラー)。
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} は既にあります。")
    shutil.copytree(source_dir, output_dir)
    splits = [split for split in SPLITS if (output_dir / split / "metadata.csv").is_file()]
    groups = _catalog_groups(trend_catalog_path)
    features = {split: split_cycle_features(
        pd.read_csv(output_dir / split / "metadata.csv", encoding="utf-8-sig"), groups) for split in splits}
    if normalization_from is not None:
        normalization = json.loads((Path(normalization_from) / NORMALIZATION_FILE).read_text(encoding="utf-8"))
    elif "model_train" in features:
        normalization = fit_cycle_feature_normalization(
            features["model_train"], Normalization.load(output_dir / "condition_normalization.json"))
    else:
        raise ValueError("model_trainが無いデータセットでは normalization_from が必要です。")
    (output_dir / NORMALIZATION_FILE).write_text(json.dumps(normalization, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    summary = {"lambdas": list(D4_LAMBDAS), "splits": {}}
    low, high = np.array(normalization["low"]), np.array(normalization["high"])
    for split, values in features.items():
        np.save(output_dir / split / f"{CYCLE_FEATURE_NAME}.npy", values)
        summary["splits"][split] = {
            "records": int(len(values)),
            "physical_percentiles_0_1_50_99_100": np.percentile(values, [0, 1, 50, 99, 100], axis=0).round(4).tolist(),
            "normalized_abs_max": float(np.abs(normalize_cycle_features(values, normalization)).max()),
            "clipped_fraction": float(np.mean((values < low) | (values > high))),
        }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
