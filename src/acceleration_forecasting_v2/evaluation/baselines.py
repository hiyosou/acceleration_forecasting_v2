"""評価の比較対象にする素朴なベースライン予測(学習不要の2種 + 履歴特徴のridge回帰)。

2026-10-13の再分析(GENERATIVE_MODULE_NEXT_STEPS.md §0.1)で、拡散モデルがこれらの素朴な
予測器に負けていたことが分かったため、`evaluate`で常に並べて比較する。guide平均
(`guide_baselines`、`baseline_MAE`)はguide自体の質の参考値で、モデルの合格基準には使わない。

入力はすべて物理値のまま扱う(`.npy`を直接読む、`evaluate.per_target_frame`と同じ流儀)。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_BASELINE_NAMES = ("persistence", "history_max", "history_ridge")
RIDGE_LAMBDA = 1.0


def _load_history(split_dir):
    values = np.load(split_dir / "history_values.npy").astype(float)
    masks = np.load(split_dir / "history_masks.npy") > 0
    return np.where(masks & np.isfinite(values), values, np.nan)


def current_values(history):
    """履歴の最後の有効値(=current値。実データでは`current_acc_z_max`と一致する)。"""
    current = np.full(len(history), np.nan)
    for index, row in enumerate(history):
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size:
            current[index] = row[valid[-1]]
    return current


def persistence_prediction(history, months=12):
    return np.repeat(current_values(history)[:, None], months, axis=1)


def history_max_prediction(history, months=12):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 有効月のないレコードはNaNのまま
        maximum = np.nanmax(history, axis=1)
    return np.repeat(maximum[:, None], months, axis=1)


def _anchor_months(metadata):
    if metadata is None or "anchor_date" not in metadata.columns:
        return None
    return pd.to_datetime(metadata["anchor_date"]).dt.month.to_numpy()


def history_features(history, anchor_months=None):
    """ridge用の特徴量: current、履歴の平均・最大・最小・傾き、有効月数、履歴の値(欠損はcurrentで補完)、
    anchor月のone-hot(1〜11月、anchor_monthsがあるときだけ)。先頭列は切片。"""
    current = current_values(history)
    valid = np.isfinite(history)
    filled = np.where(valid, history, current[:, None])
    positions = np.arange(history.shape[1], dtype=float)
    slopes = np.array([np.polyfit(positions[mask], row[mask], 1)[0] if mask.sum() >= 2 else 0.0
                       for row, mask in zip(history, valid)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        columns = [np.ones(len(history)), current, np.nanmean(history, 1), np.nanmax(history, 1),
                   np.nanmin(history, 1), valid.sum(1).astype(float), slopes]
    columns += [filled[:, month] for month in range(history.shape[1])]
    if anchor_months is not None:
        columns += [(anchor_months == month).astype(float) for month in range(1, 12)]
    return np.nan_to_num(np.column_stack(columns))


def fit_history_ridge(train_dir, *, use_anchor_month=True, ridge_lambda=RIDGE_LAMBDA):
    """`model_train`で月ごとのridge回帰(segment_weight付き、切片は正則化しない)を学習する。

    Returns:
        dict(coefficients (n_features, months), mean/std(標準化統計), use_anchor_month)。
    """
    train_dir = Path(train_dir)
    metadata = pd.read_csv(train_dir / "metadata.csv", encoding="utf-8-sig")
    months = _anchor_months(metadata) if use_anchor_month else None
    features = history_features(_load_history(train_dir), months)
    mean, std = features[:, 1:].mean(0), features[:, 1:].std(0) + 1e-9
    features[:, 1:] = (features[:, 1:] - mean) / std
    targets = np.load(train_dir / "target_values.npy").astype(float)
    target_masks = (np.load(train_dir / "target_masks.npy") > 0) & np.isfinite(targets)
    weights = np.load(train_dir / "segment_weights.npy").astype(float)
    penalty = ridge_lambda * np.eye(features.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.zeros((features.shape[1], targets.shape[1]))
    for month in range(targets.shape[1]):
        rows = target_masks[:, month]
        x, y, w = features[rows], targets[rows, month], weights[rows]
        coefficients[:, month] = np.linalg.solve(x.T @ (x * w[:, None]) + penalty, x.T @ (w * y))
    return {"coefficients": coefficients, "mean": mean, "std": std, "use_anchor_month": months is not None}


def predict_history_ridge(model, history, metadata=None):
    months = _anchor_months(metadata) if model["use_anchor_month"] else None
    if model["use_anchor_month"] and months is None:
        raise ValueError("学習時はanchor月を使ったが、予測対象のmetadataにanchor_date列がありません。")
    features = history_features(history, months)
    features[:, 1:] = (features[:, 1:] - model["mean"]) / model["std"]
    return features @ model["coefficients"]


def baseline_predictions(dataset_dir, split="inference"):
    """指定splitの各ベースラインの12か月予測(物理値、shape (N, 12))を返す。

    `history_ridge`は同じdataset_dirの`model_train`で学習する。`model_train`が無い場合は
    warningを出して省略する。anchor月は、学習・予測の両方のmetadataに`anchor_date`列が
    あるときだけ使う。
    """
    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / split
    history = _load_history(split_dir)
    months = np.load(split_dir / "target_values.npy").shape[1] if (split_dir / "target_values.npy").is_file() else 12
    predictions = {
        "persistence": persistence_prediction(history, months),
        "history_max": history_max_prediction(history, months),
    }
    train_dir = dataset_dir / "model_train"
    if not (train_dir / "history_values.npy").is_file():
        warnings.warn("model_trainが無いため、history_ridgeベースラインを省略します。", stacklevel=2)
        return predictions
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    use_anchor_month = "anchor_date" in metadata.columns
    model = fit_history_ridge(train_dir, use_anchor_month=use_anchor_month)
    predictions["history_ridge"] = predict_history_ridge(model, history, metadata)
    return predictions
