"""忘却因子付きの線形近似(拡散モデルと並べて示す比較用の予測、2026-10-22)。

区間の同じ補修サイクル(dataset_id)の、サイクル開始から予測起点までの全計測(trend_catalogの
`current_acc_z_max`、1か月に約8回)に、重み λ^(N−i)(N=データ数、i=古い順の番号、最新=1)の
重み付き最小二乗で直線を当てはめ、そのまま予測期間へ延ばす(時間軸は日)。λ=1.00〜0.90の
0.01刻み(11本)の最小〜最大を、忘却係数を変えたときの幅として示す(誤差分布は付けない)。
2026-10-21〜22のユーザーとの検討で、全計測・サイクル開始から使う版(会話中の呼び名「版D」)に決めた。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN

DEFAULT_LAMBDAS = tuple(round(1.00 - 0.01 * step, 2) for step in range(11))
# 拡散モデルの条件入力に足す特徴(D4、2026-10-27のridgeでのふるいで決定)に使うλ
D4_LAMBDAS = (1.00, 0.98, 0.95, 0.90)


def cycle_measurements(trend_catalog: pd.DataFrame, dataset_id, anchor_date) -> pd.DataFrame:
    """同じdataset_id(=1つの補修サイクル)の、予測起点までの全計測を日付昇順で返す。

    Returns:
        列`measurement_date`(datetime64)・`value`(float、`current_acc_z_max`)のDataFrame。
    """
    rows = trend_catalog.loc[trend_catalog["dataset_id"].astype(str) == str(dataset_id)]
    frame = pd.DataFrame({
        "measurement_date": pd.to_datetime(rows["measurement_date"]).dt.normalize(),
        "value": pd.to_numeric(rows["current_acc_z_max"], errors="coerce"),
    })
    frame = frame.loc[(frame["measurement_date"] <= pd.Timestamp(anchor_date).normalize()) & frame["value"].notna()]
    return frame.sort_values("measurement_date", kind="mergesort").reset_index(drop=True)


def forgetting_factor_coefficients(dates, values, lam):
    """重み λ^(N−i) の重み付き最小二乗直線の(最新の計測日での値, 1日あたりの傾き)。クリップはしない。

    データが1点だけのときは傾き0(その値で一定)とする。
    """
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("忘却因子の線形近似には1点以上のデータが必要です。")
    if len(values) == 1:
        return float(values[0]), 0.0
    x = (dates - dates.max()).days.to_numpy(float)
    weights = float(lam) ** (len(values) - 1 - np.arange(len(values)))
    design = np.column_stack([np.ones(len(values)), x])
    intercept, slope = np.linalg.solve(design.T @ (design * weights[:, None]), design.T @ (weights * values))
    return float(intercept), float(slope)


def forgetting_factor_line(dates, values, grid, lam):
    """重み λ^(N−i) の重み付き最小二乗直線を`grid`(日付)上で評価し、物理値の範囲に収める。

    データが1点だけのときは傾き0(その値で一定)とする。
    """
    dates, grid = pd.DatetimeIndex(dates), pd.DatetimeIndex(grid)
    intercept, slope = forgetting_factor_coefficients(dates, values, lam)
    grid_x = (grid - dates.max()).days.to_numpy(float)
    return np.clip(intercept + slope * grid_x, PHYSICAL_MIN, PHYSICAL_MAX)


def d4_features(dates, values, lambdas=D4_LAMBDAS):
    """λごとの直線の(最新の計測日での水準, 30日あたりの傾き)。shape (len(lambdas), 2)。クリップはしない。

    `dates`・`values`は、予測起点の日までの同じサイクルの全計測(日付昇順)。
    """
    return np.array([[intercept, slope * 30.0] for intercept, slope in
                     (forgetting_factor_coefficients(dates, values, lam) for lam in lambdas)])


def forgetting_factor_lines(dates, values, grid, lambdas=DEFAULT_LAMBDAS):
    """λごとの直線を積み重ねて返す。shape (len(lambdas), len(grid))。"""
    return np.stack([forgetting_factor_line(dates, values, grid, lam) for lam in lambdas])


def forecast_grid(anchor_date, months=12):
    """予測月の読み取り日(起点の翌月から各月1日)。正解の計測日を選ぶ基準日と同じ。"""
    first = pd.Timestamp(anchor_date).normalize().replace(day=1) + pd.DateOffset(months=1)
    return pd.DatetimeIndex([first + pd.DateOffset(months=offset) for offset in range(months)])


def predict_forgetting_factor(dataset_dir, trend_catalog_path, output_dir, *, split="inference",
                              lambdas=DEFAULT_LAMBDAS, months=12):
    """データセットの各起点について版D(λごとに1本)を計算し、`predict`と同じ形式で書き出す。

    samples.npz(trend_ids・samples (N, len(lambdas), months))・predictions.csv(median/p10/p90/std)・
    prediction_run.json。入力はtrend_catalogの、同じdataset_idのサイクル開始から起点までの全計測。
    """
    from pathlib import Path

    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(dataset_dir / split / "metadata.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(trend_catalog_path, encoding="utf-8-sig",
                          usecols=["dataset_id", "measurement_date", "current_acc_z_max"])
    catalog["dataset_id"] = catalog["dataset_id"].astype(str)
    groups = {dataset_id: group for dataset_id, group in catalog.groupby("dataset_id", sort=False)}
    samples = np.zeros((len(metadata), len(lambdas), months), dtype=np.float32)
    input_counts = np.zeros(len(metadata), dtype=int)
    for index, row in enumerate(metadata.itertuples(index=False)):
        measurements = cycle_measurements(groups[str(row.dataset_id)], row.dataset_id, row.anchor_date)
        samples[index] = forgetting_factor_lines(measurements["measurement_date"], measurements["value"],
                                                 forecast_grid(row.anchor_date, months), lambdas)
        input_counts[index] = len(measurements)

    median = np.median(samples, axis=1)
    p10, p90 = np.percentile(samples, [10, 90], axis=1)
    std = samples.std(axis=1)
    rows = [{"trend_id": metadata.iloc[index]["trend_id"], "dataset_id": metadata.iloc[index]["dataset_id"],
             "month_index": month + 1, "prediction_median": float(median[index, month]),
             "prediction_p10": float(p10[index, month]), "prediction_p90": float(p90[index, month]),
             "prediction_std": float(std[index, month])}
            for index in range(len(samples)) for month in range(months)]
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(output_dir / "samples.npz", trend_ids=metadata["trend_id"].astype(str).to_numpy().astype("U"),
                        samples=samples)
    summary = {"split": split, "record_count": int(len(samples)), "method": "forgetting_factor_D",
               "lambdas": [float(lam) for lam in lambdas], "trend_catalog": str(trend_catalog_path),
               "input_points_median": float(np.median(input_counts)) if len(input_counts) else 0.0}
    (output_dir / "prediction_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
