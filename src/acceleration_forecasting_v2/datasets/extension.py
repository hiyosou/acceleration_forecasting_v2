"""学習用の正解系列を、忘却因子付き線形近似で「補修から2年」まで延長する(2026-10-22、ユーザー提案)。

補修サイクル(dataset_id)は補修やデータ終端で短く切れるため、正解12か月がそろわず学習に使えない
予測起点が多い。そこでサイクルごとに次の手順で合成値を作り、正解の欠測月を埋める。

1. サイクルの起点 = 前回の補修日(`previous_maintenance_date`)。無ければ最初の計測日。
2. 起点から実測の最後までが`min_span_months`か月未満のサイクルは延長しない(直線が妥当に引けないため)。
3. λ ~ 一様(0.90, 1.00) を1つ選び、サイクル内の全計測に重み λ^(N−i) の最小二乗直線を引く
   (`evaluation.forgetting_factor.forgetting_factor_line`、会話中の「版D」と同じ近似)。
4. 起点の翌月から「起点+`horizon_months`か月」までの各月1日で直線を読み、同じλの直線に対する
   実測の残差から無作為に選んだ揺らぎを足して[0.1, 6.0]に収める。
5. 予測起点ごとの正解(trend_catalogのfuture_*)で、実測が無い月(mask=0)にこの値を入れ、
   `future_synthetic_mask`で合成の月を区別する。入力の6か月(履歴)には使わない。

seedは「全体seed + dataset_id」から決めるので、同じサイクルはデータセットと検索DBで同じ合成系列になる。
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import FORECAST_MONTHS, PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_v2.evaluation.forgetting_factor import forgetting_factor_line

HORIZON_MONTHS = 24
MIN_SPAN_MONTHS = 6
MIN_MONTHS_AFTER_MAINTENANCE = 5
LAMBDA_RANGE = (0.90, 1.00)
SEARCH_DAYS = 15


def month_index(date) -> int:
    date = pd.Timestamp(date)
    return date.year * 12 + date.month - 1


def month_start(index) -> pd.Timestamp:
    year, month = divmod(int(index), 12)
    return pd.Timestamp(year=year, month=month + 1, day=1)


def cycle_start(rows: pd.DataFrame) -> pd.Timestamp:
    """サイクルの起点(前回の補修日、無ければ最初の計測日)。"""
    maintenance = pd.to_datetime(rows["previous_maintenance_date"], errors="coerce").dropna()
    if len(maintenance):
        return pd.Timestamp(maintenance.iloc[0]).normalize()
    return pd.Timestamp(pd.to_datetime(rows["measurement_date"]).min()).normalize()


def cycle_rng(seed, dataset_id) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(str(dataset_id).encode("utf-8"))])


def extend_cycle(dates, values, start, *, rng, horizon_months=HORIZON_MONTHS, min_span_months=MIN_SPAN_MONTHS,
                 lam=None, fluctuation=True, search_days=SEARCH_DAYS):
    """1サイクル分の合成値を作る。

    Returns:
        (frame, lam)。frameは列`month`(各月1日)・`kind`・`value`。kindは
        "observed"(±`search_days`日以内に計測がある月)/"gap"(実測の最後の月までの欠測月)/
        "extension"(実測の最後の月より後)。延長しないサイクルは空のframeとNone。
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    dates, values = dates[keep], values[keep]
    order = np.argsort(dates.values, kind="mergesort")
    dates, values = dates[order], values[order]
    empty = pd.DataFrame({"month": pd.DatetimeIndex([]), "kind": pd.Series([], dtype=str), "value": np.zeros(0)})
    if not len(values):
        return empty, None
    start_index, last_index = month_index(start), month_index(dates.max())
    if last_index - start_index < int(min_span_months):
        return empty, None

    lam = float(rng.uniform(*LAMBDA_RANGE)) if lam is None else float(lam)
    indices = np.arange(start_index + 1, start_index + int(horizon_months) + 1)
    grid = pd.DatetimeIndex([month_start(index) for index in indices])
    line = forgetting_factor_line(dates, values, grid, lam)
    residuals = values - forgetting_factor_line(dates, values, dates, lam)
    noise = rng.choice(residuals, size=len(grid)) if fluctuation else np.zeros(len(grid))
    observed = np.abs((dates.values[None, :] - grid.values[:, None]) / np.timedelta64(1, "D")).min(axis=1) <= search_days
    kind = np.where(observed, "observed", np.where(indices <= last_index, "gap", "extension"))
    frame = pd.DataFrame({"month": grid, "kind": kind, "value": np.clip(line + noise, PHYSICAL_MIN, PHYSICAL_MAX)})
    return frame, lam


def build_cycle_extensions(raw_rows: pd.DataFrame, split_of_dataset: dict, *, seed=42,
                           horizon_months=HORIZON_MONTHS, min_span_months=MIN_SPAN_MONTHS):
    """全サイクルの合成値を作る。

    Args:
        raw_rows: 元のtrend CSVの全計測(`TrendCatalog.rows`と同じ形。列 dataset_id・measurement_date・
            acc_z_max・previous_maintenance_date)。
        split_of_dataset: dataset_id → model_split。ここに無いdataset_idは扱わない。

    Returns:
        (cycles, extensions)。cyclesはサイクルごとの1行(起点・最後の計測日・延長したか・λ)、
        extensionsは延長したサイクルの各月の1行。
    """
    cycles, frames = [], []
    for dataset_id, rows in raw_rows.groupby(raw_rows["dataset_id"].astype(str), sort=True):
        split = split_of_dataset.get(dataset_id)
        if split is None:
            continue
        start = cycle_start(rows)
        frame, lam = extend_cycle(rows["measurement_date"], rows["acc_z_max"], start, rng=cycle_rng(seed, dataset_id),
                                  horizon_months=horizon_months, min_span_months=min_span_months)
        cycles.append({
            "dataset_id": dataset_id, "split": split, "cycle_start": start.strftime("%Y-%m-%d"),
            "last_measurement_date": pd.to_datetime(rows["measurement_date"]).max().strftime("%Y-%m-%d"),
            "measurement_count": int(len(rows)), "extended": lam is not None,
            "lambda": lam if lam is not None else np.nan, "seed": int(seed),
        })
        if lam is not None:
            frame.insert(0, "dataset_id", dataset_id)
            frame.insert(1, "split", split)
            frame["lambda"], frame["seed"] = lam, int(seed)
            frames.append(frame)
    columns = ["dataset_id", "split", "month", "kind", "value", "lambda", "seed"]
    extensions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    extensions["month"] = pd.to_datetime(extensions["month"]).dt.strftime("%Y-%m-%d")
    return pd.DataFrame(cycles), extensions[columns]


def _loads(text):
    return json.loads(text) if isinstance(text, str) and text else []


def _dumps(values):
    return json.dumps([None if value is None or (isinstance(value, float) and np.isnan(value)) else value
                       for value in values], ensure_ascii=False, separators=(",", ":"))


def apply_extensions_to_catalog(catalog: pd.DataFrame, cycles: pd.DataFrame, extensions: pd.DataFrame,
                                *, forecast_months=FORECAST_MONTHS) -> pd.DataFrame:
    """trend_catalogの各行の正解(future_*)で、実測が無い月を合成値で埋める。

    埋めた月は future_mask=1・selected_dates=その月の1日・future_synthetic_mask=1 にする。
    `cycle_start`列(サイクルの起点)も足す。入力の履歴は変えない。
    """
    lookup = {(str(row.dataset_id), month_index(row.month)): float(row.value)
              for row in extensions.itertuples(index=False)}
    starts = dict(zip(cycles["dataset_id"].astype(str), cycles["cycle_start"]))
    out = catalog.copy()
    new_values, new_masks, new_dates, synthetic_masks = [], [], [], []
    for row in out.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        values, mask, dates = _loads(row.future_values), _loads(row.future_mask), _loads(row.selected_dates)
        if len(values) != forecast_months or len(mask) != forecast_months:
            raise ValueError(f"trend_id={row.trend_id} の future_* の長さが {forecast_months} ではありません。")
        dates = dates if len(dates) == forecast_months else [None] * forecast_months
        synthetic = [0] * forecast_months
        first = month_index(row.measurement_date) + 1
        for offset in range(forecast_months):
            if mask[offset]:
                continue
            value = lookup.get((dataset_id, first + offset))
            if value is None:
                continue
            values[offset], mask[offset], synthetic[offset] = value, 1, 1
            dates[offset] = month_start(first + offset).strftime("%Y-%m-%d")
        new_values.append(_dumps(values))
        new_masks.append(_dumps(mask))
        new_dates.append(_dumps(dates))
        synthetic_masks.append(_dumps(synthetic))
    out["future_values"], out["future_mask"], out["selected_dates"] = new_values, new_masks, new_dates
    out["future_synthetic_mask"] = synthetic_masks
    out["cycle_start"] = out["dataset_id"].astype(str).map(starts).fillna("")
    return out


def strip_synthetic(catalog: pd.DataFrame, keep: pd.Series) -> pd.DataFrame:
    """`keep`がFalseの行の合成の月を元の欠測(値None・mask 0)に戻す。"""
    if "future_synthetic_mask" not in catalog.columns:
        return catalog
    out = catalog.copy()
    for position in np.flatnonzero(~keep.to_numpy(dtype=bool)):
        synthetic = _loads(out.iat[position, out.columns.get_loc("future_synthetic_mask")])
        if not any(synthetic):
            continue
        columns = {name: out.columns.get_loc(name) for name in ("future_values", "future_mask", "selected_dates")}
        values, mask, dates = (_loads(out.iat[position, columns[name]]) for name in columns)
        for offset, flag in enumerate(synthetic):
            if flag:
                values[offset], mask[offset], dates[offset] = None, 0, None
        out.iat[position, columns["future_values"]] = _dumps(values)
        out.iat[position, columns["future_mask"]] = _dumps(mask)
        out.iat[position, columns["selected_dates"]] = _dumps(dates)
        out.iat[position, out.columns.get_loc("future_synthetic_mask")] = _dumps([0] * len(synthetic))
    return out


def build_extended_artifacts(artifact_dir, output_dir, *, trend_dir, seed=42, horizon_months=HORIZON_MONTHS,
                             min_span_months=MIN_SPAN_MONTHS) -> dict:
    """retrievalの成果物から延長版の成果物を別ディレクトリに作る(元の成果物は変えない)。

    書き出すもの:
      cycles.csv / cycle_extensions.csv: サイクルごとの起点・λと、各月の合成値。
      trend_catalog.csv: 全splitの正解を合成値で埋めたもの(`future_synthetic_mask`・`cycle_start`列付き)。
        どのsplitに合成値を使うかは`prepare_datasets(extend_splits=...)`で選ぶ。
      vector_database.sqlite: 元のDBのコピーで、trendsの将来値を合成値入りに置き換えたもの
        (DBにはmodel_train/model_validationだけが入っている)。合成値はサイクル内の全計測から作るので、
        合成の月を持つ行の`guide_available_date`は「サイクルの最後の計測日+15日」より前にしない
        (近傍区間のguideに、問い合わせ日より後の計測が混ざらないようにするため)。
    """
    import shutil
    from pathlib import Path

    from acceleration_forecasting_v2.retrieval.database import connect, update_trend_futures
    from acceleration_forecasting_v2.retrieval.trends import TrendCatalog

    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    if artifact_dir.resolve() == output_dir.resolve():
        raise ValueError("出力先は元の成果物と別ディレクトリにしてください。")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig").drop_duplicates("dataset_id")
    split_of_dataset = dict(zip(manifest["dataset_id"].astype(str), manifest["model_split"]))
    raw = TrendCatalog.from_directory(trend_dir).rows
    cycles, extensions = build_cycle_extensions(raw, split_of_dataset, seed=seed, horizon_months=horizon_months,
                                                min_span_months=min_span_months)
    catalog = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")
    missing = sorted(set(catalog["dataset_id"].astype(str)) - set(cycles["dataset_id"]))
    if missing:
        raise ValueError(f"元のtrend CSVに無いdataset_idがtrend_catalogにあります: {missing[:5]}")
    extended = apply_extensions_to_catalog(catalog, cycles, extensions)
    cycles.to_csv(output_dir / "cycles.csv", index=False, encoding="utf-8-sig")
    extensions.to_csv(output_dir / "cycle_extensions.csv", index=False, encoding="utf-8-sig")
    extended.to_csv(output_dir / "trend_catalog.csv", index=False, encoding="utf-8-sig")

    database_path = output_dir / "vector_database.sqlite"
    shutil.copy2(artifact_dir / "vector_database.sqlite", database_path)
    connection = connect(database_path)
    try:
        database_trends = pd.read_sql_query("SELECT trend_id, model_split, guide_available_date FROM trends", connection)
        if not database_trends["model_split"].isin(["model_train", "model_validation"]).all():
            raise ValueError("元のDBにmodel_train/model_validation以外の行があります。")
        rows = extended.merge(database_trends[["trend_id", "guide_available_date"]], on="trend_id", how="inner")
        cycle_info = cycles.set_index("dataset_id")
        rows["extension_lambda"] = rows["dataset_id"].astype(str).map(cycle_info["lambda"])
        rows["extension_seed"] = int(seed)
        has_synthetic = rows["future_synthetic_mask"].map(lambda text: any(_loads(text)))
        data_end = (pd.to_datetime(rows["dataset_id"].astype(str).map(cycle_info["last_measurement_date"]))
                    + pd.Timedelta(days=15)).dt.strftime("%Y-%m-%d")
        later = has_synthetic & (data_end > rows["guide_available_date"])
        rows.loc[later, "guide_available_date"] = data_end[later]
        update_trend_futures(connection, rows)
        connection.commit()
    finally:
        connection.close()

    synthetic_counts = extended["future_synthetic_mask"].map(lambda text: sum(_loads(text)))
    split_of_row = extended["dataset_id"].astype(str).map(split_of_dataset)
    summary = {
        "artifact_dir": str(artifact_dir), "output_dir": str(output_dir), "trend_dir": str(trend_dir),
        "seed": int(seed), "horizon_months": int(horizon_months), "min_span_months": int(min_span_months),
        "cycles": {split: {"cycles": int(len(group)), "extended": int(group["extended"].sum()),
                           "without_maintenance_date": None}
                   for split, group in cycles.groupby("split")},
        "extension_months_by_kind": {split: group["kind"].value_counts().to_dict()
                                     for split, group in extensions.groupby("split")},
        "catalog_synthetic": {split: {"rows": int((split_of_row == split).sum()),
                                      "rows_with_synthetic": int(((synthetic_counts > 0) & (split_of_row == split)).sum()),
                                      "synthetic_months": int(synthetic_counts[split_of_row == split].sum())}
                              for split in sorted(split_of_row.dropna().unique())},
        "lambda_quantiles": {str(q): float(cycles["lambda"].quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "database": {"rows_updated": int(len(rows)), "rows_with_synthetic": int(has_synthetic.sum()),
                     "guide_available_date_moved_later": int(later.sum())},
    }
    starts_from_maintenance = raw.groupby(raw["dataset_id"].astype(str))["previous_maintenance_date"].first()
    no_date = pd.to_datetime(starts_from_maintenance, errors="coerce").isna()
    for split, group in cycles.groupby("split"):
        summary["cycles"][split]["without_maintenance_date"] = int(no_date.reindex(group["dataset_id"]).fillna(True).sum())
    (output_dir / "extension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    return summary
