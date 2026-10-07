"""1レコードの予測結果(100本のDDIMサンプル・実測値・history・guide・施工マーカー)を
暦日x軸でPNG化する診断ツール。SPEC.mdのパイプライン段階ではなく、
self-check/diagnose-guide-conditioningと同じ「既存成果物の事後検証」カテゴリ
(cli.pyモジュールdocstring参照)。

`acceleration_forecasting_12m/.../evaluation/visualize.py::plot_target`を参考にしたが、
v2は正規化を経由しない(split保存の.npyは物理値のまま、SPEC.md 1.2)ため
ForecastDatasetV2/Normalizationは使わず、evaluation/evaluate.py::per_target_frameと
同じ「metadata.csvのtrend_id→index対応＋np.load直読み」方式を踏襲する。

施工記録は`trend_catalog.csv`の`cutoff_maintenance_date`のみを使う(ユーザー判断により
`施工記録_長野`生データは今回配線しない)。この列は1行ごとではなくdataset_id単位で
一定(1つのdataset_id=1つの施工間隔、`retrieval/trends.py`参照)なので、区間
(direction, bin_start_m, bin_end_m)ごとにdataset_id単位でユニークな値を集めれば、
その区間の施工イベント履歴を復元できる。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN


def reconstruct_segment_maintenance_events(trend_catalog: pd.DataFrame) -> dict:
    """区間(direction, bin_start_m, bin_end_m)ごとに、施工イベント履歴を復元する。

    Returns:
        {(direction, bin_start_m, bin_end_m): [{"date", "maintenance_type",
        "maintenance_description"}, ...]}(日付昇順、重複日付は除去)。
        復元可能な施工イベントが無い区間はキー自体が存在しない。
    """
    working = trend_catalog.copy()
    working["cutoff_maintenance_date"] = pd.to_datetime(working["cutoff_maintenance_date"], errors="coerce")
    per_dataset = working.drop_duplicates(subset="dataset_id", keep="first")

    events = {}
    for key, group in per_dataset.groupby(["direction", "bin_start_m", "bin_end_m"], sort=False):
        rows = group.dropna(subset=["cutoff_maintenance_date"]).sort_values("cutoff_maintenance_date")
        rows = rows.drop_duplicates(subset="cutoff_maintenance_date", keep="first")
        if rows.empty:
            continue
        direction, bin_start_m, bin_end_m = key
        events[(direction, float(bin_start_m), float(bin_end_m))] = [
            {
                "date": row["cutoff_maintenance_date"],
                "maintenance_type": row.get("maintenance_type"),
                "maintenance_description": row.get("maintenance_description"),
            }
            for _, row in rows.iterrows()
        ]
    return events


def months_since_maintenance(anchor_date, events: list[dict]):
    """anchor_date以前で最も新しい施工イベントからの経過月数。該当イベントが無ければNone。"""
    anchor = pd.Timestamp(anchor_date)
    past = [event["date"] for event in events if event["date"] <= anchor]
    if not past:
        return None
    return (anchor - max(past)).days / 30.44


def classify_maintenance_relation(months_since, *, immediately_after_months=2.0, elapsed_months=6.0):
    """経過月数から"immediately_after"/"elapsed"/Noneを返す。"""
    if months_since is None:
        return None
    if months_since <= immediately_after_months:
        return "immediately_after"
    if months_since >= elapsed_months:
        return "elapsed"
    return None


def select_record(metadata: pd.DataFrame, trend_catalog: pd.DataFrame, *, trend_id=None,
                   maintenance_relation="any", immediately_after_months=2.0, elapsed_months=6.0,
                   available_trend_ids=None, seed=42) -> dict:
    """描画対象のtrend_idを選ぶ。

    trend_id指定時はバケット判定を完全に迂回する(available_trend_idsのチェックもしない)。
    未指定時は、区間の施工イベントからの経過月数で"immediately_after"/"elapsed"に
    分類できるレコードのプールを作り、maintenance_relationで絞り込んだ上でseed付き
    乱数選択を行う。
    """
    if trend_id is not None:
        trend_id = str(trend_id)
        if trend_id not in set(metadata["trend_id"].astype(str)):
            raise ValueError(f"trend_id={trend_id!r} がmetadata.csvに存在しません。")
        return {"trend_id": trend_id, "maintenance_relation": None, "months_since_maintenance": None,
                "selection_pool_size": None}

    events_by_segment = reconstruct_segment_maintenance_events(trend_catalog)
    pool = []
    category_counts = {"immediately_after": 0, "elapsed": 0}
    for _, row in metadata.iterrows():
        key = (row["direction"], float(row["bin_start_m"]), float(row["bin_end_m"]))
        months = months_since_maintenance(row["anchor_date"], events_by_segment.get(key, []))
        category = classify_maintenance_relation(
            months, immediately_after_months=immediately_after_months, elapsed_months=elapsed_months,
        )
        if category is None:
            continue
        if available_trend_ids is not None and str(row["trend_id"]) not in available_trend_ids:
            continue
        category_counts[category] += 1
        if maintenance_relation != "any" and category != maintenance_relation:
            continue
        pool.append((str(row["trend_id"]), category, months))

    if not pool:
        raise ValueError(
            f"条件(maintenance_relation={maintenance_relation!r}, "
            f"immediately_after_months={immediately_after_months}, elapsed_months={elapsed_months})に"
            f"合致するレコードがありません(絞り込み後の内訳: {category_counts})。閾値かmaintenance_relationを"
            "調整してください。"
        )
    pool.sort(key=lambda item: item[0])
    rng = np.random.default_rng(seed)
    chosen = pool[int(rng.integers(len(pool)))]
    return {"trend_id": chosen[0], "maintenance_relation": chosen[1], "months_since_maintenance": chosen[2],
            "selection_pool_size": len(pool)}


def compute_fixed_bins(*, bin_width=0.1, y_bounds=(PHYSICAL_MIN, PHYSICAL_MAX)):
    """物理値レンジを`bin_width`(既定0.1)刻みで覆う、共通のヒストグラムビン境界を返す。"""
    low, high = y_bounds
    n_steps = int(round((high - low) / bin_width))
    return low + np.arange(n_steps + 1) * bin_width


def render_generation_histogram(ax, dates, samples, bin_edges, *, color="tab:orange", max_half_width_days=6.0):
    """月ごとに100サンプルのヒストグラムを取り、カウント(=同じ値を生成した頻度)に
    比例した長さの横棒を、その月の実日付を中心に左右対称に描く(「生成頻度」レイヤー)。

    samples: shape (num_samples, n_months)。
    """
    dates = pd.DatetimeIndex(dates)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts_per_month = np.stack(
        [np.histogram(samples[:, month], bins=bin_edges)[0] for month in range(samples.shape[1])]
    )
    max_count = counts_per_month.max()
    first = True
    for month_index, date in enumerate(dates):
        date_num = mdates.date2num(date)
        for bin_index in range(len(bin_edges) - 1):
            count = counts_per_month[month_index, bin_index]
            if count <= 0 or max_count <= 0:
                continue
            half_width = (count / max_count) * max_half_width_days
            ax.barh(
                y=bin_centers[bin_index], width=half_width * 2, left=date_num - half_width,
                height=bin_width * 0.9, color=color, align="center", linewidth=0,
                label="生成頻度(100回生成)" if first else "_nolegend_", zorder=2,
            )
            first = False


def render_quartile_band(ax, dates, samples, *, color="tab:orange"):
    """月ごとのQ1-Q3を、生成頻度とは別レイヤーの帯+破線2本で描く。"""
    dates = pd.DatetimeIndex(dates)
    q1 = np.percentile(samples, 25, axis=0)
    q3 = np.percentile(samples, 75, axis=0)
    ax.fill_between(dates, q1, q3, color=color, alpha=0.15, label="四分位範囲", zorder=1)
    ax.plot(dates, q1, color=color, linestyle=":", linewidth=1.2, label="第1四分位数", zorder=3)
    ax.plot(dates, q3, color=color, linestyle=":", linewidth=1.2, label="第3四分位数", zorder=3)


def plot_prediction_record(output_path, *, trend_id, dataset_id, direction, bin_start_m, bin_end_m,
                            anchor_date, current_value, history_dates, history_values, history_masks,
                            forecast_dates, guide_values, guide_masks, target_values, target_masks,
                            samples, maintenance_events, bin_width=0.1,
                            y_bounds=(PHYSICAL_MIN, PHYSICAL_MAX), dpi=150):
    """1枚のmatplotlib図を組み立てて保存する。"""
    plt.rcParams["font.family"] = "MS Gothic"
    history_dates = pd.DatetimeIndex(history_dates)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    anchor = pd.Timestamp(anchor_date)

    history_values = np.asarray(history_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    guide_values = np.asarray(guide_values, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    history_valid = np.asarray(history_masks) > 0
    ax.plot(history_dates[history_valid], history_values[history_valid], "o-", color="tab:blue",
            label="実績(説明系列)", zorder=4)
    ax.plot([anchor], [current_value], "*", color="tab:blue", markersize=14, zorder=5)

    for rank in range(guide_values.shape[0]):
        guide_valid = np.asarray(guide_masks[rank]) > 0
        if not guide_valid.any():
            continue
        ax.plot(forecast_dates[guide_valid], guide_values[rank][guide_valid], "x--", color="0.5",
                linewidth=1.0, markersize=5, label=f"類似系列{rank + 1}(月オフセット整列)", zorder=2)

    bin_edges = compute_fixed_bins(bin_width=bin_width, y_bounds=y_bounds)
    render_generation_histogram(ax, forecast_dates, samples, bin_edges)
    render_quartile_band(ax, forecast_dates, samples)

    target_valid = np.asarray(target_masks) > 0
    ax.plot(forecast_dates[target_valid], target_values[target_valid], "D-", color="tab:green",
            label="正解値", zorder=4)

    plot_start, plot_end = history_dates.min(), forecast_dates.max()
    first_maintenance = True
    for event in maintenance_events:
        date = event["date"]
        if date < plot_start or date > plot_end:
            continue
        ax.axvline(date, color="0.35", linestyle="--", linewidth=1.0,
                   label="施工(メンテナンス)" if first_maintenance else "_nolegend_", zorder=1)
        first_maintenance = False
        label_parts = [str(value) for value in (event.get("maintenance_type"), event.get("maintenance_description"))
                       if value is not None and str(value) != "nan"]
        if label_parts:
            ax.text(date, y_bounds[1] * 0.98, "\n".join(label_parts), rotation=90, fontsize=7,
                    va="top", ha="right", color="0.3")

    ax.axvline(anchor, color="steelblue", linestyle="-", linewidth=1.0, label="予測起点")
    ax.axvline(forecast_dates[0], color="navy", linestyle=":", linewidth=1.3, label="予測開始")

    ax.set_ylim(*y_bounds)
    ax.set_ylabel("絶対値最大加速度 [m/s²]")
    ax.set_title(f"{direction}方向 {bin_start_m:.0f}-{bin_end_m:.0f}m / 起点 {anchor.date()} / {trend_id}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    fig.autofmt_xdate(rotation=45)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)


def _load_samples_for_trend_id(prediction_dir, trend_id):
    """prediction_dirのsamples.npzから、指定trend_id 1件分のサンプル配列を取り出す。"""
    with np.load(Path(prediction_dir) / "samples.npz", allow_pickle=False) as samples_npz:
        sample_trend_ids = samples_npz["trend_ids"].astype(str)
        positions = np.where(sample_trend_ids == trend_id)[0]
        if len(positions) == 0:
            raise ValueError(f"trend_id={trend_id!r} がprediction_dirのsamples.npzに見つかりません。")
        return samples_npz["samples"][positions[0]]


def _available_trend_ids(prediction_dir):
    with np.load(Path(prediction_dir) / "samples.npz", allow_pickle=False) as samples_npz:
        return set(samples_npz["trend_ids"].astype(str))


def _render_record(metadata, trend_catalog, split_dir, samples, trend_id, output_path, *,
                    bin_width=0.1, dpi=150):
    """1レコード分のnpy読み込み〜PNG書き出しまでを行う、`plot_prediction`/
    `plot_prediction_batch`共通の下請け関数。レコードの基本情報(dict)を返す。
    """
    split_dir = Path(split_dir)
    lookup = {str(value): index for index, value in enumerate(metadata["trend_id"])}
    if trend_id not in lookup:
        raise ValueError(f"trend_id={trend_id!r} がmetadata.csvに存在しません。")
    index = lookup[trend_id]
    row = metadata.iloc[index]

    history_values = np.load(split_dir / "history_values.npy")[index]
    history_masks = np.load(split_dir / "history_masks.npy")[index]
    guide_values = np.load(split_dir / "guide_values.npy")[index]
    guide_masks = np.load(split_dir / "guide_masks.npy")[index]
    target_values = np.load(split_dir / "target_values.npy")[index]
    target_masks = np.load(split_dir / "target_masks.npy")[index]

    catalog_rows = trend_catalog.loc[trend_catalog["trend_id"].astype(str) == trend_id]
    if catalog_rows.empty:
        raise ValueError(f"trend_id={trend_id!r} がtrend_catalog.csvに見つかりません。")
    selected_dates = json.loads(catalog_rows.iloc[0]["selected_dates"])
    forecast_dates = pd.to_datetime(selected_dates)

    anchor = pd.Timestamp(row["anchor_date"])
    anchor_month = anchor.replace(day=1)
    history_dates = [anchor_month - pd.DateOffset(months=5 - i) for i in range(5)] + [anchor]

    direction, bin_start_m, bin_end_m = row["direction"], float(row["bin_start_m"]), float(row["bin_end_m"])
    events_by_segment = reconstruct_segment_maintenance_events(trend_catalog)
    maintenance_events = events_by_segment.get((direction, bin_start_m, bin_end_m), [])

    plot_prediction_record(
        output_path, trend_id=trend_id, dataset_id=row["dataset_id"], direction=direction,
        bin_start_m=bin_start_m, bin_end_m=bin_end_m, anchor_date=anchor,
        current_value=float(row["current_acc_z_max"]), history_dates=history_dates,
        history_values=history_values, history_masks=history_masks, forecast_dates=forecast_dates,
        guide_values=guide_values, guide_masks=guide_masks, target_values=target_values,
        target_masks=target_masks, samples=samples, maintenance_events=maintenance_events,
        bin_width=bin_width, dpi=dpi,
    )
    months_since = months_since_maintenance(anchor, maintenance_events)
    return {
        "trend_id": trend_id, "dataset_id": str(row["dataset_id"]), "direction": direction,
        "bin_start_m": bin_start_m, "bin_end_m": bin_end_m, "anchor_date": str(anchor.date()),
        "months_since_maintenance": months_since, "num_samples": int(samples.shape[0]),
    }


def plot_prediction(dataset_dir, prediction_dir, artifact_dir, output_dir, *, split="inference",
                     trend_id=None, maintenance_relation="any", immediately_after_months=2.0,
                     elapsed_months=6.0, seed=42, bin_width=0.1, dpi=150) -> dict:
    """CLIから呼ばれるオーケストレーション関数。1レコードを選んでPNGを1枚書き出す。"""
    dataset_dir, prediction_dir, artifact_dir, output_dir = (
        Path(dataset_dir), Path(prediction_dir), Path(artifact_dir), Path(output_dir)
    )
    split_dir = dataset_dir / split
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    trend_catalog = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")

    selection = select_record(
        metadata, trend_catalog, trend_id=trend_id, maintenance_relation=maintenance_relation,
        immediately_after_months=immediately_after_months, elapsed_months=elapsed_months,
        available_trend_ids=_available_trend_ids(prediction_dir), seed=seed,
    )
    chosen_trend_id = selection["trend_id"]
    samples = _load_samples_for_trend_id(prediction_dir, chosen_trend_id)

    row = metadata.loc[metadata["trend_id"].astype(str) == chosen_trend_id].iloc[0]
    direction, bin_start_m, bin_end_m = row["direction"], float(row["bin_start_m"]), float(row["bin_end_m"])
    anchor = pd.Timestamp(row["anchor_date"])
    filename = f"{direction}_{bin_start_m:.0f}-{bin_end_m:.0f}m_{anchor.date()}_{chosen_trend_id}.png"
    output_path = output_dir / filename

    info = _render_record(metadata, trend_catalog, split_dir, samples, chosen_trend_id, output_path,
                          bin_width=bin_width, dpi=dpi)

    maintenance_relation_for_summary = selection["maintenance_relation"] or classify_maintenance_relation(
        info["months_since_maintenance"], immediately_after_months=immediately_after_months,
        elapsed_months=elapsed_months,
    )
    return {
        **info, "output_path": str(output_path), "maintenance_relation": maintenance_relation_for_summary,
        "selection_pool_size": selection["selection_pool_size"],
    }


def select_records(metadata: pd.DataFrame, trend_catalog: pd.DataFrame, *, count_per_category=3,
                    immediately_after_months=2.0, elapsed_months=6.0, available_trend_ids=None,
                    seed=42) -> dict:
    """「immediately_after」「elapsed」の各カテゴリから、重複なくcount_per_category件まで
    ランダムに選ぶ(`select_record`の複数選択版)。カテゴリのプールがcount_per_category未満
    の場合は、そのカテゴリの全件を使う(エラーにはしない、実際に選んだ件数を結果に含める)。

    Returns:
        {"immediately_after": [trend_id, ...], "elapsed": [trend_id, ...],
        "pool_sizes": {"immediately_after": int, "elapsed": int}}
    """
    events_by_segment = reconstruct_segment_maintenance_events(trend_catalog)
    pools = {"immediately_after": [], "elapsed": []}
    for _, row in metadata.iterrows():
        key = (row["direction"], float(row["bin_start_m"]), float(row["bin_end_m"]))
        months = months_since_maintenance(row["anchor_date"], events_by_segment.get(key, []))
        category = classify_maintenance_relation(
            months, immediately_after_months=immediately_after_months, elapsed_months=elapsed_months,
        )
        if category is None:
            continue
        trend_id = str(row["trend_id"])
        if available_trend_ids is not None and trend_id not in available_trend_ids:
            continue
        pools[category].append(trend_id)

    rng = np.random.default_rng(seed)
    chosen = {}
    for category, pool in pools.items():
        pool = sorted(pool)
        count = min(count_per_category, len(pool))
        chosen[category] = list(rng.choice(pool, size=count, replace=False)) if count else []
    return {**chosen, "pool_sizes": {category: len(pool) for category, pool in pools.items()}}


def plot_prediction_batch(dataset_dir, artifact_dir, output_dir, methods: dict, *, split="inference",
                          count_per_category=3, immediately_after_months=2.0, elapsed_months=6.0,
                          seed=42, bin_width=0.1, dpi=150) -> dict:
    """複数レコード×複数手法(手法名→prediction_dirの対応表)を一括生成する。

    出力は`output_dir/<trend_id>/<手法名>.png`(レコードごとにフォルダ、同一レコードの
    手法間比較がしやすい構成、2026-10-07にユーザーと決定)。レコードは区間の施工記録から
    「直後」「経過」それぞれcount_per_category件ずつランダム選択し、全手法の
    prediction_dirに共通して存在するtrend_idだけを対象にする(どれか1つでも欠けている
    レコードは選ばない)。
    """
    dataset_dir, artifact_dir, output_dir = Path(dataset_dir), Path(artifact_dir), Path(output_dir)
    split_dir = dataset_dir / split
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    trend_catalog = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")

    available_per_method = {name: _available_trend_ids(path) for name, path in methods.items()}
    common_available = set.intersection(*available_per_method.values()) if available_per_method else set()

    selection = select_records(
        metadata, trend_catalog, count_per_category=count_per_category,
        immediately_after_months=immediately_after_months, elapsed_months=elapsed_months,
        available_trend_ids=common_available, seed=seed,
    )
    selected_trend_ids = [
        (trend_id, category) for category in ("immediately_after", "elapsed") for trend_id in selection[category]
    ]

    records = []
    for trend_id, category in selected_trend_ids:
        record_dir = output_dir / trend_id
        outputs = {}
        for method_name, prediction_dir in methods.items():
            samples = _load_samples_for_trend_id(prediction_dir, trend_id)
            output_path = record_dir / f"{method_name}.png"
            info = _render_record(metadata, trend_catalog, split_dir, samples, trend_id, output_path,
                                  bin_width=bin_width, dpi=dpi)
            outputs[method_name] = str(output_path)
        records.append({"trend_id": trend_id, "maintenance_relation": category,
                        "months_since_maintenance": info["months_since_maintenance"], "outputs": outputs})

    return {
        "split": split, "methods": list(methods), "record_count": len(records),
        "pool_sizes": selection["pool_sizes"], "records": records,
    }
