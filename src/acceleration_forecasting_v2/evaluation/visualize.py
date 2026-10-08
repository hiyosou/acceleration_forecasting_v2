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

2026-10-07追加: 旧リポジトリの`plot_target`にあった「区間の全実測値(走行速度で色分け)」
散布図を再現する。速度レンジでの絞り込みは行わない(旧リポジトリのevaluate.py側の
呼び出し時フィルタであり、plot_target自体の挙動ではないため、ここでは区間の全実測値を
そのまま見せる)。

2026-10-08改訂: 当初`artifacts/retrieval/vector_database.sqlite`の`trends`/
`waveform_records`テーブルから全実測値・走行速度を読んでいたが、どちらも
`model_split`が`model_train`/`model_validation`のレコードしか含まない(リーク防止の
ため、検索guide用の埋め込みDBに推論対象自身のデータを入れていない)。そのため描画対象
レコード自身の入力・出力期間(inference split)が丸ごと空白になる不具合があった。
データソースを、train/validation/inference全splitを含む`trend_catalog.csv`
(current_acc_z_max)と`split_manifest.csv`(mean_velocity_kmh、`import-snapshot`で
生成される元manifest)に変更した(どちらも全split分を含み、速度が本来欠損する理由は
無い——ユーザー指摘、PROGRESS.md参照)。
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


def load_segment_measurement_history(trend_catalog: pd.DataFrame, direction, bin_start_m, *,
                                      velocity_manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """区間(direction, bin_start_m)の全実測値を`trend_catalog`から読み込む。

    2026-10-08: データソースを`vector_database.sqlite`の`trends`テーブルから`trend_catalog`に
    変更した。`trends`テーブルは`model_split`が`model_train`/`model_validation`のレコードしか
    含まない(リーク防止のため、検索guide用の埋め込みDBに推論対象自身のデータを入れていない)。
    そのため、描画対象レコードの入力期間・出力期間自体がinference splitに属する場合、
    まさにその期間だけ全実測値が丸ごと空白になる不具合があった(実データ`U方向2600-2700m`区間で
    確認: `trends`テーブルは2024-04-15までしか無く、入力(2024-06〜)・出力(2025-02〜)期間が
    完全に欠落していた)。`trend_catalog`はtrain/validation/inference全splitの測定値を含むため、
    この欠落が起きない。

    2026-10-08追記: 走行速度(velocity)も同様の理由で`vector_database.sqlite`の
    `waveform_records`テーブル(`trends`と同じくmodel_train/model_validationのみ)から
    結合していたが、ユーザー指摘により`artifact_dir/split_manifest.csv`(`import-snapshot`で
    生成される、train/validation/inference全split・全420,686行分の`mean_velocity_kmh`を含む
    元manifest)に変更した。速度自体はモデル分割と無関係にCSVから読み取れる値のため、本来
    欠損する理由が無い。

    velocity_manifest: 指定時は`split_manifest.csv`を読み込んだDataFrame
    (`trend_id`・`direction`・`bin_start_m`・`mean_velocity_kmh`列を持つこと)を渡す。
    省略時は速度列はすべてNaN(速度欠損として描画)になる。

    Returns:
        列`measurement_date`(datetime64)・`current_acc_z_max`(float)・`velocity`
        (float、走行速度[km/h]の平均。該当するレコードが無い場合はNaN)を持つDataFrame。
    """
    segment = trend_catalog.loc[
        (trend_catalog["direction"] == direction)
        & (trend_catalog["bin_start_m"].astype(float) == float(bin_start_m))
    ][["trend_id", "measurement_date", "current_acc_z_max"]].copy()
    segment["measurement_date"] = pd.to_datetime(segment["measurement_date"], errors="coerce")

    velocity = pd.DataFrame(columns=["trend_id", "velocity"])
    if velocity_manifest is not None:
        subset = velocity_manifest.loc[
            (velocity_manifest["direction"] == direction)
            & (velocity_manifest["bin_start_m"].astype(float) == float(bin_start_m))
        ]
        velocity = (
            subset.groupby("trend_id", as_index=False)["mean_velocity_kmh"]
            .mean()
            .rename(columns={"mean_velocity_kmh": "velocity"})
        )
    return segment.merge(velocity, on="trend_id", how="left")


def render_measurement_history(ax, measurement_history: pd.DataFrame, *, vmin=0.0, vmax=100.0):
    """区間の全実測値を、走行速度で色分けした散布図として背景に描画する(旧リポジトリの
    `plot_target`の「全実測値」要素を再現)。速度が不明な点はグレーで描く。
    """
    has_velocity = measurement_history["velocity"].notna()
    scatter = None
    if has_velocity.any():
        subset = measurement_history.loc[has_velocity]
        scatter = ax.scatter(
            subset["measurement_date"], subset["current_acc_z_max"], c=subset["velocity"],
            cmap="jet", vmin=vmin, vmax=vmax, s=24, alpha=0.7, label="全実測値", zorder=0,
        )
    if (~has_velocity).any():
        subset = measurement_history.loc[~has_velocity]
        ax.scatter(subset["measurement_date"], subset["current_acc_z_max"], color="0.55", s=24,
                  alpha=0.6, label="全実測値(速度欠損)", zorder=0)
    return scatter


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
                            y_bounds=(PHYSICAL_MIN, PHYSICAL_MAX), dpi=150, measurement_history=None):
    """1枚のmatplotlib図を組み立てて保存する。

    measurement_history: `load_segment_measurement_history`が返す形のDataFrame(省略時は
    描画しない)。区間の全実測値(走行速度で色分け)を背景に重ねる。
    """
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

    if measurement_history is not None and not measurement_history.empty:
        scatter = render_measurement_history(ax, measurement_history)
        if scatter is not None:
            colorbar = fig.colorbar(scatter, ax=ax, pad=0.01)
            colorbar.set_label("走行速度 [km/h]")

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


def compute_target_ranges(split_dir, metadata: pd.DataFrame) -> dict:
    """正解値(target_values)12か月分の「幅」(有効な月のみでの最大値-最小値)を、
    trend_id文字列をキーにしたdictで返す(有効な月が1つも無いレコードはキーに含めない)。

    plot_prediction/plot_prediction_batchの`min_target_range`フィルタ(2026-10-08追加、
    「予測期間中に加速度が大きく変化するレコードだけ見たい」というユーザー要望)の下請け。
    """
    split_dir = Path(split_dir)
    target_values = np.load(split_dir / "target_values.npy")
    target_masks = np.load(split_dir / "target_masks.npy")
    ranges = {}
    for index, trend_id in enumerate(metadata["trend_id"]):
        valid = np.asarray(target_masks[index]) > 0
        if not valid.any():
            continue
        values = np.asarray(target_values[index])[valid]
        ranges[str(trend_id)] = float(values.max() - values.min())
    return ranges


def _load_velocity_manifest(artifact_dir):
    """`artifact_dir/split_manifest.csv`(train/validation/inference全split分の
    `mean_velocity_kmh`を含む元manifest、`import-snapshot`で生成される)を読み込む。
    無ければNoneを返す(全実測値散布図は速度欠損として描画される)。
    """
    manifest_path = Path(artifact_dir) / "split_manifest.csv"
    if not manifest_path.is_file():
        return None
    return pd.read_csv(manifest_path, encoding="utf-8-sig",
                       usecols=["trend_id", "direction", "bin_start_m", "mean_velocity_kmh"])


def _render_record(metadata, trend_catalog, split_dir, samples, trend_id, output_path, *,
                    bin_width=0.1, dpi=150, velocity_manifest=None):
    """1レコード分のnpy読み込み〜PNG書き出しまでを行う、`plot_prediction`/
    `plot_prediction_batch`共通の下請け関数。レコードの基本情報(dict)を返す。

    区間の全実測値(`load_segment_measurement_history`)は`trend_catalog`(train/validation/
    inference全split)から常に読み込む。velocity_manifest指定時はさらに`split_manifest.csv`
    から走行速度を結合する(省略時は速度欠損として描画)。読み込みに失敗しても(trend_catalogに
    該当区間が無い等)プロット自体は失敗させない(全実測値は無しとして続行し、理由を標準エラーに
    出す)。
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

    measurement_history = None
    try:
        measurement_history = load_segment_measurement_history(
            trend_catalog, direction, bin_start_m, velocity_manifest=velocity_manifest,
        )
    except Exception as error:  # split_manifestが無い・壊れている等でも全実測値抜きで続行する
        import sys
        print(f"警告: 全実測値の読み込みに失敗しました({error!r})。この要素なしで描画します。",
             file=sys.stderr)

    plot_prediction_record(
        output_path, trend_id=trend_id, dataset_id=row["dataset_id"], direction=direction,
        bin_start_m=bin_start_m, bin_end_m=bin_end_m, anchor_date=anchor,
        current_value=float(row["current_acc_z_max"]), history_dates=history_dates,
        history_values=history_values, history_masks=history_masks, forecast_dates=forecast_dates,
        guide_values=guide_values, guide_masks=guide_masks, target_values=target_values,
        target_masks=target_masks, samples=samples, maintenance_events=maintenance_events,
        bin_width=bin_width, dpi=dpi, measurement_history=measurement_history,
    )
    months_since = months_since_maintenance(anchor, maintenance_events)
    return {
        "trend_id": trend_id, "dataset_id": str(row["dataset_id"]), "direction": direction,
        "bin_start_m": bin_start_m, "bin_end_m": bin_end_m, "anchor_date": str(anchor.date()),
        "months_since_maintenance": months_since, "num_samples": int(samples.shape[0]),
    }


def plot_prediction(dataset_dir, prediction_dir, artifact_dir, output_dir, *, split="inference",
                     trend_id=None, maintenance_relation="any", immediately_after_months=2.0,
                     elapsed_months=6.0, seed=42, bin_width=0.1, dpi=150, min_target_range=None) -> dict:
    """CLIから呼ばれるオーケストレーション関数。1レコードを選んでPNGを1枚書き出す。

    min_target_range: 指定時は、予測期間(target_values12か月分、有効な月のみ)の
    最大値-最小値がこの値以上のレコードだけを自動選択の対象にする(trend_id明示指定時は
    select_record自体が選別を迂回するため、この絞り込みも適用されない)。
    """
    dataset_dir, prediction_dir, artifact_dir, output_dir = (
        Path(dataset_dir), Path(prediction_dir), Path(artifact_dir), Path(output_dir)
    )
    split_dir = dataset_dir / split
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    trend_catalog = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")

    available_trend_ids = _available_trend_ids(prediction_dir)
    if min_target_range is not None:
        ranges = compute_target_ranges(split_dir, metadata)
        qualifying = {tid for tid, value in ranges.items() if value >= min_target_range}
        available_trend_ids = available_trend_ids & qualifying

    selection = select_record(
        metadata, trend_catalog, trend_id=trend_id, maintenance_relation=maintenance_relation,
        immediately_after_months=immediately_after_months, elapsed_months=elapsed_months,
        available_trend_ids=available_trend_ids, seed=seed,
    )
    chosen_trend_id = selection["trend_id"]
    samples = _load_samples_for_trend_id(prediction_dir, chosen_trend_id)

    row = metadata.loc[metadata["trend_id"].astype(str) == chosen_trend_id].iloc[0]
    direction, bin_start_m, bin_end_m = row["direction"], float(row["bin_start_m"]), float(row["bin_end_m"])
    anchor = pd.Timestamp(row["anchor_date"])
    filename = f"{direction}_{bin_start_m:.0f}-{bin_end_m:.0f}m_{anchor.date()}_{chosen_trend_id}.png"
    output_path = output_dir / filename

    info = _render_record(metadata, trend_catalog, split_dir, samples, chosen_trend_id, output_path,
                          bin_width=bin_width, dpi=dpi,
                          velocity_manifest=_load_velocity_manifest(artifact_dir))

    maintenance_relation_for_summary = selection["maintenance_relation"] or classify_maintenance_relation(
        info["months_since_maintenance"], immediately_after_months=immediately_after_months,
        elapsed_months=elapsed_months,
    )
    return {
        **info, "output_path": str(output_path), "maintenance_relation": maintenance_relation_for_summary,
        "selection_pool_size": selection["selection_pool_size"],
    }


def select_records(metadata: pd.DataFrame, trend_catalog: pd.DataFrame, *, count_per_category=3,
                    maintenance_relation="any", immediately_after_months=2.0, elapsed_months=6.0,
                    available_trend_ids=None, seed=42) -> dict:
    """「immediately_after」「elapsed」の各カテゴリから、重複なくcount_per_category件まで
    ランダムに選ぶ(`select_record`の複数選択版)。カテゴリのプールがcount_per_category未満
    の場合は、そのカテゴリの全件を使う(エラーにはしない、実際に選んだ件数を結果に含める)。

    maintenance_relation: "any"(既定)/"immediately_after"/"elapsed"。"any"以外を指定すると、
    指定カテゴリ以外は最初からプールに入れない(2026-10-08追加・PROGRESS.md参照)。
    「施工直後(immediately_after)」のanchorは、6か月の入力(history)のうち施工より前の
    月がそのdataset_id(施工間隔)に存在しないため構造的に半分(3/6か月)しか有効でない
    (`datasets/anchors.py`参照)。入力の完全性が重要な分析では"elapsed"を指定し、
    このカテゴリを除外する運用とする。

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
        if maintenance_relation != "any" and category != maintenance_relation:
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
                          count_per_category=3, maintenance_relation="any", immediately_after_months=2.0,
                          elapsed_months=6.0, seed=42, bin_width=0.1, dpi=150, min_target_range=None) -> dict:
    """複数レコード×複数手法(手法名→prediction_dirの対応表)を一括生成する。

    出力は`output_dir/<区間_起点日>/<手法名>.png`(レコードごとにフォルダ、同一レコードの
    手法間比較がしやすい構成)。フォルダ名は`{direction}_{bin_start_m:.0f}-{bin_end_m:.0f}m_
    {anchor_date}`(区間の位置+予測起点日、2026-10-07にユーザーと決定・trend_idは含めない)。
    同一区間・同一起点日で別dataset_id(=別の施工間隔)のレコードが存在した場合はフォルダ名が
    衝突するため、その場合は例外を出す(現行データでの実際の衝突は未確認)。各フォルダが
    どのtrend_id/dataset_idに対応するかは`output_dir/manifest.csv`に記録する(2026-10-07
    決定)。

    レコードは区間の施工記録から「直後」「経過」それぞれcount_per_category件ずつランダム
    選択し、全手法のprediction_dirに共通して存在するtrend_idだけを対象にする(どれか1つでも
    欠けているレコードは選ばない)。

    maintenance_relation: "any"(既定)/"immediately_after"/"elapsed"。"elapsed"を指定すると
    「施工直後」カテゴリを選択対象から外す(2026-10-08追加・PROGRESS.md参照。「施工直後」
    anchorは入力(history)が構造的に半分(3/6か月)しか有効でないため、入力の完全性が
    重要な分析ではこちらを使う)。

    min_target_range: 指定時は、予測期間(target_values12か月分、有効な月のみ)の
    最大値-最小値がこの値以上のレコードだけを選択対象プールに残す(2026-10-08追加、
    「予測期間中に加速度が大きく変化するレコードだけ見たい」というユーザー要望)。
    """
    dataset_dir, artifact_dir, output_dir = Path(dataset_dir), Path(artifact_dir), Path(output_dir)
    split_dir = dataset_dir / split
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    trend_catalog = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")
    metadata_lookup = {str(value): index for index, value in enumerate(metadata["trend_id"])}

    available_per_method = {name: _available_trend_ids(path) for name, path in methods.items()}
    common_available = set.intersection(*available_per_method.values()) if available_per_method else set()
    if min_target_range is not None:
        ranges = compute_target_ranges(split_dir, metadata)
        qualifying = {tid for tid, value in ranges.items() if value >= min_target_range}
        common_available = common_available & qualifying

    selection = select_records(
        metadata, trend_catalog, count_per_category=count_per_category,
        maintenance_relation=maintenance_relation, immediately_after_months=immediately_after_months,
        elapsed_months=elapsed_months, available_trend_ids=common_available, seed=seed,
    )
    selected_trend_ids = [
        (trend_id, category) for category in ("immediately_after", "elapsed") for trend_id in selection[category]
    ]

    velocity_manifest = _load_velocity_manifest(artifact_dir)

    records = []
    manifest_rows = []
    folder_names_seen = {}
    for trend_id, category in selected_trend_ids:
        row = metadata.iloc[metadata_lookup[trend_id]]
        direction = row["direction"]
        bin_start_m, bin_end_m = float(row["bin_start_m"]), float(row["bin_end_m"])
        anchor_date = pd.Timestamp(row["anchor_date"]).date()
        folder_name = f"{direction}_{bin_start_m:.0f}-{bin_end_m:.0f}m_{anchor_date}"
        if folder_name in folder_names_seen:
            raise ValueError(
                f"フォルダ名{folder_name!r}がtrend_id={trend_id!r}と"
                f"trend_id={folder_names_seen[folder_name]!r}の間で衝突しました"
                "(同一区間・同一起点日で別dataset_idのレコードが存在します)。"
            )
        folder_names_seen[folder_name] = trend_id

        record_dir = output_dir / folder_name
        outputs = {}
        for method_name, prediction_dir in methods.items():
            samples = _load_samples_for_trend_id(prediction_dir, trend_id)
            output_path = record_dir / f"{method_name}.png"
            info = _render_record(metadata, trend_catalog, split_dir, samples, trend_id, output_path,
                                  bin_width=bin_width, dpi=dpi, velocity_manifest=velocity_manifest)
            outputs[method_name] = str(output_path)
        records.append({"trend_id": trend_id, "folder": folder_name, "maintenance_relation": category,
                        "months_since_maintenance": info["months_since_maintenance"], "outputs": outputs})
        manifest_rows.append({
            "folder": folder_name, "trend_id": trend_id, "dataset_id": str(row["dataset_id"]),
            "direction": direction, "bin_start_m": bin_start_m, "bin_end_m": bin_end_m,
            "anchor_date": str(anchor_date), "maintenance_relation": category,
            "months_since_maintenance": info["months_since_maintenance"],
            **{f"output_{name}": path for name, path in outputs.items()},
        })

    manifest_path = None
    if manifest_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")

    return {
        "split": split, "methods": list(methods), "record_count": len(records),
        "pool_sizes": selection["pool_sizes"], "records": records,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
    }
