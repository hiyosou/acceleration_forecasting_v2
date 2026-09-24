"""6か月統一履歴(今月+過去5か月)の構築。

旧 `acceleration_forecasting_12m/.../datasets/history.py` の `select_monthly_history`
(過去5か月のみ)を出発点に、SPEC.md 1.1「入力形状: 6か月履歴に統一」の決定に従い、
anchor自身の値(今月)を6点目として同じ配列・同じエンコード経路に統合した。
情報量は変えていない(旧実装の current_acc_z_max 相当を、履歴配列の末尾に
合流させただけ)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_history_window(section_history: pd.DataFrame, anchor_date, *, months: int = 6, search_days: int = 15):
    """今月を含む直近`months`か月分の実測値を月単位で選び出す。

    Args:
        section_history: 対象dataset_id(施工間セグメント)の全行。
            少なくとも `measurement_date`(datetime64) と `acc_z_max`(float) 列を持つこと。
            `retrieval.trends.TrendCatalog._groups[dataset_id]` と同じ形を想定している。
        anchor_date: この履歴窓の基準日(=anchorの計測日)。
        months: 履歴の長さ。SPEC.md 1.1の決定によりデフォルト6
            (indexの並びは「最も古い月 → 今月」の昇順、最後の1点=index[months-1]が今月)。
        search_days: 対象月の月央からのずれ許容日数(過去5か月分の探索にのみ適用。
            今月分はanchor自身の実測値をそのまま使うため探索しない)。

    Returns:
        (values, mask, dates): shape (months,) のfloat32配列2つと、採用日付の文字列リスト
        (空文字は欠測)。mask=0の位置のvaluesは必ずNaN(保存時センチネル。SPEC.md 1.2と
        同じ規約 — モデル入力直前でnan_to_num(0)を適用するまでNaNを保持する)。
    """
    if months < 1:
        raise ValueError("months must be at least 1")
    anchor = pd.Timestamp(anchor_date).normalize()
    anchor_month = anchor.replace(day=1)
    past_months = months - 1

    available = section_history.loc[section_history["measurement_date"] < anchor].copy()
    used, values, masks, dates = set(), [], [], []
    for offset in range(past_months, 0, -1):
        target = anchor_month - pd.DateOffset(months=offset)
        candidates = available.loc[
            (available["measurement_date"] - target).dt.days.abs() <= int(search_days)
        ].copy()
        candidates = candidates.loc[~candidates["measurement_date"].isin(used)]
        if candidates.empty:
            values.append(np.nan)
            masks.append(0.0)
            dates.append("")
            continue
        candidates["_distance"] = (candidates["measurement_date"] - target).dt.days.abs()
        candidates["_future_tie"] = (candidates["measurement_date"] > target).astype(int)
        row = candidates.sort_values(["_distance", "_future_tie", "measurement_date"], kind="mergesort").iloc[0]
        date = pd.Timestamp(row["measurement_date"]).normalize()
        value = float(row["acc_z_max"])
        used.add(date)
        values.append(value if np.isfinite(value) else np.nan)
        masks.append(1.0 if np.isfinite(value) else 0.0)
        dates.append(date.strftime("%Y-%m-%d") if np.isfinite(value) else "")

    # 今月(index = months-1): 過去5か月と違い探索は行わず、anchor自身の実測値を採用する。
    current_rows = section_history.loc[section_history["measurement_date"] == anchor]
    if current_rows.empty:
        raise ValueError("section_history に anchor_date と一致する行がありません。")
    current_value = float(current_rows.iloc[0]["acc_z_max"])
    if np.isfinite(current_value):
        values.append(current_value)
        masks.append(1.0)
        dates.append(anchor.strftime("%Y-%m-%d"))
    else:
        values.append(np.nan)
        masks.append(0.0)
        dates.append("")

    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
        dates,
    )
