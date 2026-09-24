"""anchor(予測起点)選定。SPEC.md 決定「セグメント内、min_target_monthsを満たす
月をすべて拾う」を実装する。旧実装(`acceleration_forecasting_12m/.../datasets/build.py`
の`_select_anchors`)は1セグメントにつきベスト1件のみを選んでいたが、本実装は
条件を満たす候補を全て集める点が最大の違い。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .history import build_history_window


@dataclass(frozen=True)
class AnchorCandidate:
    """1つの(dataset_id, measurement_date, bin_start_m, bin_end_m)候補。"""

    trend_id: str
    dataset_id: str
    measurement_date: pd.Timestamp
    bin_start_m: float
    bin_end_m: float
    direction: str
    current_acc_z_max: float
    history_values: np.ndarray
    history_mask: np.ndarray
    future_values: np.ndarray
    future_mask: np.ndarray
    valid_history_months: int
    valid_future_months: int


def _parse_future(row, forecast_months):
    values = json.loads(row["future_values"])
    mask = json.loads(row["future_mask"])
    if len(values) != forecast_months or len(mask) != forecast_months:
        raise ValueError(
            f"trend_id={row['trend_id']} の future_values/future_mask の長さが "
            f"forecast_months={forecast_months} と一致しません。"
        )
    values = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    return values, mask


def select_anchors_for_segment(
    segment_rows: pd.DataFrame,
    *,
    split: Literal["model_train", "model_validation", "inference"],
    min_history_months: int,
    min_target_months: int,
    history_months: int = 6,
    forecast_months: int = 12,
) -> list[AnchorCandidate]:
    """1つのdataset_id(施工間セグメント)内で、適格性条件を満たす月を全て拾う。

    Args:
        segment_rows: 単一dataset_idのtrend_catalog行(measurement_date昇順である
            必要はない、本関数内でソートする)。少なくとも
            trend_id/dataset_id/measurement_date/direction/bin_start_m/bin_end_m/
            current_acc_z_max/future_values/future_mask 列を持つこと。
        split: "inference"なら未来値を参照しない`production_ready`基準で判定し、
            それ以外(model_train/model_validation)は未来値も含めた`offline_valid`
            基準で判定する(SPEC.md 2.3 チェック項目7)。
        min_history_months: 6か月中、有効な月が何か月以上必要か。
        min_target_months: 12か月中、有効な月が何か月以上必要か(inferenceでは未使用)。

    Returns:
        measurement_date昇順にソートされたAnchorCandidateのリスト。
        1件も適格な候補がなければ空リストを返す(例外は投げない)。
    """
    if segment_rows.empty:
        return []
    ordered = segment_rows.sort_values("measurement_date", kind="mergesort")
    section_history = ordered.rename(columns={"current_acc_z_max": "acc_z_max"})[["measurement_date", "acc_z_max"]].copy()
    section_history["measurement_date"] = pd.to_datetime(section_history["measurement_date"])

    selected: list[AnchorCandidate] = []
    for _, row in ordered.iterrows():
        anchor_date = pd.Timestamp(row["measurement_date"])
        current = float(row["current_acc_z_max"])
        current_valid = bool(np.isfinite(current))

        history_values, history_mask, _ = build_history_window(
            section_history, anchor_date, months=history_months
        )
        valid_history_months = int(np.count_nonzero(history_mask > 0))

        future_values, future_mask = _parse_future(row, forecast_months)
        valid_future_months = int(np.count_nonzero((future_mask > 0) & np.isfinite(future_values)))

        production_ready = current_valid and valid_history_months >= min_history_months
        offline_valid = production_ready and valid_future_months >= min_target_months
        accepted = production_ready if split == "inference" else offline_valid
        if not accepted:
            continue

        selected.append(
            AnchorCandidate(
                trend_id=str(row["trend_id"]),
                dataset_id=str(row["dataset_id"]),
                measurement_date=anchor_date,
                bin_start_m=float(row["bin_start_m"]),
                bin_end_m=float(row["bin_end_m"]),
                direction=str(row["direction"]),
                current_acc_z_max=current,
                history_values=history_values,
                history_mask=history_mask,
                future_values=future_values,
                future_mask=future_mask,
                valid_history_months=valid_history_months,
                valid_future_months=valid_future_months,
            )
        )
    return selected


def compute_segment_weights(anchors) -> dict[str, float]:
    """dataset_idごとに 1/N_s (N_s=そのdataset_idのanchor数) を計算する(SPEC.md 1.1)。

    Args:
        anchors: 単一split内の全AnchorCandidate(複数dataset_id混在可)。
            他splitのanchorを混ぜてはいけない(重みはsplit内で完結させる)。
    """
    counts: dict[str, int] = {}
    for anchor in anchors:
        counts[anchor.dataset_id] = counts.get(anchor.dataset_id, 0) + 1
    return {dataset_id: 1.0 / count for dataset_id, count in counts.items()}
