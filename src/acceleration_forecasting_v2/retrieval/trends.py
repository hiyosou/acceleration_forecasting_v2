"""施工間セグメントの月次トレンド(anchor日付ごとの未来値配列)を構築する。

`acceleration_retrieval/trends.py` を出発点に移植したもの(SPEC.mdの決定
「retrievalコードはコピーして出発点にする」に対応)。ロジック自体は変更していない
(FUTURE_MONTHS を12か月に短縮した点のみ retrieval/constants.py 側の差分)。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import FUTURE_MONTHS, SEARCH_DAYS


REQUIRED_TREND_COLUMNS = {
    "dataset_id",
    "direction",
    "bin_start_m",
    "bin_end_m",
    "segment_index",
    "measurement_date",
    "acc_z_max",
}


def _read_csv_fallback(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # noqa: BLE001 - フォールバック探索のため捕捉
            last_error = exc
    raise last_error


def _json_values(values):
    return json.dumps(
        [None if pd.isna(value) else value for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class TrendKey:
    measurement_date: str
    direction: str
    bin_start_m: float
    bin_end_m: float


class TrendCatalog:
    """施工間セグメント(dataset_id)ごとの月次実測値をanchor日付から検索する。"""

    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.copy()
        if self.rows.empty:
            self._key_to_row = {}
            self._groups = {}
            self._cutoffs = {}
            return

        self.rows["measurement_date"] = pd.to_datetime(
            self.rows["measurement_date"], errors="coerce"
        ).dt.normalize()
        self.rows["direction"] = self.rows["direction"].astype(str).str.upper()
        self.rows["bin_start_m"] = pd.to_numeric(self.rows["bin_start_m"], errors="coerce")
        self.rows["bin_end_m"] = pd.to_numeric(self.rows["bin_end_m"], errors="coerce")
        self.rows["acc_z_max"] = pd.to_numeric(self.rows["acc_z_max"], errors="coerce")
        self.rows = self.rows.dropna(
            subset=["measurement_date", "direction", "bin_start_m", "bin_end_m", "acc_z_max"]
        )
        self.rows = self.rows.sort_values(
            ["dataset_id", "measurement_date", "measured_at"], kind="mergesort"
        ).drop_duplicates(
            ["measurement_date", "direction", "bin_start_m", "bin_end_m"], keep="first"
        )

        self._key_to_row = {}
        for row in self.rows.itertuples(index=False):
            key = TrendKey(
                row.measurement_date.strftime("%Y-%m-%d"),
                row.direction,
                float(row.bin_start_m),
                float(row.bin_end_m),
            )
            self._key_to_row[key] = row

        self._groups = {
            str(dataset_id): group.sort_values("measurement_date", kind="mergesort")
            for dataset_id, group in self.rows.groupby("dataset_id", sort=False)
        }
        self._cutoffs = self._infer_cutoffs()

    @classmethod
    def from_directory(cls, directory) -> "TrendCatalog":
        directory = Path(directory)
        frames = []
        for path in sorted(directory.rglob("*.csv")):
            try:
                frame = _read_csv_fallback(path)
            except Exception:  # noqa: BLE001 - 壊れたCSVは読み飛ばす
                continue
            if not REQUIRED_TREND_COLUMNS.issubset(frame.columns):
                continue
            if "measured_at" not in frame:
                frame["measured_at"] = ""
            if "previous_maintenance_date" not in frame:
                frame["previous_maintenance_date"] = ""
            if "maintenance_type" not in frame:
                frame["maintenance_type"] = ""
            if "maintenance_description" not in frame:
                frame["maintenance_description"] = ""
            frames.append(frame)
        if not frames:
            return cls(pd.DataFrame(columns=sorted(REQUIRED_TREND_COLUMNS)))
        return cls(pd.concat(frames, ignore_index=True))

    def _infer_cutoffs(self):
        cutoffs = {}
        segment_rows = (
            self.rows[
                [
                    "dataset_id",
                    "direction",
                    "bin_start_m",
                    "bin_end_m",
                    "segment_index",
                    "previous_maintenance_date",
                    "maintenance_type",
                    "maintenance_description",
                ]
            ]
            .drop_duplicates("dataset_id")
            .copy()
        )
        segment_rows["segment_index"] = pd.to_numeric(segment_rows["segment_index"], errors="coerce")
        for _, group in segment_rows.groupby(["direction", "bin_start_m", "bin_end_m"], sort=False):
            ordered = group.sort_values("segment_index", kind="mergesort")
            records = list(ordered.to_dict("records"))
            for index, record in enumerate(records):
                cutoff = {"date": None, "type": "", "description": ""}
                if index + 1 < len(records):
                    next_record = records[index + 1]
                    date = pd.to_datetime(next_record["previous_maintenance_date"], errors="coerce")
                    if pd.notna(date):
                        cutoff = {
                            "date": pd.Timestamp(date).normalize(),
                            "type": str(next_record["maintenance_type"] or ""),
                            "description": str(next_record["maintenance_description"] or ""),
                        }
                cutoffs[str(record["dataset_id"])] = cutoff
        return cutoffs

    def get_anchor(self, measurement_date, direction, bin_start_m, bin_end_m):
        date = pd.Timestamp(measurement_date).normalize().strftime("%Y-%m-%d")
        key = TrendKey(date, str(direction).upper(), float(bin_start_m), float(bin_end_m))
        return self._key_to_row.get(key)

    def build_trend(self, anchor_row, *, forecast_months=FUTURE_MONTHS, search_days=SEARCH_DAYS):
        """1つのanchor(measurement)から先 forecast_months か月分のトレンドを構築する。

        SPEC.md 1.2 の打ち切り仕様(次の補修月以降を mask=0・values=NaN にする)を
        ここで確定させる。以降の工程(datasets/build.py 相当)はこの
        future_values/future_mask をそのまま信頼してよい。
        """
        anchor_date = pd.Timestamp(anchor_row.measurement_date).normalize()
        dataset_id = str(anchor_row.dataset_id)
        group = self._groups[dataset_id]
        cutoff = self._cutoffs.get(dataset_id, {"date": None, "type": "", "description": ""})
        used_dates = set()
        future_values = []
        future_mask = []
        selected_dates = []

        first_month = anchor_date.replace(day=1) + pd.DateOffset(months=1)
        for month_offset in range(forecast_months):
            target_date = first_month + pd.DateOffset(months=month_offset)
            candidates = group.loc[
                (group["measurement_date"] > anchor_date)
                & ((group["measurement_date"] - target_date).dt.days.abs().le(search_days))
            ].copy()
            if cutoff["date"] is not None:
                candidates = candidates.loc[candidates["measurement_date"] < cutoff["date"]]
            candidates = candidates.loc[~candidates["measurement_date"].isin(used_dates)]

            if candidates.empty:
                future_values.append(None)
                future_mask.append(0)
                selected_dates.append(None)
                continue

            candidates["_distance_days"] = (candidates["measurement_date"] - target_date).dt.days.abs()
            candidates["_future_tie"] = (candidates["measurement_date"] > target_date).astype(int)
            chosen = candidates.sort_values(
                ["_distance_days", "_future_tie", "measurement_date"], kind="mergesort"
            ).iloc[0]
            chosen_date = pd.Timestamp(chosen["measurement_date"]).normalize()
            used_dates.add(chosen_date)
            future_values.append(float(chosen["acc_z_max"]))
            future_mask.append(1)
            selected_dates.append(chosen_date.strftime("%Y-%m-%d"))

        trend_key = (
            f"{dataset_id}|{anchor_date:%Y-%m-%d}|"
            f"{float(anchor_row.bin_start_m):.6f}|{float(anchor_row.bin_end_m):.6f}"
        )
        trend_id = hashlib.sha256(trend_key.encode("utf-8")).hexdigest()[:24]
        return {
            "trend_id": trend_id,
            "dataset_id": dataset_id,
            "measurement_date": anchor_date.strftime("%Y-%m-%d"),
            "direction": str(anchor_row.direction).upper(),
            "bin_start_m": float(anchor_row.bin_start_m),
            "bin_end_m": float(anchor_row.bin_end_m),
            "current_acc_z_max": float(anchor_row.acc_z_max),
            "future_values": _json_values(future_values),
            "future_mask": _json_values(future_mask),
            "selected_dates": _json_values(selected_dates),
            "cutoff_maintenance_date": (
                cutoff["date"].strftime("%Y-%m-%d") if cutoff["date"] is not None else ""
            ),
            "maintenance_type": cutoff["type"],
            "maintenance_description": cutoff["description"],
        }


def write_trend_catalog(records, output_path):
    columns = [
        "trend_id",
        "dataset_id",
        "measurement_date",
        "direction",
        "bin_start_m",
        "bin_end_m",
        "current_acc_z_max",
        "future_values",
        "future_mask",
        "selected_dates",
        "cutoff_maintenance_date",
        "maintenance_type",
        "maintenance_description",
    ]
    frame = pd.DataFrame.from_records(list(records), columns=columns)
    frame = frame.drop_duplicates("trend_id", keep="first")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame
