"""生の振動計測CSVから100m区間ごとの波形を抽出し、trend_catalogと突き合わせる。

`acceleration_retrieval/data.py` を出発点に移植したもの。旧実装との主な差分は
以下の2点のみ(ロジック自体は変更していない):
1. `position_correction_common.load_vibration_csv` の代わりに、本パッケージへ
   複製した `csv_io.load_vibration_csv` を使う(csv_io.py の docstring 参照)。
2. 旧実装は抽出直後に「日付単位のdatabase/query分割」を行っていたが、本実装は
   分割を行わない(dataset_id単位の分割は `splitting.py` に分離した)。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .csv_io import load_vibration_csv
from .constants import (
    ACCELERATION_COLUMN,
    BIN_WIDTH_M,
    DISTANCE_COLUMN,
    END_M,
    MAX_MEAN_SPEED_KMH,
    MIN_MEAN_SPEED_KMH,
    SAMPLES_PER_BIN,
    START_M,
    STEP_M,
    VELOCITY_COLUMN,
)
from .trends import TrendCatalog, write_trend_catalog


MEASUREMENT_PATTERN = re.compile(
    r"^NAGANO2_(\d{8})_(\d{6})_([UD])_(\d+)(?:_.*)?\.csv$",
    re.IGNORECASE,
)
MANIFEST_COLUMNS = [
    "record_id",
    "waveform_index",
    "measurement_id",
    "measurement_date",
    "direction",
    "bin_start_m",
    "bin_end_m",
    "mean_velocity_kmh",
    "source_csv_path",
    "waveform_sha256",
    "dataset_id",
    "trend_id",
]


def parse_measurement(path):
    match = MEASUREMENT_PATTERN.match(Path(path).name)
    if match is None:
        return None
    date_text, time_text, direction, sequence = match.groups()
    return {
        "measurement_id": f"{date_text}_{time_text}_{direction.upper()}_{int(sequence)}",
        "measurement_date": pd.to_datetime(date_text, format="%Y%m%d", errors="raise").strftime("%Y-%m-%d"),
        "direction": direction.upper(),
    }


def extract_waveform(
    frame,
    bin_start_m,
    bin_end_m,
    expected_samples=SAMPLES_PER_BIN,
    step_m=STEP_M,
    min_speed=MIN_MEAN_SPEED_KMH,
    max_speed=MAX_MEAN_SPEED_KMH,
):
    required = {DISTANCE_COLUMN, ACCELERATION_COLUMN, VELOCITY_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        return None, f"missing_columns:{','.join(sorted(missing))}"

    distances = pd.to_numeric(frame[DISTANCE_COLUMN], errors="coerce")
    mask = (distances >= float(bin_start_m) - 1e-9) & (distances < float(bin_end_m) - 1e-9)
    segment = frame.loc[mask, [DISTANCE_COLUMN, ACCELERATION_COLUMN, VELOCITY_COLUMN]].copy()
    segment[DISTANCE_COLUMN] = pd.to_numeric(segment[DISTANCE_COLUMN], errors="coerce")
    segment = segment.sort_values(DISTANCE_COLUMN, kind="mergesort")
    if len(segment) != expected_samples:
        return None, "sample_count"

    x = segment[DISTANCE_COLUMN].to_numpy(dtype=float)
    expected_x = float(bin_start_m) + np.arange(expected_samples) * float(step_m)
    if not np.isfinite(x).all() or not np.allclose(x, expected_x, rtol=0.0, atol=1e-6):
        return None, "distance_grid"

    waveform = pd.to_numeric(segment[ACCELERATION_COLUMN], errors="coerce").to_numpy(dtype=float)
    velocity = pd.to_numeric(segment[VELOCITY_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(waveform).all():
        return None, "waveform_nonfinite"
    if not np.isfinite(velocity).all():
        return None, "velocity_nonfinite"
    mean_velocity = float(velocity.mean())
    if not min_speed <= mean_velocity <= max_speed:
        return None, "mean_speed_out_of_range"
    return {"waveform": waveform.astype(np.float32), "mean_velocity_kmh": mean_velocity}, None


def _record_id(measurement_id, direction, bin_start_m, bin_end_m, waveform_hash):
    key = f"{measurement_id}|{direction}|{float(bin_start_m):.6f}|{float(bin_end_m):.6f}|{waveform_hash}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _append_diagnostic(diagnostics, path, measurement, start_m, end_m, reason):
    diagnostics.append(
        {
            "source_csv_path": str(path),
            "measurement_id": measurement["measurement_id"] if measurement is not None else "",
            "measurement_date": measurement["measurement_date"] if measurement is not None else "",
            "direction": measurement["direction"] if measurement is not None else "",
            "bin_start_m": start_m,
            "bin_end_m": end_m,
            "reason": reason,
        }
    )


def group_frame_by_bins(frame, start_m, end_m, bin_width_m):
    if DISTANCE_COLUMN not in frame.columns:
        return {}
    distances = pd.to_numeric(frame[DISTANCE_COLUMN], errors="coerce").to_numpy(dtype=float)
    inside = np.isfinite(distances) & (distances >= float(start_m) - 1e-9) & (distances < float(end_m) - 1e-9)
    row_indices = np.flatnonzero(inside)
    if row_indices.size == 0:
        return {}
    bin_indices = np.floor((distances[row_indices] - float(start_m) + 1e-9) / float(bin_width_m)).astype(int)
    grouped = {}
    for bin_index in np.unique(bin_indices):
        grouped[int(bin_index)] = frame.iloc[row_indices[bin_indices == bin_index]]
    return grouped


def extract_manifest_and_trends(
    waveform_dir,
    trend_dir,
    artifact_dir,
    *,
    start_m=START_M,
    end_m=END_M,
    bin_width_m=BIN_WIDTH_M,
    progress=True,
):
    """波形CSV一式を走査し、waveforms.bin・manifest・trend_catalogを書き出す。

    dataset_id単位のsplit割り当ては行わない(`splitting.py`が別途担当する)。

    Returns:
        summary dict(件数などのサマリ)。書き出したファイルは
        artifact_dir/waveforms.bin, split_manifest.csv, trend_catalog.csv,
        exclusion_diagnostics.csv, prepare_summary.json。
    """
    waveform_dir = Path(waveform_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    binary_path = artifact_dir / "waveforms.bin"
    manifest_path = artifact_dir / "split_manifest.csv"
    trend_path = artifact_dir / "trend_catalog.csv"
    diagnostic_path = artifact_dir / "exclusion_diagnostics.csv"
    summary_path = artifact_dir / "prepare_summary.json"
    temporary_paths = {
        binary_path: artifact_dir / ".waveforms.bin.tmp",
        manifest_path: artifact_dir / ".split_manifest.csv.tmp",
        trend_path: artifact_dir / ".trend_catalog.csv.tmp",
        diagnostic_path: artifact_dir / ".exclusion_diagnostics.csv.tmp",
        summary_path: artifact_dir / ".prepare_summary.json.tmp",
    }

    trend_catalog = TrendCatalog.from_directory(trend_dir)
    diagnostics = []
    manifest_records = []
    trend_records = {}
    seen_identity = {}
    bin_starts = np.arange(float(start_m), float(end_m), float(bin_width_m))
    source_paths = sorted(waveform_dir.rglob("*.csv"))

    with temporary_paths[binary_path].open("wb") as binary_file:
        iterator = tqdm(source_paths, desc="波形CSVを処理", unit="file", dynamic_ncols=True) if progress else source_paths
        for source_path in iterator:
            measurement = parse_measurement(source_path)
            if measurement is None:
                continue
            try:
                frame = load_vibration_csv(str(source_path))
            except Exception as exc:  # noqa: BLE001 - 個別ファイルの読み込み失敗は除外理由として記録する
                _append_diagnostic(diagnostics, source_path, measurement, "", "", f"read_error:{exc}")
                continue
            if frame is None or frame.empty:
                _append_diagnostic(diagnostics, source_path, measurement, "", "", "empty_csv")
                continue

            grouped_bins = group_frame_by_bins(frame, start_m, end_m, bin_width_m)
            for bin_index, bin_start in enumerate(bin_starts):
                bin_end = float(bin_start + bin_width_m)
                segment_frame = grouped_bins.get(bin_index, frame.iloc[0:0])
                extracted, reason = extract_waveform(segment_frame, bin_start, bin_end)
                if reason is not None:
                    _append_diagnostic(diagnostics, source_path, measurement, float(bin_start), bin_end, reason)
                    continue

                anchor = trend_catalog.get_anchor(measurement["measurement_date"], measurement["direction"], bin_start, bin_end)
                if anchor is None:
                    _append_diagnostic(diagnostics, source_path, measurement, float(bin_start), bin_end, "trend_not_found")
                    continue

                waveform_bytes = extracted["waveform"].tobytes(order="C")
                waveform_hash = hashlib.sha256(waveform_bytes).hexdigest()
                identity = (measurement["measurement_id"], measurement["direction"], float(bin_start), bin_end)
                previous_hash = seen_identity.get(identity)
                if previous_hash is not None:
                    duplicate_reason = "duplicate" if previous_hash == waveform_hash else "measurement_identity_conflict"
                    _append_diagnostic(diagnostics, source_path, measurement, float(bin_start), bin_end, duplicate_reason)
                    continue
                seen_identity[identity] = waveform_hash

                trend_record = trend_catalog.build_trend(anchor)
                trend_records[trend_record["trend_id"]] = trend_record
                waveform_index = len(manifest_records)
                binary_file.write(waveform_bytes)
                manifest_records.append(
                    {
                        "record_id": _record_id(measurement["measurement_id"], measurement["direction"], bin_start, bin_end, waveform_hash),
                        "waveform_index": waveform_index,
                        **measurement,
                        "bin_start_m": float(bin_start),
                        "bin_end_m": bin_end,
                        "mean_velocity_kmh": extracted["mean_velocity_kmh"],
                        "source_csv_path": str(source_path.resolve()),
                        "waveform_sha256": waveform_hash,
                        "dataset_id": trend_record["dataset_id"],
                        "trend_id": trend_record["trend_id"],
                    }
                )

    manifest = pd.DataFrame.from_records(manifest_records, columns=MANIFEST_COLUMNS)
    manifest.to_csv(temporary_paths[manifest_path], index=False, encoding="utf-8-sig")
    write_trend_catalog(trend_records.values(), temporary_paths[trend_path])
    pd.DataFrame.from_records(diagnostics).to_csv(temporary_paths[diagnostic_path], index=False, encoding="utf-8-sig")
    summary = {
        "eligible_records": int(len(manifest)),
        "dataset_count": int(manifest["dataset_id"].nunique()) if not manifest.empty else 0,
        "trend_records": int(len(trend_records)),
        "excluded": int(len(diagnostics)),
        "waveforms_path": str(binary_path),
        "manifest_path": str(manifest_path),
        "trend_catalog_path": str(trend_path),
        "diagnostics_path": str(diagnostic_path),
    }
    temporary_paths[summary_path].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for final_path, temporary_path in temporary_paths.items():
        os.replace(temporary_path, final_path)
    return summary


def open_waveforms(binary_path, record_count):
    return np.memmap(binary_path, dtype=np.float32, mode="r", shape=(int(record_count), SAMPLES_PER_BIN))
