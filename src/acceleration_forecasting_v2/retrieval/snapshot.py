"""既存の波形抽出結果(旧`acceleration_retrieval`のwaveforms.bin/split_manifest.csv)を
入力スナップショットとして取り込む。

生CSVから波形を切り出す処理(数時間かかる)を省くための、一度限りの入力取り込み。
コード上の依存ではない: 旧リポジトリのモジュールはimportせず、ファイルを読むだけ。
抽出ロジックが新旧で同一であること(record_id・波形SHA256・trend_id・トレンドが実データ10ファイルで
全件一致)は PROGRESS.md に記録済み。

取り込み時に新リポジトリの仕様へ揃える点:
- トレンドは`FUTURE_MONTHS`(12か月)で再構築する(旧は18か月)。
- manifestに`dataset_id`を付与し、旧の日付単位`split`列は捨てる(分割は`splitting.py`が行う)。
- 新しいトレンドカタログに対応するanchorが無い行は除外し、波形は詰め直す。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_TREND_DIR, SAMPLES_PER_BIN
from .trends import TrendCatalog, write_trend_catalog
from .waveforms import MANIFEST_COLUMNS

REQUIRED_SOURCE_COLUMNS = {"record_id", "waveform_index", "measurement_id", "measurement_date", "direction",
                           "bin_start_m", "bin_end_m", "mean_velocity_kmh", "source_csv_path", "waveform_sha256"}


def import_extraction_snapshot(source_artifact_dir, artifact_dir, *, trend_dir=DEFAULT_TREND_DIR, progress=True) -> dict:
    """source_artifact_dir(waveforms.bin + split_manifest.csv)を artifact_dir へ取り込む。

    artifact_dirには waveforms.bin / split_manifest.csv(dataset_id付き、model_splitなし) /
    trend_catalog.csv / prepare_summary.json が書き出される。
    """
    source_dir, artifact_dir = Path(source_artifact_dir), Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = pd.read_csv(source_dir / "split_manifest.csv", encoding="utf-8-sig")
    missing = REQUIRED_SOURCE_COLUMNS - set(source_manifest.columns)
    if missing:
        raise ValueError(f"取り込み元のmanifestに列がありません: {sorted(missing)}")
    source_waveforms = source_dir / "waveforms.bin"
    record_count = len(source_manifest)
    expected_size = record_count * SAMPLES_PER_BIN * np.dtype(np.float32).itemsize
    if source_waveforms.stat().st_size != expected_size:
        raise ValueError(f"waveforms.binのサイズが不正です: expected={expected_size}, actual={source_waveforms.stat().st_size}")
    indices = source_manifest["waveform_index"].to_numpy()
    if len(set(indices.tolist())) != record_count or indices.min() < 0 or indices.max() >= record_count:
        raise ValueError("waveform_indexが一意な0..N-1ではありません。")

    catalog = TrendCatalog.from_directory(trend_dir)
    trend_by_key: dict = {}
    trend_ids, dataset_ids, keep = [], [], []
    iterator = source_manifest.itertuples(index=False)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=record_count, desc="トレンド再構築", unit="record")
    for row in iterator:
        key = (str(row.measurement_date), str(row.direction).upper(), float(row.bin_start_m), float(row.bin_end_m))
        if key not in trend_by_key:
            anchor = catalog.get_anchor(*key)
            trend_by_key[key] = None if anchor is None else catalog.build_trend(anchor)
        trend = trend_by_key[key]
        keep.append(trend is not None)
        trend_ids.append(None if trend is None else trend["trend_id"])
        dataset_ids.append(None if trend is None else trend["dataset_id"])

    keep = np.asarray(keep, dtype=bool)
    manifest = source_manifest.assign(dataset_id=dataset_ids, trend_id=trend_ids).loc[keep].copy()
    dropped = int((~keep).sum())
    if manifest.empty:
        raise ValueError("トレンドに対応する波形が1件もありません(trend_dirを確認してください)。")

    destination = artifact_dir / "waveforms.bin"
    if dropped == 0 and (indices == np.arange(record_count)).all():
        shutil.copyfile(source_waveforms, destination)
    else:
        source = np.memmap(source_waveforms, dtype=np.float32, mode="r", shape=(record_count, SAMPLES_PER_BIN))
        with destination.open("wb") as handle:
            for position in np.flatnonzero(keep):
                handle.write(np.asarray(source[int(indices[position])], dtype=np.float32).tobytes(order="C"))
    manifest["waveform_index"] = np.arange(len(manifest))
    manifest = manifest.loc[:, MANIFEST_COLUMNS]
    manifest.to_csv(artifact_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    trends = write_trend_catalog([trend for trend in trend_by_key.values() if trend is not None],
                                 artifact_dir / "trend_catalog.csv")

    summary = {
        "source": "snapshot", "source_artifact_dir": str(source_dir), "eligible_records": int(len(manifest)),
        "dataset_count": int(manifest["dataset_id"].nunique()), "trend_records": int(len(trends)),
        "dropped_records": dropped, "waveforms_path": str(destination),
        "manifest_path": str(artifact_dir / "split_manifest.csv"),
        "trend_catalog_path": str(artifact_dir / "trend_catalog.csv"),
    }
    (artifact_dir / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
