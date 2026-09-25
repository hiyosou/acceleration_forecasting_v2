"""retrieval.snapshot(既存の抽出結果の取り込み)の単体テストと、取り込み経路のend-to-end確認。"""

import numpy as np
import pandas as pd
import pytest

from acceleration_forecasting_v2.datasets.prepare import prepare_datasets
from acceleration_forecasting_v2.datasets.verify import verify_dataset_leakage
from acceleration_forecasting_v2.retrieval.pipeline import build_retrieval_from_extraction
from acceleration_forecasting_v2.retrieval.snapshot import import_extraction_snapshot
from acceleration_forecasting_v2.retrieval.trends import TrendCatalog
from acceleration_forecasting_v2.retrieval.waveforms import MANIFEST_COLUMNS


def _make_source(tmp_path, *, n_datasets=3, months=8, orphan_positions=()):
    """旧形式のsource(dataset_idなし・splitあり・waveforms.bin)と、トレンドCSVを作る。

    orphan_positions: そのmanifest行の日付をトレンドに存在しない日付にする(対応anchorなし)。
    波形は各行のwaveform_index値で埋めておき、詰め直し後に同一性を追跡できるようにする。
    """
    trend_dir, source_dir = tmp_path / "trends", tmp_path / "source"
    trend_dir.mkdir()
    source_dir.mkdir()
    raw_frames, manifest_rows = [], []
    for k in range(n_datasets):
        bin_start = 2000.0 + 1000.0 * k
        for i in range(months):
            date = (pd.Timestamp("2022-01-15") + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
            raw_frames.append({
                "dataset_id": f"seg{k:02d}", "direction": "U", "bin_start_m": bin_start, "bin_end_m": bin_start + 100.0,
                "segment_index": 1, "measurement_date": date, "acc_z_max": 1.0 + 0.1 * i, "measured_at": date,
                "previous_maintenance_date": "", "maintenance_type": "", "maintenance_description": "",
            })
            index = len(manifest_rows)
            manifest_rows.append({
                "record_id": f"old-{index}", "waveform_index": index, "measurement_id": f"m{index}",
                "measurement_date": "2099-01-01" if index in orphan_positions else date, "direction": "U",
                "bin_start_m": bin_start, "bin_end_m": bin_start + 100.0, "mean_velocity_kmh": 60.0,
                "source_csv_path": "x.csv", "waveform_sha256": "sha", "trend_id": "old-hash", "split": "database",
            })
    pd.DataFrame(raw_frames).to_csv(trend_dir / "trends.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(source_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    waveforms = np.repeat(np.arange(len(manifest_rows), dtype=np.float32)[:, None], 500, axis=1)
    waveforms.tofile(source_dir / "waveforms.bin")
    return source_dir, trend_dir, pd.DataFrame(raw_frames)


def test_import_creates_new_format_manifest_with_dataset_id_and_recomputed_trends(tmp_path):
    source_dir, trend_dir, raw = _make_source(tmp_path)
    summary = import_extraction_snapshot(source_dir, tmp_path / "out", trend_dir=trend_dir, progress=False)
    manifest = pd.read_csv(tmp_path / "out" / "split_manifest.csv", encoding="utf-8-sig")
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert "split" not in manifest.columns
    assert summary["eligible_records"] == 24 and summary["dropped_records"] == 0
    assert manifest["dataset_id"].nunique() == 3
    # 旧manifestの"old-hash"ではなく、新トレンドカタログから再計算したtrend_idが入る。
    catalog = TrendCatalog(raw)
    for row in manifest.itertuples(index=False):
        anchor = catalog.get_anchor(row.measurement_date, row.direction, row.bin_start_m, row.bin_end_m)
        assert row.trend_id == catalog.build_trend(anchor)["trend_id"]
        assert row.dataset_id == anchor.dataset_id


def test_import_rebuilds_trends_with_twelve_future_months(tmp_path):
    source_dir, trend_dir, _ = _make_source(tmp_path)
    import_extraction_snapshot(source_dir, tmp_path / "out", trend_dir=trend_dir, progress=False)
    trends = pd.read_csv(tmp_path / "out" / "trend_catalog.csv", encoding="utf-8-sig")
    import json
    assert len(trends) == 24
    assert {len(json.loads(value)) for value in trends["future_values"]} == {12}


def test_import_copies_waveforms_byte_identically_when_nothing_is_dropped(tmp_path):
    source_dir, trend_dir, _ = _make_source(tmp_path)
    import_extraction_snapshot(source_dir, tmp_path / "out", trend_dir=trend_dir, progress=False)
    assert (tmp_path / "out" / "waveforms.bin").read_bytes() == (source_dir / "waveforms.bin").read_bytes()


def test_import_drops_rows_without_trend_and_repacks_waveforms(tmp_path):
    source_dir, trend_dir, _ = _make_source(tmp_path, orphan_positions=(0, 5, 13))
    summary = import_extraction_snapshot(source_dir, tmp_path / "out", trend_dir=trend_dir, progress=False)
    assert summary["dropped_records"] == 3 and summary["eligible_records"] == 21
    manifest = pd.read_csv(tmp_path / "out" / "split_manifest.csv", encoding="utf-8-sig")
    assert manifest["waveform_index"].tolist() == list(range(21))
    waveforms = np.memmap(tmp_path / "out" / "waveforms.bin", dtype=np.float32, mode="r", shape=(21, 500))
    kept_old_indices = [i for i in range(24) if i not in (0, 5, 13)]
    # 詰め直し後も、各行の波形は元の行の波形のまま(元のwaveform_indexの値で埋めてある)。
    for new_index, old_index in enumerate(kept_old_indices):
        assert manifest.loc[new_index, "record_id"] == f"old-{old_index}"
        assert waveforms[new_index, 0] == old_index


def test_import_rejects_waveforms_file_of_wrong_size(tmp_path):
    source_dir, trend_dir, _ = _make_source(tmp_path)
    with (source_dir / "waveforms.bin").open("ab") as handle:
        handle.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="サイズ"):
        import_extraction_snapshot(source_dir, tmp_path / "out", trend_dir=trend_dir, progress=False)


def test_import_rejects_manifest_missing_required_columns(tmp_path):
    source_dir, trend_dir, _ = _make_source(tmp_path)
    manifest = pd.read_csv(source_dir / "split_manifest.csv", encoding="utf-8-sig").drop(columns=["waveform_sha256"])
    manifest.to_csv(source_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="waveform_sha256"):
        import_extraction_snapshot(source_dir, tmp_path / "out", trend_dir=trend_dir, progress=False)


def test_import_rejects_duplicate_waveform_index(tmp_path):
    source_dir, trend_dir, _ = _make_source(tmp_path)
    manifest = pd.read_csv(source_dir / "split_manifest.csv", encoding="utf-8-sig")
    manifest.loc[1, "waveform_index"] = 0
    manifest.to_csv(source_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="waveform_index"):
        import_extraction_snapshot(source_dir, tmp_path / "out", trend_dir=trend_dir, progress=False)


def test_import_rejects_when_no_record_has_a_matching_trend(tmp_path):
    source_dir, trend_dir, _ = _make_source(tmp_path, orphan_positions=tuple(range(24)))
    with pytest.raises(ValueError, match="1件もありません"):
        import_extraction_snapshot(source_dir, tmp_path / "out", trend_dir=trend_dir, progress=False)


def test_snapshot_path_runs_through_retrieval_build_and_dataset_preparation(tmp_path):
    # 取り込み → split/AE学習/DB構築 → 3splitのデータセット構築 → リーク検査 まで通る。
    source_dir, trend_dir, _ = _make_source(tmp_path, n_datasets=12, months=22)
    artifact_dir = tmp_path / "artifacts"
    import_extraction_snapshot(source_dir, artifact_dir, trend_dir=trend_dir, progress=False)
    result = build_retrieval_from_extraction(artifact_dir, device="cpu", epochs=1, batch_size=32, embedding_dim=8)
    assert (artifact_dir / "vector_database.sqlite").is_file()
    assert result["split"]["dataset_count"].keys() >= {"model_train", "model_validation", "inference"}
    summary = prepare_datasets(artifact_dir, tmp_path / "dataset", device="cpu")
    assert set(summary["splits"]) == {"model_train", "model_validation", "inference"}
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig")
    assert verify_dataset_leakage(tmp_path / "dataset", manifest) == []
