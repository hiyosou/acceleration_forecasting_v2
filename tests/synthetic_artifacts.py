"""テスト用の合成retrieval成果物(manifest・trend_catalog・waveforms.bin・autoencoder.pt・vector DB)。

実データを使わずに、`prepare_datasets`以降(学習・推論・評価)を通しで検証するためのフィクスチャ。
"""

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_v2.retrieval.model import WaveformAutoencoder
from acceleration_forecasting_v2.retrieval.pipeline import write_vector_database
from acceleration_forecasting_v2.retrieval.splitting import assign_model_split
from acceleration_forecasting_v2.retrieval.trends import TrendCatalog, write_trend_catalog


def build_artifacts(artifact_dir, *, n_datasets=12, months=22, embedding_dim=16, seed=0, inference_offset=0.0):
    """n_datasets個のセグメント(各`months`か月ぶんの毎月の計測)からなる成果物一式を作る。

    各セグメントは別の区間(bin_start_mが1000mずつ離れている)で、全セグメントが同じ日付列を持つ
    (guide検索は「同じ日付は除外」なので、他セグメントの別の月がguide候補になる)。
    inference_offset: inference側の波形にだけ足す値(model_trainのみから統計を取っている検証用)。
    """
    from pathlib import Path

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    trend_records, manifest_rows = [], []
    for k in range(n_datasets):
        bin_start = 2000.0 + 1000.0 * k
        dates = [pd.Timestamp("2022-01-15") + pd.DateOffset(months=i) for i in range(months)]
        raw = pd.DataFrame([{
            "dataset_id": f"seg{k:02d}", "direction": "U", "bin_start_m": bin_start, "bin_end_m": bin_start + 100.0,
            "segment_index": 1, "measurement_date": d.strftime("%Y-%m-%d"),
            "acc_z_max": 1.0 + 0.01 * k + 0.02 * i, "measured_at": d.strftime("%Y-%m-%d"),
            "previous_maintenance_date": "", "maintenance_type": "", "maintenance_description": "",
        } for i, d in enumerate(dates)])
        catalog = TrendCatalog(raw)
        for d in dates:
            trend = catalog.build_trend(catalog.get_anchor(d.strftime("%Y-%m-%d"), "U", bin_start, bin_start + 100.0))
            trend_records.append(trend)
            manifest_rows.append({
                "record_id": f"rec-{trend['trend_id']}", "waveform_index": len(manifest_rows),
                "measurement_id": f"meas-{trend['trend_id']}", "measurement_date": trend["measurement_date"],
                "direction": "U", "bin_start_m": bin_start, "bin_end_m": bin_start + 100.0, "mean_velocity_kmh": 60.0,
                "source_csv_path": "dummy.csv", "waveform_sha256": "abc", "dataset_id": trend["dataset_id"],
                "trend_id": trend["trend_id"],
            })
    write_trend_catalog(trend_records, artifact_dir / "trend_catalog.csv")
    manifest = assign_model_split(pd.DataFrame(manifest_rows), seed=seed)

    waveforms = rng.normal(size=(len(manifest), 500)).astype(np.float32)
    waveforms[(manifest["model_split"] == "inference").to_numpy()] += np.float32(inference_offset)
    waveforms.tofile(artifact_dir / "waveforms.bin")
    manifest.to_csv(artifact_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")

    torch.manual_seed(seed)
    model = WaveformAutoencoder(embedding_dim)
    torch.save({"model_state_dict": model.state_dict(), "embedding_dim": embedding_dim, "mean": 0.0, "std": 1.0},
               artifact_dir / "autoencoder.pt")
    write_vector_database(artifact_dir, manifest, device="cpu", embedding_dim=embedding_dim)
    return manifest
