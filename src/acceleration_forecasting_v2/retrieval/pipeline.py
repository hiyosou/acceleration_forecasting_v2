"""retrievalパイプラインの一括実行(抽出 → split確定 → Autoencoder学習 → DB構築)。

旧実装(`acceleration_retrieval`の`prepare`→`resplit`→`train`→`build-db`という
4段階CLIワークフロー)を、SPEC.md 2.1 で確定した単一DBスキーマに合わせて
1つの関数呼び出しで完結するように統合した。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .autoencoder_training import MemmapWaveformDataset, load_trained_model, train_autoencoder
from .constants import DEFAULT_TREND_DIR, DEFAULT_WAVEFORM_DIR, EMBEDDING_DIM, SAMPLES_PER_BIN
from .database import initialize_database, insert_trends, insert_waveform_record, store_metadata
from .model import l2_normalize
from .splitting import assign_model_split, split_summary
from .waveforms import extract_manifest_and_trends, open_waveforms


def encode_waveforms(waveform_path, total_count, indices, checkpoint_path, device=None, batch_size=512):
    """指定したwaveform_indexの列を、学習済みAutoencoderでL2正規化埋め込みに変換する。

    `indices`の順序どおりに埋め込み配列(shape (len(indices), embedding_dim))を返す。
    development(model_train+model_validation)側をDBへ登録する際にも、
    inference側をクエリ用にその場で埋め込む際にも、この関数を共通で使う。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model, mean, std = load_trained_model(checkpoint_path, device)
    dataset = MemmapWaveformDataset(waveform_path, total_count, indices, mean, std)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    embeddings = []
    with torch.no_grad():
        for waveform in loader:
            embedding = l2_normalize(model.encode(waveform.to(device)))
            embeddings.append(embedding.cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, model.embedding_dim), dtype=np.float32)


def write_vector_database(artifact_dir, manifest, *, device=None, embedding_dim=EMBEDDING_DIM, overwrite_db=True):
    """model_split列付きmanifestのdevelopment(model_train+model_validation)側を
    埋め込み、`vector_database.sqlite`へ登録する。inference側はDBに入れない
    (guide候補にもならない。クエリ時にその場で埋め込む、SPEC.md 2.3)。

    前提: artifact_dirに waveforms.bin / trend_catalog.csv / autoencoder.pt がある。
    """
    artifact_dir = Path(artifact_dir)
    development_rows = manifest.loc[manifest["model_split"].isin(["model_train", "model_validation"])].copy()
    if development_rows.empty:
        raise ValueError("DBへ登録するdevelopment(model_train+model_validation)レコードがありません。")
    total_count = len(manifest)
    embeddings = encode_waveforms(
        artifact_dir / "waveforms.bin", total_count,
        development_rows["waveform_index"].astype(int).to_numpy(),
        artifact_dir / "autoencoder.pt", device=device, batch_size=512,
    )

    trends = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")
    trend_split = development_rows.drop_duplicates("trend_id").set_index("trend_id")["model_split"]
    trends = trends.loc[trends["trend_id"].isin(trend_split.index)].copy()
    trends["model_split"] = trends["trend_id"].map(trend_split)

    db_path = artifact_dir / "vector_database.sqlite"
    connection = initialize_database(db_path, overwrite=overwrite_db)
    try:
        insert_trends(connection, trends)
        for (_, row), embedding in zip(development_rows.iterrows(), embeddings):
            insert_waveform_record(connection, row, embedding)
        store_metadata(connection, {
            "embedding_dim": int(embedding_dim),
            "samples_per_waveform": SAMPLES_PER_BIN,
            "record_count": int(len(development_rows)),
            "similarity": "cosine",
        })
        connection.commit()
    finally:
        connection.close()

    return {
        "database_path": str(db_path),
        "database_records": int(len(development_rows)),
        "database_trends": int(len(trends)),
    }


def build_retrieval_database(
    artifact_dir,
    *,
    waveform_dir=DEFAULT_WAVEFORM_DIR,
    trend_dir=DEFAULT_TREND_DIR,
    device=None,
    epochs=100,
    batch_size=128,
    embedding_dim=EMBEDDING_DIM,
    progress=True,
    overwrite_db=True,
):
    """生CSV一式から検索用SQLite DB(vector_database.sqlite)を構築する。

    Returns:
        summary dict。主な副作用として artifact_dir 以下に
        waveforms.bin / split_manifest.csv(model_split列付き) / trend_catalog.csv /
        autoencoder.pt / normalization.json / training_history.csv /
        vector_database.sqlite を書き出す。
    """
    artifact_dir = Path(artifact_dir)

    extraction_summary = extract_manifest_and_trends(waveform_dir, trend_dir, artifact_dir, progress=progress)

    manifest_path = artifact_dir / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    manifest = assign_model_split(manifest)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    split_stats = split_summary(manifest)

    training_summary = train_autoencoder(
        manifest_path, artifact_dir / "waveforms.bin", artifact_dir,
        device=device, epochs=epochs, batch_size=batch_size, embedding_dim=embedding_dim,
    )

    database = write_vector_database(artifact_dir, manifest, device=device, embedding_dim=embedding_dim, overwrite_db=overwrite_db)

    return {
        "extraction": extraction_summary,
        "split": split_stats,
        "training": training_summary,
        **database,
    }
