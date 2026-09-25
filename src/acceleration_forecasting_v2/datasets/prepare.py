"""retrievalの成果物(DB・manifest・trend_catalog・autoencoder)から、
model_train / model_validation / inference の3splitの学習用データセットを一括で作る。

各部品(anchor選定・履歴窓・guide検索・重み・正規化・書き出し)は既に実装・テスト済みで、
本モジュールはそれらを SPEC.md 2.3 のリーク防止規則に従って結び付けるだけ:

- anchorはsplitごとに、そのsplitに属するdataset_idの行からのみ選ぶ。
- guide検索の許可split: train→{model_train}、validation→{model_train}、
  inference→{model_train, model_validation}(自分自身のsplitやinferenceは見えない)。
- 正規化統計はmodel_trainのみからfit(`write_dataset`)。
- inferenceの波形埋め込みはDBに存在しないため、学習済みAutoencoderでその場で計算する。
- segment_weightはsplit内で完結(`compute_segment_weights`をsplitごとに呼ぶ)。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import (
    FORECAST_MONTHS, HISTORY_MONTHS, MIN_GUIDE_MONTHS, MIN_HISTORY_MONTHS, MIN_TARGET_MONTHS,
)
from acceleration_forecasting_v2.retrieval.database import connect_read_only
from acceleration_forecasting_v2.retrieval.pipeline import encode_waveforms
from acceleration_forecasting_v2.retrieval.search import GuideIndex, SearchConfig
from .anchors import compute_segment_weights, select_anchors_for_segment
from .build import build_raw_split, write_dataset

SPLITS = ("model_train", "model_validation", "inference")
GUIDE_SOURCE_SPLITS = {
    "model_train": {"model_train"},
    "model_validation": {"model_train"},
    "inference": {"model_train", "model_validation"},
}


def _development_query_vectors(database_path) -> dict:
    connection = connect_read_only(database_path)
    try:
        vectors: dict = {}
        for trend_id, blob, dimension in connection.execute("SELECT trend_id, embedding, embedding_dim FROM waveform_records"):
            vectors.setdefault(str(trend_id), []).append(np.frombuffer(blob, dtype=np.float32, count=int(dimension)).copy())
        return vectors
    finally:
        connection.close()


def _inference_query_vectors(artifact_dir, manifest, trend_ids, device) -> dict:
    rows = manifest.loc[(manifest["model_split"] == "inference") & manifest["trend_id"].astype(str).isin(trend_ids)]
    if rows.empty:
        return {}
    embeddings = encode_waveforms(
        artifact_dir / "waveforms.bin", len(manifest), rows["waveform_index"].astype(int).to_numpy(),
        artifact_dir / "autoencoder.pt", device=device,
    )
    vectors: dict = {}
    for trend_id, embedding in zip(rows["trend_id"].astype(str), embeddings):
        if np.all(np.isfinite(embedding)) and float(np.linalg.norm(embedding)) > 0:
            vectors.setdefault(trend_id, []).append(embedding)
    return vectors


def thin_evenly(anchors, limit):
    """anchors(日付昇順)を、日付順に均等な間隔で最大`limit`件へ間引く(決定的、乱数を使わない)。"""
    if limit is None or len(anchors) <= int(limit):
        return list(anchors)
    positions = np.unique(np.round(np.linspace(0, len(anchors) - 1, int(limit))).astype(int))
    return [anchors[int(position)] for position in positions]


def prepare_datasets(artifact_dir, output_dir, *, device=None, min_history_months=MIN_HISTORY_MONTHS,
                     min_target_months=MIN_TARGET_MONTHS, min_guide_months=MIN_GUIDE_MONTHS,
                     temperature=0.1, max_current_difference=0.5, max_datasets_per_split=None,
                     inference_max_per_segment=None) -> dict:
    """`build_retrieval_database`の出力ディレクトリから学習用データセットを構築する。

    Args:
        max_datasets_per_split: 動作確認用。指定すると各splitで(dataset_id昇順の)先頭N個のセグメントだけを使う。
        inference_max_per_segment: 指定すると、inferenceを「評価用セット」として構築する:
            正解が`min_target_months`か月以上ある(=評価に使える)anchorだけを残し、セグメントごとに
            日付が均等になるよう最大N件へ間引く。推論(DDIM)の計算量を抑えるためで、trainとvalidationには影響しない。
            未指定なら従来どおり、推論時点で利用可能なanchorを全て含める(production_ready基準)。
    """
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig")
    if "model_split" not in manifest.columns:
        raise ValueError("split_manifest.csvにmodel_split列がありません(先にretrievalの構築を実行してください)。")
    trends = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")
    unique_datasets = manifest.drop_duplicates("dataset_id")
    split_of_dataset = pd.Series(unique_datasets["model_split"].to_numpy(), index=unique_datasets["dataset_id"].astype(str))
    trends["dataset_id"] = trends["dataset_id"].astype(str)
    trends["model_split"] = trends["dataset_id"].map(split_of_dataset)
    trends = trends.dropna(subset=["model_split"])

    database_path = artifact_dir / "vector_database.sqlite"
    guide_index = GuideIndex(sqlite3.connect(f"file:{database_path.resolve().as_posix()}?mode=ro", uri=True))
    development_vectors = _development_query_vectors(database_path)
    search_config = SearchConfig(min_valid_months=int(min_guide_months), max_current_difference=max_current_difference)

    anchors_by_split = {}
    for split in SPLITS:
        split_trends = trends.loc[trends["model_split"] == split]
        dataset_ids = sorted(split_trends["dataset_id"].unique())
        if max_datasets_per_split is not None:
            dataset_ids = dataset_ids[: int(max_datasets_per_split)]
        anchors = []
        for dataset_id in dataset_ids:
            segment_anchors = select_anchors_for_segment(
                split_trends.loc[split_trends["dataset_id"] == dataset_id], split=split,
                min_history_months=min_history_months, min_target_months=min_target_months,
                history_months=HISTORY_MONTHS, forecast_months=FORECAST_MONTHS,
            )
            if split == "inference" and inference_max_per_segment is not None:
                segment_anchors = thin_evenly(
                    [anchor for anchor in segment_anchors if anchor.valid_future_months >= min_target_months],
                    inference_max_per_segment,
                )
            anchors.extend(segment_anchors)
        anchors_by_split[split] = anchors

    inference_ids = {anchor.trend_id for anchor in anchors_by_split["inference"]}
    query_vectors = {
        "model_train": development_vectors, "model_validation": development_vectors,
        "inference": _inference_query_vectors(artifact_dir, manifest, inference_ids, device),
    }

    raw_splits = {}
    for split in SPLITS:
        anchors = anchors_by_split[split]
        if not anchors:
            if split == "inference":
                continue
            raise ValueError(f"{split} に適格なanchorが1件もありません(データ量またはmin_*_monthsを見直してください)。")
        raw_splits[split] = build_raw_split(
            anchors, split=split, guide_index=guide_index, query_vectors_by_trend_id=query_vectors[split],
            allowed_guide_splits=GUIDE_SOURCE_SPLITS[split], segment_weights=compute_segment_weights(anchors),
            search_config=search_config, temperature=temperature,
        )
    write_dataset(raw_splits, output_dir)

    summary = {
        "config": {"min_history_months": int(min_history_months), "min_target_months": int(min_target_months),
                   "min_guide_months": int(min_guide_months), "temperature": float(temperature),
                   "max_current_difference": max_current_difference, "max_datasets_per_split": max_datasets_per_split,
                   "inference_max_per_segment": inference_max_per_segment},
        "splits": {split: {"anchors": int(len(raw.metadata)), "datasets": int(raw.metadata["dataset_id"].nunique()),
                           "anchors_with_guides": int((raw.metadata["guide_count"] > 0).sum())}
                   for split, raw in raw_splits.items()},
    }
    (output_dir / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
