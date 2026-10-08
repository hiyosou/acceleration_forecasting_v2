"""guide検索方法だけを6か月履歴ベース(`GuideIndex.search_by_history`)に差し替えた
inference splitを構築する(2026-10-08、ユーザー判断による検証用)。

既存の`prepare_datasets`の出力(embeddingベース検索によるguideを含む本番データセット)を
そのまま再利用し、「モデル・条件は一緫」を保つため以下を一切変更しない:
- history_values / history_masks / segment_weights / target_values / target_masks
  (いずれも検索方法に依存しない量)
- condition_normalization.json / target_normalization.json
  (既存チェックポイントが学習時に使った正規化統計そのもの)
- model_train / model_validation split(全ファイルを無変更でコピーするだけ)

差し替えるのは`inference` splitのguide関連配列(guide_values/guide_masks/guide_deltas/
guide_similarities/retrieval_masks、参考値のguide_baselines/guide_softmax_weights)だけ。
`datasets.self_reference.build_self_reference_dataset`と同じ「既存データセットの一部だけ
差し替えて書き出す」設計を踏襲している。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import FORECAST_MONTHS, TOP_K
from acceleration_forecasting_v2.retrieval.database import connect_read_only
from acceleration_forecasting_v2.retrieval.search import GuideIndex, SearchConfig
from .build import ARRAY_NAMES, _build_guide_arrays

SPLITS = ("model_train", "model_validation", "inference")
# prepare.pyのGUIDE_SOURCE_SPLITS["inference"]と同じ(inferenceは自分自身のsplitを見ない)。
INFERENCE_GUIDE_SOURCE_SPLITS = {"model_train", "model_validation"}


def build_history_search_inference_dataset(
    source_dataset_dir, artifact_dir, output_dataset_dir, *,
    search_config: SearchConfig = SearchConfig(), top_k: int = TOP_K,
    forecast_months: int = FORECAST_MONTHS, temperature: float = 0.1,
) -> dict:
    """`inference` splitのguideだけを`search_by_history`で作り直した新データセットを書き出す。

    Returns:
        summary dict(件数・設定)。`output_dataset_dir/dataset_summary.json`にも書く。
    """
    source_dataset_dir = Path(source_dataset_dir)
    artifact_dir = Path(artifact_dir)
    output_dataset_dir = Path(output_dataset_dir)
    if source_dataset_dir.resolve() == output_dataset_dir.resolve():
        raise ValueError("出力先は元のデータセットと別ディレクトリにしてください。")
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    for name in ("condition_normalization.json", "target_normalization.json"):
        shutil.copy2(source_dataset_dir / name, output_dataset_dir / name)

    for split in ("model_train", "model_validation"):
        split_dir = source_dataset_dir / split
        if not split_dir.is_dir():
            continue
        out_dir = output_dataset_dir / split
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(split_dir, out_dir)

    database_path = artifact_dir / "vector_database.sqlite"
    guide_index = GuideIndex(connect_read_only(database_path))

    source_split_dir = source_dataset_dir / "inference"
    out_dir = output_dataset_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    history_values = np.load(source_split_dir / "history_values.npy")
    history_masks = np.load(source_split_dir / "history_masks.npy")
    metadata = pd.read_csv(source_split_dir / "metadata.csv", encoding="utf-8-sig")

    guide_values_all, guide_masks_all, guide_deltas_all = [], [], []
    similarities_all, retrieval_masks_all, baselines_all, weights_all = [], [], [], []
    guide_counts, guide_trend_id_lists, guide_dataset_id_lists, guide_split_lists = [], [], [], []

    for position, row in enumerate(metadata.itertuples(index=False)):
        current = float(row.current_acc_z_max)
        guides = guide_index.search_by_history(
            history_values[position], history_masks[position],
            query_date=str(row.anchor_date), query_dataset_id=str(row.dataset_id),
            query_current=current, query_bin_start_m=float(row.bin_start_m),
            allowed_splits=INFERENCE_GUIDE_SOURCE_SPLITS, config=search_config,
        )
        guide_values, guide_masks, guide_deltas, similarities, retrieval_masks, baseline, weights = _build_guide_arrays(
            guides, top_k, forecast_months, current, temperature,
        )
        guide_values_all.append(guide_values)
        guide_masks_all.append(guide_masks)
        guide_deltas_all.append(guide_deltas)
        similarities_all.append(similarities)
        retrieval_masks_all.append(retrieval_masks)
        baselines_all.append(baseline)
        weights_all.append(weights)
        guide_counts.append(len(guides))
        guide_trend_id_lists.append(json.dumps([str(guide["trend_id"]) for guide in guides]))
        guide_dataset_id_lists.append(json.dumps([str(guide["dataset_id"]) for guide in guides]))
        guide_split_lists.append(json.dumps([str(guide["model_split"]) for guide in guides]))

    overrides = {
        "guide_values": np.asarray(guide_values_all, dtype=np.float32),
        "guide_masks": np.asarray(guide_masks_all, dtype=np.float32),
        "guide_deltas": np.asarray(guide_deltas_all, dtype=np.float32),
        "guide_similarities": np.asarray(similarities_all, dtype=np.float32),
        "retrieval_masks": np.asarray(retrieval_masks_all, dtype=np.float32),
        "guide_baselines": np.asarray(baselines_all, dtype=np.float32),
        "guide_softmax_weights": np.asarray(weights_all, dtype=np.float32),
    }
    for name in ARRAY_NAMES:
        if name in overrides:
            np.save(out_dir / f"{name}.npy", overrides[name])
        else:
            shutil.copy2(source_split_dir / f"{name}.npy", out_dir / f"{name}.npy")

    metadata = metadata.copy()
    metadata["guide_search_method"] = "history_rmse"
    metadata["guide_count"] = guide_counts
    metadata["guide_trend_ids"] = guide_trend_id_lists
    metadata["guide_dataset_ids"] = guide_dataset_id_lists
    metadata["guide_model_splits"] = guide_split_lists
    metadata.to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    summary = {
        "guide_search_method": "history_rmse",
        "source_dataset": str(source_dataset_dir), "artifact_dir": str(artifact_dir),
        "output_dataset": str(output_dataset_dir), "search_config": asdict(search_config),
        "inference_count": int(len(metadata)),
        "anchors_with_guides": int(sum(1 for count in guide_counts if count > 0)),
    }
    (output_dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
