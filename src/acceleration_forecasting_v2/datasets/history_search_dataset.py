"""guide検索方法だけを6か月履歴ベース(`GuideIndex.search_by_history`)に差し替えた
データセットを構築する。

2026-10-08: `inference` splitのguideだけを差し替える検証用途
(`build_history_search_inference_dataset`)。既存の`prepare_datasets`の出力
(embeddingベース検索によるguideを含む本番データセット)をそのまま再利用し、
「モデル・条件は一緫」を保つため以下を一切変更しない:
- history_values / history_masks / segment_weights / target_values / target_masks
  (いずれも検索方法に依存しない量)
- condition_normalization.json / target_normalization.json
  (既存チェックポイントが学習時に使った正規化統計そのもの)
- model_train / model_validation split(全ファイルを無変更でコピーするだけ)

2026-10-09: P3(GENERATIVE_MODULE_NEXT_STEPS.md)用に、3 split全てのguideを
差し替えて**再学習用データセットを作り直す**`build_history_search_dataset`を追加。
`build_history_search_inference_dataset`はこの一般化版への薄いラッパーに書き換えた
(`splits_to_rebuild=("inference",)`で呼ぶだけ。model_train/model_validationが
不変なら正規化統計の再fit結果も数値的に既存と同一になるため、挙動・既存テストは
変わらない)。

いずれも`datasets.self_reference.build_self_reference_dataset`と同じ「既存
データセットの一部だけ差し替えて書き出す」設計を踏襲している。全split版は
`datasets.build.RawSplit`/`write_dataset`をそのまま再利用する(正規化統計の
再fitや3 split分の書き出しを自前実装しない)。
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
from .build import ARRAY_NAMES, RawSplit, _build_guide_arrays, fit_condition_and_target_normalization, write_split
from .normalization import Normalization
from .prepare import GUIDE_SOURCE_SPLITS

SPLITS = ("model_train", "model_validation", "inference")


def _search_split_guides(guide_index, split_dir, *, allowed_splits, search_config, top_k,
                          forecast_months, temperature):
    """1split分のhistory_values/metadataから、search_by_historyでguide配列一式を作る。

    `build_history_search_inference_dataset`(inference限定)と
    `build_history_search_dataset`(全split版)の共通ループ。
    """
    history_values = np.load(split_dir / "history_values.npy")
    history_masks = np.load(split_dir / "history_masks.npy")
    metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")

    guide_values_all, guide_masks_all, guide_deltas_all = [], [], []
    similarities_all, retrieval_masks_all, baselines_all, weights_all = [], [], [], []
    guide_counts, guide_trend_id_lists, guide_dataset_id_lists, guide_split_lists = [], [], [], []

    for position, row in enumerate(metadata.itertuples(index=False)):
        current = float(row.current_acc_z_max)
        guides = guide_index.search_by_history(
            history_values[position], history_masks[position],
            query_date=str(row.anchor_date), query_dataset_id=str(row.dataset_id),
            query_current=current, query_bin_start_m=float(row.bin_start_m),
            allowed_splits=allowed_splits, config=search_config,
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
    new_metadata = metadata.copy()
    new_metadata["guide_search_method"] = "history_rmse"
    new_metadata["guide_count"] = guide_counts
    new_metadata["guide_trend_ids"] = guide_trend_id_lists
    new_metadata["guide_dataset_ids"] = guide_dataset_id_lists
    new_metadata["guide_model_splits"] = guide_split_lists
    return overrides, new_metadata


def build_history_search_dataset(
    source_dataset_dir, artifact_dir, output_dataset_dir, *,
    splits_to_rebuild=SPLITS, search_config: SearchConfig = SearchConfig(), top_k: int = TOP_K,
    forecast_months: int = FORECAST_MONTHS, temperature: float = 0.1,
) -> dict:
    """`splits_to_rebuild`で指定した各splitのguideを`search_by_history`で作り直す
    (P3: 学習データ自体を新しい検索方法で作り直す)。

    `splits_to_rebuild`に含まれないsplitは`shutil.copytree`でbyte-identicalにコピー
    するだけ(`pd.read_csv`→`to_csv`の往復はfloat列の桁落ちを起こし得るため使わない
    ——例: `segment_weight`の`1/12`が`0.08333333333333333`→`0.0833333333333333`に
    丸められる、等)。`model_train`を再構築する場合のみ、その新しい配列から
    `fit_condition_and_target_normalization`で正規化統計を再fitする(historyは不変・
    guideのみ変わるため、target_normalizationは数値的に不変、condition_normalization
    のみ実質的に変わりうる)。`model_train`を再構築しない場合は正規化統計もそのまま
    コピーする(`build_history_search_inference_dataset`の後方互換性はこれに依る)。

    Returns:
        summary dict。`output_dataset_dir/dataset_summary.json`にも書く。
    """
    source_dataset_dir = Path(source_dataset_dir)
    artifact_dir = Path(artifact_dir)
    output_dataset_dir = Path(output_dataset_dir)
    if source_dataset_dir.resolve() == output_dataset_dir.resolve():
        raise ValueError("出力先は元のデータセットと別ディレクトリにしてください。")
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    splits_to_rebuild = set(splits_to_rebuild)

    guide_index = None
    if splits_to_rebuild:
        database_path = artifact_dir / "vector_database.sqlite"
        guide_index = GuideIndex(connect_read_only(database_path))

    raw_splits: dict[str, RawSplit] = {}
    rebuilt_counts = {}
    for split in SPLITS:
        split_dir = source_dataset_dir / split
        if not split_dir.is_dir():
            continue
        if split in splits_to_rebuild:
            arrays = {name: np.load(split_dir / f"{name}.npy") for name in ARRAY_NAMES}
            overrides, metadata = _search_split_guides(
                guide_index, split_dir, allowed_splits=GUIDE_SOURCE_SPLITS[split],
                search_config=search_config, top_k=top_k, forecast_months=forecast_months,
                temperature=temperature,
            )
            arrays.update(overrides)
            raw_splits[split] = RawSplit(arrays=arrays, metadata=metadata)
            rebuilt_counts[split] = int(len(metadata))
        else:
            out_dir = output_dataset_dir / split
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(split_dir, out_dir)

    if "model_train" in raw_splits:
        condition_norm, target_norm = fit_condition_and_target_normalization(raw_splits["model_train"])
        condition_norm.save(output_dataset_dir / "condition_normalization.json")
        target_norm.save(output_dataset_dir / "target_normalization.json")
    else:
        for name in ("condition_normalization.json", "target_normalization.json"):
            shutil.copy2(source_dataset_dir / name, output_dataset_dir / name)
        condition_norm = Normalization.load(output_dataset_dir / "condition_normalization.json")

    for split, raw in raw_splits.items():
        write_split(raw, output_dataset_dir / split)

    summary = {
        "guide_search_method": "history_rmse", "splits_rebuilt": sorted(splits_to_rebuild),
        "source_dataset": str(source_dataset_dir), "artifact_dir": str(artifact_dir),
        "output_dataset": str(output_dataset_dir), "search_config": asdict(search_config),
        "rebuilt_counts": rebuilt_counts,
        "condition_normalization": {"mean": condition_norm.mean, "std": condition_norm.std,
                                     "count": condition_norm.count},
    }
    (output_dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_history_search_inference_dataset(
    source_dataset_dir, artifact_dir, output_dataset_dir, *,
    search_config: SearchConfig = SearchConfig(), top_k: int = TOP_K,
    forecast_months: int = FORECAST_MONTHS, temperature: float = 0.1,
) -> dict:
    """`inference` splitのguideだけを`search_by_history`で作り直した新データセットを書き出す。

    `build_history_search_dataset(..., splits_to_rebuild=("inference",))`への薄い
    ラッパー(2026-10-09、全split版の追加に伴いリファクタ)。model_train/
    model_validationは再構築されないため、正規化統計は数値的に元のデータセットと
    一致する。

    Returns:
        summary dict(件数・設定)。`output_dataset_dir/dataset_summary.json`にも書く。
    """
    summary = build_history_search_dataset(
        source_dataset_dir, artifact_dir, output_dataset_dir,
        splits_to_rebuild=("inference",), search_config=search_config, top_k=top_k,
        forecast_months=forecast_months, temperature=temperature,
    )
    output_dataset_dir = Path(output_dataset_dir)
    metadata = pd.read_csv(output_dataset_dir / "inference" / "metadata.csv", encoding="utf-8-sig")
    summary["inference_count"] = int(len(metadata))
    summary["anchors_with_guides"] = int((metadata["guide_count"] > 0).sum())
    (output_dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
