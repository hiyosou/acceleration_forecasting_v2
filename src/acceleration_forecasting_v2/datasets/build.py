"""anchor選定・履歴窓・guide検索を結合し、学習用テンソル一式を構築する。

SPEC.md 2.1のデータ契約に対応。target_modeは壁打ちで確定した"absolute"のみを
扱う(guideは常にattentionの条件としてのみ使い、ターゲットからの引き算はしない。
ただし`guide_baselines`/`guide_softmax_weights`は評価時の参考値として引き続き
算出・保存する)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import FORECAST_MONTHS, TOP_K
from acceleration_forecasting_v2.retrieval.search import GuideIndex, SearchConfig
from .anchors import AnchorCandidate
from .baseline import softmax_guide_baseline
from .normalization import Normalization


ARRAY_NAMES = (
    "history_values", "history_masks", "guide_values", "guide_masks", "guide_deltas",
    "guide_similarities", "retrieval_masks", "guide_baselines", "guide_softmax_weights",
    "segment_weights", "target_values", "target_masks",
)


@dataclass(frozen=True)
class RawSplit:
    """正規化前(物理値)の配列一式とメタデータ。"""

    arrays: dict
    metadata: pd.DataFrame


def _build_guide_arrays(guides, top_k, forecast_months, current, temperature):
    guide_values = np.full((top_k, forecast_months), np.nan, np.float32)
    guide_masks = np.zeros((top_k, forecast_months), np.float32)
    guide_deltas = np.zeros((top_k, forecast_months), np.float32)
    similarities = np.zeros(top_k, np.float32)
    retrieval_masks = np.zeros(top_k, np.float32)
    for rank, guide in enumerate(guides):
        values = np.asarray([np.nan if value is None else value for value in guide["future_values"]], np.float32)
        mask = np.asarray(guide["future_mask"], np.float32) * np.isfinite(values)
        guide_values[rank], guide_masks[rank] = values, mask
        guide_deltas[rank] = np.where(mask > 0, values - float(guide["current_acc_z_max"]), np.nan)
        similarities[rank], retrieval_masks[rank] = guide["similarity"], 1.0
    baseline, weights = softmax_guide_baseline(guide_values, guide_masks, similarities, retrieval_masks, current, temperature)
    return guide_values, guide_masks, guide_deltas, similarities, retrieval_masks, baseline, weights


def build_raw_split(
    anchors: list[AnchorCandidate],
    *,
    split: str,
    guide_index: GuideIndex,
    query_vectors_by_trend_id: dict,
    allowed_guide_splits: set,
    segment_weights: dict,
    search_config: SearchConfig = SearchConfig(),
    temperature: float = 0.1,
    top_k: int = TOP_K,
    forecast_months: int = FORECAST_MONTHS,
) -> RawSplit:
    """anchor一覧からguide検索を行い、正規化前(物理値)の配列一式を組み立てる。

    Args:
        anchors: `select_anchors_for_segment`が返したAnchorCandidateのリスト
            (複数dataset_idにまたがってよい。全て同一splitに属すること)。
        query_vectors_by_trend_id: 各anchorの`trend_id`をキーに、その日の
            波形埋め込み(1件以上、複数走行があれば複数件)のリストを持つ辞書。
            development(model_train/model_validation)側はDBに保存済みの埋め込みを、
            inference側は`retrieval.pipeline.encode_waveforms`でその場で埋め込んだ
            ものを、呼び出し側が事前に用意する(本関数は埋め込み計算自体には関与しない)。
        allowed_guide_splits: guide検索で参照してよい`model_split`の集合
            (SPEC.md 2.3: train/validation→{model_train}、inference→{model_train,model_validation})。
        segment_weights: `compute_segment_weights`の出力(このsplit内で完結したもの)。

    Returns:
        RawSplit(arrays, metadata)。arraysの各キーはARRAY_NAMESに対応するshape
        (N, ...)のnumpy配列(まだ正規化していない物理値)。target_valuesは
        target_mode=absolute(anchor.future_valuesそのもの)。
    """
    metadata_rows = []
    arrays = {name: [] for name in ARRAY_NAMES}
    for anchor in anchors:
        query_vectors = query_vectors_by_trend_id.get(anchor.trend_id, [])
        guides = []
        if query_vectors:
            guides = guide_index.search(
                np.stack(query_vectors), query_date=anchor.measurement_date.strftime("%Y-%m-%d"),
                query_dataset_id=anchor.dataset_id, query_current=anchor.current_acc_z_max,
                query_bin_start_m=anchor.bin_start_m, allowed_splits=allowed_guide_splits,
                config=search_config,
            )
        guide_values, guide_masks, guide_deltas, similarities, retrieval_masks, baseline, weights = _build_guide_arrays(
            guides, top_k, forecast_months, anchor.current_acc_z_max, temperature,
        )
        segment_weight = segment_weights[anchor.dataset_id]

        metadata_rows.append({
            "trend_id": anchor.trend_id, "dataset_id": anchor.dataset_id,
            "anchor_date": anchor.measurement_date.strftime("%Y-%m-%d"),
            "direction": anchor.direction, "bin_start_m": anchor.bin_start_m, "bin_end_m": anchor.bin_end_m,
            "current_acc_z_max": anchor.current_acc_z_max,
            "valid_history_months": anchor.valid_history_months,
            "valid_future_months": anchor.valid_future_months,
            "guide_count": len(guides), "segment_weight": segment_weight,
            # guideの出所(リーク検証・追跡用): 各guideのtrend_id/dataset_id/model_split。
            "guide_trend_ids": json.dumps([str(guide["trend_id"]) for guide in guides]),
            "guide_dataset_ids": json.dumps([str(guide["dataset_id"]) for guide in guides]),
            "guide_model_splits": json.dumps([str(guide["model_split"]) for guide in guides]),
        })
        arrays["history_values"].append(anchor.history_values)
        arrays["history_masks"].append(anchor.history_mask)
        arrays["guide_values"].append(guide_values)
        arrays["guide_masks"].append(guide_masks)
        arrays["guide_deltas"].append(guide_deltas)
        arrays["guide_similarities"].append(similarities)
        arrays["retrieval_masks"].append(retrieval_masks)
        arrays["guide_baselines"].append(baseline)
        arrays["guide_softmax_weights"].append(weights)
        arrays["segment_weights"].append(np.float32(segment_weight))
        arrays["target_values"].append(anchor.future_values)
        arrays["target_masks"].append(anchor.future_mask)

    stacked = {name: np.asarray(values, dtype=np.float32) for name, values in arrays.items()}
    metadata = pd.DataFrame(metadata_rows)
    return RawSplit(arrays=stacked, metadata=metadata)


def fit_condition_and_target_normalization(model_train_raw: RawSplit) -> tuple[Normalization, Normalization]:
    """model_trainの物理値のみから正規化統計をfitする(SPEC.md 2.1: リーク防止)。

    Returns:
        (condition_norm, target_norm)。condition_normはhistory_valuesと
        guide_values(いずれも同じ物理量 acc_z_max)を合わせてfitする
        (旧実装の"condition_normalization.json"と同じ考え方)。
    """
    history = model_train_raw.arrays["history_values"][model_train_raw.arrays["history_masks"] > 0]
    guides = model_train_raw.arrays["guide_values"][model_train_raw.arrays["guide_masks"] > 0]
    condition_values = np.concatenate([history, guides]) if guides.size else history
    condition_norm = Normalization.fit(condition_values, "model_train_condition")

    target = model_train_raw.arrays["target_values"][model_train_raw.arrays["target_masks"] > 0]
    target_norm = Normalization.fit(target, "model_train_target_absolute")
    return condition_norm, target_norm


def write_split(raw: RawSplit, split_dir) -> None:
    """RawSplit(正規化前の物理値)を`ForecastDatasetV2`が読める形で保存する。

    正規化はDataset側の`__getitem__`で適用するため、ここでは物理値のまま
    (欠測はNaNのまま、SPEC.md 1.2)保存する。
    """
    from pathlib import Path

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_NAMES:
        np.save(split_dir / f"{name}.npy", np.asarray(raw.arrays[name], dtype=np.float32))
    raw.metadata.to_csv(split_dir / "metadata.csv", index=False, encoding="utf-8-sig")


def write_dataset(raw_splits: dict, dataset_dir) -> dict:
    """model_train/model_validation/inferenceのRawSplitを保存し、正規化統計を書き出す。

    正規化統計は必ず`model_train`のみからfitする(SPEC.md 2.1・2.3)。

    Returns:
        {"condition_norm": Normalization, "target_norm": Normalization}
    """
    from pathlib import Path

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    required = ("model_train", "model_validation")
    for split in required:
        if split not in raw_splits:
            raise ValueError(f"{split} のRawSplitが必要です。")
    condition_norm, target_norm = fit_condition_and_target_normalization(raw_splits["model_train"])
    condition_norm.save(dataset_dir / "condition_normalization.json")
    target_norm.save(dataset_dir / "target_normalization.json")
    for split, raw in raw_splits.items():
        write_split(raw, dataset_dir / split)
    return {"condition_norm": condition_norm, "target_norm": target_norm}
