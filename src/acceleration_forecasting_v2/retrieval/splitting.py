"""dataset_id(施工間セグメント)単位の2段階ランダム分割。

`acceleration_retrieval/splitting.py` の `create_dataset_split` を出発点に、
「未分割artifactsを読んでresplitする」という旧来の2段階運用ではなく、
`waveforms.extract_manifest_and_trends` の出力に対して直接1回で分割を確定させる
形に整理した(SPEC.md 2.3 で踏襲を確認した分割方針そのものに変更はない)。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEVELOPMENT_RATIO, MODEL_TRAIN_RATIO_WITHIN_DEVELOPMENT, RANDOM_SEED


def _split_groups(values, first_ratio, seed):
    unique_values = sorted({str(value) for value in values})
    if len(unique_values) < 2:
        raise ValueError("分割には2件以上のdataset_idが必要です。")
    shuffled = unique_values.copy()
    random.Random(int(seed)).shuffle(shuffled)
    first_count = min(max(int(np.floor(len(shuffled) * float(first_ratio))), 1), len(shuffled) - 1)
    return set(shuffled[:first_count]), set(shuffled[first_count:])


def assign_model_split(
    manifest: pd.DataFrame,
    *,
    development_ratio: float = DEVELOPMENT_RATIO,
    train_ratio: float = MODEL_TRAIN_RATIO_WITHIN_DEVELOPMENT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """`dataset_id`列を持つmanifestに`model_split`列を追加して返す。

    2段階:
      1. 全dataset_idを development(=train+validation) / inference に development_ratio で分割
      2. development側をさらに model_train / model_validation に train_ratio で分割

    どちらの段階も dataset_id のリストをシャッフルするだけで、record数の重みは
    一切考慮しない(SPEC.md 2.3 チェック項目1: セグメントは必ず単一splitに属する)。
    """
    manifest = manifest.copy()
    development_ids, inference_ids = _split_groups(manifest["dataset_id"], development_ratio, seed)
    model_train_ids, validation_ids = _split_groups(development_ids, train_ratio, int(seed) + 1)

    manifest["outer_split"] = np.where(manifest["dataset_id"].astype(str).isin(development_ids), "development", "inference")
    manifest["model_split"] = np.select(
        [
            manifest["dataset_id"].astype(str).isin(model_train_ids),
            manifest["dataset_id"].astype(str).isin(validation_ids),
        ],
        ["model_train", "model_validation"],
        default="inference",
    )
    return manifest


def split_summary(manifest: pd.DataFrame) -> dict:
    counts = manifest["model_split"].value_counts().to_dict()
    dataset_counts = manifest.drop_duplicates("dataset_id")["model_split"].value_counts().to_dict()
    return {
        "record_count": {key: int(value) for key, value in counts.items()},
        "dataset_count": {key: int(value) for key, value in dataset_counts.items()},
    }
