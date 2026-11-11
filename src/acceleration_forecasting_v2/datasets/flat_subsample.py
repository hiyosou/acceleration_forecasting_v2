"""学習データの「横ばいの件」を減らしたデータセットを作る(2026-11)。

横ばい = 正解(合成の月を含む全月)の最後の有効3か月の平均 − 予測起点の値 が ±0.2 未満の件。
model_trainでは、各セグメント(dataset_id)の横ばいの件から、乱数の順位の上位 round(r × 件数) 件だけを残す
(横ばいでない件は全て残す)。順位は割合 r によらず同じなので、残る件は r が小さいほど入れ子になる。
model_validationは元のまま(検証a)と、同じ規則で削った`model_validation_flatcut`(検証b)の両方を持つ。
`segment_weights`は元の値のまま切り出し、計算し直さない。正規化のjsonは元のままコピーする(inferenceは作らない)。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FLAT_THRESHOLD = 0.2
RISE_THRESHOLD = 0.5
TRAIN_SPLIT = "model_train"
VALIDATION_SPLIT = "model_validation"
VALIDATION_FLATCUT_SPLIT = "model_validation_flatcut"
SUMMARY_FILE = "flat_subsample_summary.json"
CATEGORIES = ("下降", "横ばい", "小さく上昇", "大きく上昇")


def change_amounts(split_dir, metadata=None) -> np.ndarray:
    """各行の変化量(最後の有効3か月の平均 − current_acc_z_max)。行順はmetadata.csvと同じ。

    有効な月 = target_masks(合成の月を含む)が立っていて値が有限な月。有効な月が無い行はNaN。
    """
    split_dir = Path(split_dir)
    if metadata is None:
        metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
    values = np.load(split_dir / "target_values.npy").astype(float)
    valid = (np.load(split_dir / "target_masks.npy") > 0) & np.isfinite(values)
    current = metadata["current_acc_z_max"].to_numpy(float)
    result = np.full(len(metadata), np.nan)
    for index in range(len(metadata)):
        months = np.flatnonzero(valid[index])[-3:]
        if len(months):
            result[index] = values[index, months].mean() - current[index]
    return result


def categorize(delta) -> np.ndarray:
    """0:下降(<=-0.2) 1:横ばい(|Δ|<0.2) 2:小さく上昇(0.2〜0.5) 3:大きく上昇(>=0.5)。NaNは-1。"""
    delta = np.asarray(delta, dtype=float)
    category = np.full(len(delta), -1)
    finite = np.isfinite(delta)
    category[finite & (delta <= -FLAT_THRESHOLD)] = 0
    category[finite & (np.abs(delta) < FLAT_THRESHOLD)] = 1
    category[finite & (delta >= FLAT_THRESHOLD) & (delta < RISE_THRESHOLD)] = 2
    category[finite & (delta >= RISE_THRESHOLD)] = 3
    return category


def flat_keep_mask(dataset_ids, is_flat, keep_ratio, seed=42) -> np.ndarray:
    """残す行のブール配列。横ばいでない行は全て残し、横ばいの行はセグメントごとに round(r×件数) 件残す。

    順位は乱数`default_rng(seed).random(行数)`で決める(rに依存しない → rが小さい方が大きい方の部分集合)。
    """
    if not 0.0 <= float(keep_ratio) <= 1.0:
        raise ValueError("keep_ratio must be between 0 and 1")
    dataset_ids = np.asarray(dataset_ids).astype(str)
    is_flat = np.asarray(is_flat, dtype=bool)
    keys = np.random.default_rng(int(seed)).random(len(dataset_ids))
    keep = ~is_flat
    for dataset_id in np.unique(dataset_ids[is_flat]):
        rows = np.flatnonzero(is_flat & (dataset_ids == dataset_id))
        quota = int(np.floor(float(keep_ratio) * len(rows) + 0.5))
        keep[rows[np.argsort(keys[rows], kind="stable")[:quota]]] = True
    return keep


def _write_subset(source_split, output_split, keep) -> None:
    """source_splitの全`.npy`とmetadata.csvを、keepの行だけ(元の順序のまま)output_splitに書く。"""
    output_split.mkdir(parents=True, exist_ok=True)
    rows = np.flatnonzero(keep)
    for path in sorted(source_split.glob("*.npy")):
        np.save(output_split / path.name, np.load(path)[rows])
    metadata = pd.read_csv(source_split / "metadata.csv", encoding="utf-8-sig")
    metadata.iloc[rows].to_csv(output_split / "metadata.csv", index=False, encoding="utf-8-sig")


def _split_summary(split_dir, keep, weights) -> dict:
    delta = change_amounts(split_dir)
    category = categorize(delta)
    kept = category[keep]

    def count(values):
        return {name: int((values == index).sum()) for index, name in enumerate(CATEGORIES)}

    flat = category == 1
    return {
        "records_before": int(len(keep)), "records_after": int(keep.sum()),
        "categories_before": count(category), "categories_after": count(kept),
        "flat_fraction_before": float(flat.mean()), "flat_fraction_after": float(flat[keep].sum() / max(keep.sum(), 1)),
        "flat_weight_fraction_before": float(weights[flat].sum() / weights.sum()),
        "flat_weight_fraction_after": float(weights[keep & flat].sum() / weights[keep].sum()),
    }


def subsample_flat(source_dir, output_dir, keep_ratio, seed=42) -> dict:
    """`source_dir`の横ばいの件を減らしたデータセットを`output_dir`に作る(元は変更しない)。"""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} は既にあります。")
    output_dir.mkdir(parents=True)
    for path in sorted(source_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)
    summary = {"keep_ratio": float(keep_ratio), "seed": int(seed), "splits": {}}
    plan = ((TRAIN_SPLIT, TRAIN_SPLIT, keep_ratio), (VALIDATION_SPLIT, VALIDATION_SPLIT, 1.0),
            (VALIDATION_SPLIT, VALIDATION_FLATCUT_SPLIT, keep_ratio))
    for source_name, output_name, ratio in plan:
        source_split = source_dir / source_name
        metadata = pd.read_csv(source_split / "metadata.csv", encoding="utf-8-sig")
        is_flat = categorize(change_amounts(source_split, metadata)) == 1
        keep = flat_keep_mask(metadata["dataset_id"], is_flat, ratio, seed)
        _write_subset(source_split, output_dir / output_name, keep)
        weights = np.load(source_split / "segment_weights.npy").astype(float)
        summary["splits"][output_name] = _split_summary(source_split, keep, weights)
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
