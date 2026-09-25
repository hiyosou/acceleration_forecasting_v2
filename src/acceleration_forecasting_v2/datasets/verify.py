"""構築済みデータセットのリーク検査(SPEC.md 2.3のチェックリストを、実データにも使える形にしたもの)。

`prepare_datasets`の出力ディレクトリを読み、規則違反を全て列挙して返す。違反が空なら問題なし。
テスト(合成データ)と、実データでの実行(ステップ14)の両方で同じ検査を使う。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .normalization import Normalization
from .prepare import GUIDE_SOURCE_SPLITS, SPLITS


def verify_dataset_leakage(dataset_dir, manifest: pd.DataFrame | None = None) -> list[str]:
    """違反の説明文のリストを返す(空ならリークなし)。

    検査項目:
      1. 同じdataset_idが複数のsplitに存在しない
      2. 各guideのmodel_splitが、そのsplitに許可された集合(GUIDE_SOURCE_SPLITS)に含まれる
      3. guideが自分と同じdataset_idから来ていない
      4. segment_weightがsplit内でdataset_idごとに合計1になる
      5. 正規化統計(target)が、model_trainの有効な正解値のみから計算されている
      6. manifestを渡した場合、guideの(trend_id → dataset_id/model_split)がmanifestと一致する
    """
    dataset_dir = Path(dataset_dir)
    violations: list[str] = []
    metadata = {split: pd.read_csv(dataset_dir / split / "metadata.csv", encoding="utf-8-sig")
                for split in SPLITS if (dataset_dir / split / "metadata.csv").is_file()}

    seen: dict[str, str] = {}
    for split, frame in metadata.items():
        for dataset_id in frame["dataset_id"].astype(str).unique():
            if dataset_id in seen and seen[dataset_id] != split:
                violations.append(f"dataset_id {dataset_id} が {seen[dataset_id]} と {split} の両方に存在する")
            seen.setdefault(dataset_id, split)

    by_trend = None
    if manifest is not None:
        by_trend = manifest.drop_duplicates("trend_id").set_index(manifest["trend_id"].astype(str).drop_duplicates())
    for split, frame in metadata.items():
        allowed = GUIDE_SOURCE_SPLITS[split]
        for _, row in frame.iterrows():
            trend_ids = json.loads(row["guide_trend_ids"])
            dataset_ids = json.loads(row["guide_dataset_ids"])
            splits = json.loads(row["guide_model_splits"])
            for trend_id, guide_dataset, guide_split in zip(trend_ids, dataset_ids, splits):
                if guide_split not in allowed:
                    violations.append(f"{split}/{row['trend_id']}: 許可されていないsplit({guide_split})のguide {trend_id}")
                if guide_dataset == str(row["dataset_id"]):
                    violations.append(f"{split}/{row['trend_id']}: 同一dataset_idのguide {trend_id}")
                if by_trend is not None and trend_id in by_trend.index:
                    record = by_trend.loc[trend_id]
                    if str(record["dataset_id"]) != guide_dataset or str(record["model_split"]) != guide_split:
                        violations.append(f"{split}/{row['trend_id']}: guide {trend_id} の出所がmanifestと不一致")
        totals = frame.groupby("dataset_id")["segment_weight"].sum()
        for dataset_id, total in totals.items():
            if not np.isclose(total, 1.0, atol=1e-5):
                violations.append(f"{split}/{dataset_id}: segment_weightの合計が1でない({total:.6f})")

    if (dataset_dir / "model_train").is_dir():
        values = np.load(dataset_dir / "model_train" / "target_values.npy")
        masks = np.load(dataset_dir / "model_train" / "target_masks.npy")
        train_values = values[masks > 0]
        norm = Normalization.load(dataset_dir / "target_normalization.json")
        if norm.count != train_values.size or not np.isclose(norm.mean, float(train_values.mean()), rtol=1e-4):
            violations.append("target正規化統計がmodel_trainの有効な正解値のみから計算されていない")
    return violations
