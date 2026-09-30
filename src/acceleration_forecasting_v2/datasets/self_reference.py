"""自己参照(guide=自分自身の正解)データセットの構築。

検索によって得られる「類似した時系列」の代わりに、各anchor自身の正解(target_values)を
guideとして与えたとき、モデルがどこまでの精度を出せるか(=生成モジュール側の上限性能)を
測るための特殊用途のデータセット変換。

旧リポジトリの`datasets/self_guide.py`(`diagnostic_mode="self_target_single_guide"`)を
踏襲し、TOP_K=3スロットのうち**1スロットだけ**に自分自身の正解を入れ、残り2スロットは
無効(mask=0, retrieval_mask=0)のままにする(`_single_guide`という名前の通り、
3件を自分自身で埋めるのではなく1件だけ)。

SPEC.md本体のリーク防止規則(guideは同一dataset_idから来てはいけない、
`datasets.verify.verify_dataset_leakage`のチェック項目3)を意図的に破る特殊用途の
データセットであり、`verify_dataset_leakage`はこのデータセットには適用しない
(同一dataset_idからのguideが検出されるのはこのデータセットでは正しい挙動)。

既存の`prepare_datasets`が構築した実データセット(history_values/target_values等)をそのまま
再利用し、guide関連の5配列(guide_values/guide_masks/guide_deltas/guide_similarities/
retrieval_masks)と、参考値のguide_baselines/guide_softmax_weightsだけを差し替える。
history_values/history_masks/segment_weights/target_values/target_masksと、正規化統計
(condition_normalization.json/target_normalization.json)は元のデータセットからそのまま
引き継ぐ(guideの中身だけを変えた条件で比較するため)。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_v2.common.constants import TOP_K
from .build import ARRAY_NAMES

SPLITS = ("model_train", "model_validation", "inference")


def _self_reference_arrays(target_values, target_masks, current, *, top_k=TOP_K):
    """1splitぶんのtarget_values/target_masks/currentから、self-reference用のguide配列一式を作る。

    旧リポジトリの`self_target_single_guide`と同じく、スロット0だけに自分自身の正解を入れ、
    残りのスロットは無効(mask=0)のままにする。
    """
    target_values = np.asarray(target_values, dtype=np.float32)
    target_masks = np.asarray(target_masks, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    n, months = target_values.shape
    valid = (target_masks > 0) & np.isfinite(target_values)

    guide_values = np.zeros((n, top_k, months), dtype=np.float32)
    guide_masks = np.zeros((n, top_k, months), dtype=np.float32)
    guide_deltas = np.zeros((n, top_k, months), dtype=np.float32)
    guide_similarities = np.zeros((n, top_k), dtype=np.float32)
    retrieval_masks = np.zeros((n, top_k), dtype=np.float32)
    guide_softmax_weights = np.zeros((n, top_k, months), dtype=np.float32)

    guide_values[:, 0, :] = np.where(valid, target_values, 0.0)
    guide_masks[:, 0, :] = valid.astype(np.float32)
    guide_deltas[:, 0, :] = np.where(valid, target_values - current[:, None], 0.0)
    guide_similarities[:, 0] = 1.0
    retrieval_masks[:, 0] = 1.0
    guide_softmax_weights[:, 0, :] = valid.astype(np.float32)
    guide_baselines = np.where(valid, target_values, current[:, None]).astype(np.float32)

    return {
        "guide_values": guide_values, "guide_masks": guide_masks, "guide_deltas": guide_deltas,
        "guide_similarities": guide_similarities, "retrieval_masks": retrieval_masks,
        "guide_baselines": guide_baselines, "guide_softmax_weights": guide_softmax_weights,
    }


def build_self_reference_dataset(source_dataset_dir, output_dataset_dir, *, top_k=TOP_K) -> dict:
    """`prepare_datasets`が構築した実データセットから、guideを自分自身の正解に差し替えた
    データセットを新規に書き出す(生成モジュールの上限性能を測るための特殊用途)。

    Returns:
        summary dict(splitごとの件数)。`output_dataset_dir`直下にdataset_summary.jsonも書く。
    """
    source_dataset_dir, output_dataset_dir = Path(source_dataset_dir), Path(output_dataset_dir)
    if source_dataset_dir.resolve() == output_dataset_dir.resolve():
        raise ValueError("出力先は元のデータセットと別ディレクトリにしてください。")
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    for name in ("condition_normalization.json", "target_normalization.json"):
        shutil.copy2(source_dataset_dir / name, output_dataset_dir / name)

    counts = {}
    for split in SPLITS:
        split_dir = source_dataset_dir / split
        if not split_dir.is_dir():
            continue
        out_dir = output_dataset_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        target_values = np.load(split_dir / "target_values.npy")
        target_masks = np.load(split_dir / "target_masks.npy")
        metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
        current = metadata["current_acc_z_max"].to_numpy(dtype=np.float32)

        overrides = _self_reference_arrays(target_values, target_masks, current, top_k=top_k)
        for name in ARRAY_NAMES:
            if name in overrides:
                np.save(out_dir / f"{name}.npy", overrides[name])
            else:
                shutil.copy2(split_dir / f"{name}.npy", out_dir / f"{name}.npy")

        # guideの出所をメタデータ上も実態(=自分自身)に更新する。監査目的のみで、
        # verify_dataset_leakageはこのデータセットには適用しない(同一dataset_idからの
        # guideが検出されるのはこのデータセットでは意図した挙動のため)。
        metadata = metadata.copy()
        metadata["diagnostic_mode"] = "self_target_single_guide"
        metadata["guide_count"] = 1
        metadata["guide_trend_ids"] = metadata["trend_id"].map(lambda value: json.dumps([str(value)]))
        metadata["guide_dataset_ids"] = metadata["dataset_id"].map(lambda value: json.dumps([str(value)]))
        metadata["guide_model_splits"] = json.dumps([split])
        metadata.to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8-sig")
        counts[split] = int(len(metadata))

    summary = {
        "diagnostic_mode": "self_target_single_guide", "top_k": int(top_k),
        "source_dataset": str(source_dataset_dir), "output_dataset": str(output_dataset_dir),
        "counts": counts,
    }
    (output_dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_self_target_batch(dataset, batch, current_values):
    """既存のbatch(実際の検索結果によるguide、またはミニバッチ学習中のbatch)から、
    guideを**そのレコード自身の正解**に差し替えたbatchを作る。

    `_self_reference_arrays`と同じ規約: TOP_K=3スロットのうち**1スロットだけ**に
    自分自身の正解を入れ、残り2スロットは無効(mask=0)のままにする。

    guide_valuesはcondition_norm(history/guideと同じ正規化統計)で正規化する必要がある
    ——targetはtarget_normで正規化されているため、一度物理値に戻してから
    condition_normで正規化し直す(単純にtargetをそのまま流用できない)。

    2つの用途で使う:
    - `inference/self_guide_diagnostics.py`: 再学習なしの推論時診断(self_target条件)。
    - `training/train.py`: 通常学習への補助損失(self_target_loss_weight > 0のとき)。
      `dataset`には学習に使っている`ForecastDatasetV2`インスタンス(`train_data`)を渡す。
    """
    target, mask = batch["target"], batch["target_mask"]
    device = target.device
    physical_target = dataset.denormalize_target(target.detach().cpu().numpy())
    mask_np = mask.detach().cpu().numpy()
    valid = mask_np > 0

    guide_values_physical = np.where(valid, physical_target, 0.0)
    guide_values_normalized = dataset.condition_norm.normalize(guide_values_physical)
    guide_values_normalized = np.nan_to_num(guide_values_normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    delta_physical = np.where(valid, physical_target - np.asarray(current_values, dtype=np.float32)[:, None], 0.0)
    delta_normalized = np.nan_to_num(
        delta_physical / dataset.condition_norm.std, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    batch_size, months = physical_target.shape
    guide_values = np.zeros((batch_size, 3, months), dtype=np.float32)
    guide_masks = np.zeros((batch_size, 3, months), dtype=np.float32)
    guide_deltas = np.zeros((batch_size, 3, months), dtype=np.float32)
    guide_similarities = np.zeros((batch_size, 3), dtype=np.float32)
    retrieval_masks = np.zeros((batch_size, 3), dtype=np.float32)
    guide_values[:, 0] = guide_values_normalized
    guide_masks[:, 0] = valid.astype(np.float32)
    guide_deltas[:, 0] = delta_normalized
    guide_similarities[:, 0] = 1.0
    retrieval_masks[:, 0] = 1.0

    self_target = dict(batch)
    self_target["guide_values"] = torch.from_numpy(guide_values).to(device)
    self_target["guide_mask"] = torch.from_numpy(guide_masks).to(device)
    self_target["guide_deltas"] = torch.from_numpy(guide_deltas).to(device)
    self_target["guide_similarities"] = torch.from_numpy(guide_similarities).to(device)
    self_target["retrieval_mask"] = torch.from_numpy(retrieval_masks).to(device)
    return self_target
