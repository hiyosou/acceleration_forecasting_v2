"""混合guide(実guideと正解をλで混ぜたguide)データセットの構築。

生成モジュールの改善を「guideの質」と切り離して判定するための特殊用途のデータセット変換
(2026-10-15、手順0「guideの質 × 生成精度の基準曲線」)。実guideは6か月履歴以上の情報を
持たないため(GENERATIVE_MODULE_NEXT_STEPS.md §0.2)、実guideだけでMAEを測っても
生成モジュールの良し悪しは判定できない。そこで各guideスロットkを

    mixed_k = λ·target + (1−λ)·guide_k

に置き換え、λ=0(実guide)からλ=1(正解そのもの)まで質を段階的に変える。誤差は
(1−λ)·(guide_k − target)になるので、実guideの誤差の性質(水準のずれと形のずれの割合)を
保ったまま量だけが変わる。

- guide_deltasは「混ぜた後の値 − 混ぜた後のcurrent」で計算し直す。guide自身のcurrentは
  `guide_values − guide_deltas`(`datasets/build.py::_build_guide_arrays`の定義より月によらず
  一定)から逆算し、混ぜた後のcurrentは`λ·query_current + (1−λ)·guide_current`とする。
- 有効な月は`mask_mode`で決める。"intersection"(手順0)は「guideと正解の両方に値がある月」だけ
  (λ>0ではguideのマスクに正解の欠測が反映される)。"guide"(手順2、学習時の混合
  `training/guide_mixing.py`と同じ定義)は元のguideのマスクのままで、正解が欠測している月は混ぜない。
- λ=0のときは元の配列を一切変更せずにコピーする(既存の結果を再現できることの検証用)。
- similarities/retrieval_masksは変更しない。retrieval_masks=0のスロットも変更しない。
- `guide_baselines`/`guide_softmax_weights`は`softmax_guide_baseline`で計算し直す。

`datasets.self_reference.build_self_reference_dataset`と同じく、history・target・
segment_weights・正規化統計は元のデータセットからそのままコピーする(再fitしない)。
正解が混ざるのは意図どおりなので、`verify_dataset_leakage`はこのデータセットには適用しない。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import softmax_guide_baseline
from .build import ARRAY_NAMES

SPLITS = ("model_train", "model_validation", "inference")
MASK_MODES = ("intersection", "guide")
GUIDE_ARRAY_NAMES = ("guide_values", "guide_masks", "guide_deltas", "guide_baselines", "guide_softmax_weights")


def _guide_current(values, masks, deltas):
    """1スロット分のguide自身のcurrent(有効月の`values − deltas`)。有効月が無ければNaN。"""
    valid = (masks > 0) & np.isfinite(values) & np.isfinite(deltas)
    if not valid.any():
        return np.nan
    return float(np.median(values[valid] - deltas[valid]))


def _mixed_guide_arrays(guide_values, guide_masks, guide_deltas, similarities, retrieval_masks,
                        target_values, target_masks, current, lam, *, temperature=0.1, mask_mode="intersection"):
    """1split分のguide配列を、実guideと正解をλで混ぜたものに置き換える(物理値のまま)。

    mask_mode:
        "intersection"(手順0): 有効月 = guide∧正解。正解の欠測がマスクに反映される。
        "guide"(手順2、学習と同じ定義): 有効月は元のguideのまま。正解が欠測している月は
        混ぜずにguideの値・deltasをそのまま残す(その月だけλ=0扱い)。マスクからλが推測できない。

    Returns:
        `GUIDE_ARRAY_NAMES`の5配列のdict。λ=0のときはguide_values/guide_masks/guide_deltasの
        コピーだけを返す(guide_baselines/guide_softmax_weightsは呼び出し側で元の配列をコピーする)。
    """
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lamは0〜1の範囲で指定してください: {lam}")
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_modeは{MASK_MODES}のいずれか: {mask_mode!r}")
    guide_values = np.asarray(guide_values, np.float32)
    guide_masks = np.asarray(guide_masks, np.float32)
    guide_deltas = np.asarray(guide_deltas, np.float32)
    if lam == 0.0:
        return {"guide_values": guide_values.copy(), "guide_masks": guide_masks.copy(),
                "guide_deltas": guide_deltas.copy()}
    similarities = np.asarray(similarities, np.float32)
    retrieval_masks = np.asarray(retrieval_masks, np.float32)
    target_values = np.asarray(target_values, np.float32)
    target_masks = np.asarray(target_masks, np.float32)
    current = np.asarray(current, np.float32)

    values_out, masks_out, deltas_out = guide_values.copy(), guide_masks.copy(), guide_deltas.copy()
    baselines = np.zeros(target_values.shape, np.float32)
    weights = np.zeros(guide_values.shape, np.float32)
    target_valid = (target_masks > 0) & np.isfinite(target_values)
    for index in range(len(guide_values)):
        for slot in range(guide_values.shape[1]):
            if retrieval_masks[index, slot] <= 0:
                continue
            values, masks, deltas = guide_values[index, slot], guide_masks[index, slot], guide_deltas[index, slot]
            guide_current = _guide_current(values, masks, deltas)
            guide_valid = (masks > 0) & np.isfinite(values)
            if not np.isfinite(guide_current):
                guide_valid[:] = False
            mixed_months = guide_valid & target_valid[index]
            valid = mixed_months if mask_mode == "intersection" else guide_valid
            mixed = lam * target_values[index] + (1.0 - lam) * values
            mixed_current = lam * current[index] + (1.0 - lam) * guide_current
            values_out[index, slot] = np.where(mixed_months, mixed, np.where(valid, values, np.nan))
            masks_out[index, slot] = valid.astype(np.float32)
            deltas_out[index, slot] = np.where(mixed_months, mixed - mixed_current, np.where(valid, deltas, np.nan))
        baselines[index], weights[index] = softmax_guide_baseline(
            values_out[index], masks_out[index], similarities[index], retrieval_masks[index], current[index],
            temperature,
        )
    return {"guide_values": values_out, "guide_masks": masks_out, "guide_deltas": deltas_out,
            "guide_baselines": baselines, "guide_softmax_weights": weights}


def build_mixed_guide_dataset(source_dataset_dir, output_dataset_dir, lam, *, temperature=0.1,
                              mask_mode="intersection") -> dict:
    """既存データセットのguideを、実guideと正解をλで混ぜたものに差し替えて新規に書き出す。

    Returns:
        summary dict(λ・splitごとの件数・元/出力先)。`output_dataset_dir`直下にdataset_summary.jsonも書く。
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

        arrays = {name: np.load(split_dir / f"{name}.npy") for name in (
            "guide_values", "guide_masks", "guide_deltas", "guide_similarities", "retrieval_masks",
            "target_values", "target_masks")}
        metadata = pd.read_csv(split_dir / "metadata.csv", encoding="utf-8-sig")
        current = metadata["current_acc_z_max"].to_numpy(dtype=np.float32)
        overrides = _mixed_guide_arrays(
            arrays["guide_values"], arrays["guide_masks"], arrays["guide_deltas"], arrays["guide_similarities"],
            arrays["retrieval_masks"], arrays["target_values"], arrays["target_masks"], current, lam,
            temperature=temperature, mask_mode=mask_mode,
        )
        for name in ARRAY_NAMES:
            if name in overrides:
                np.save(out_dir / f"{name}.npy", overrides[name])
            else:
                shutil.copy2(split_dir / f"{name}.npy", out_dir / f"{name}.npy")

        metadata = metadata.copy()
        metadata["diagnostic_mode"] = "mixed_guide"
        metadata["guide_mix_lambda"] = float(lam)
        metadata["guide_mix_mask_mode"] = mask_mode
        metadata.to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8-sig")
        counts[split] = int(len(metadata))

    summary = {
        "diagnostic_mode": "mixed_guide", "guide_mix_lambda": float(lam), "mask_mode": mask_mode,
        "temperature": float(temperature),
        "source_dataset": str(source_dataset_dir), "output_dataset": str(output_dataset_dir), "counts": counts,
    }
    (output_dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
