"""学習時の混合guideと残差方式(2026-10-16、手順2)。

手順0(PROGRESS.md 2026-10-15)で、通常の3スロット配置では正解そのものをguideにしても
拡散モデルがほとんど使わないこと、self_target補助損失が「1スロットだけ有効」という配置で
コピーを切り替えるよう学習させていたことが分かった。そこで学習時に、推論時と同じ3スロット配置の
まま、guideを`λ·正解 + (1−λ)·実guide`に置き換える(λはサンプルごとに無作為)。

定義は`datasets/mixed_guide.py`の`mask_mode="guide"`と同じ(テストで一致を確認している):
有効な月は元のguideのマスクのまま、正解が欠測している月は混ぜない(マスクからλを推測できない)。
バッチは`ForecastDatasetV2`の正規化済みテンソル。正規化はアフィンなので、正解を
`condition_norm`の空間に移せばguideと同じ空間で凸結合できる。

残差方式(`target_mode="residual"`): 生成対象を`正解 − 基準線`にする。基準線は
`q + 有効なguideスロットのdeltasの月ごとの平均`(q = 履歴の最後の有効値 = current)。
guideの絶対値ではなく変化量を使うのは、guideの水準のずれを基準線に持ち込まないため。
"""

from __future__ import annotations

import torch


def _current_condition(batch):
    """履歴の最後の有効値(condition_norm空間、shape (B,))。有効な履歴が無い記録は0(=平均)。"""
    masks = batch["history_masks"] > 0
    positions = torch.arange(masks.shape[1], device=masks.device).expand_as(masks)
    last = torch.where(masks, positions, torch.full_like(positions, -1)).max(dim=1).values
    values = batch["history_values"].gather(1, last.clamp_min(0)[:, None]).squeeze(1)
    return torch.where(last >= 0, values, torch.zeros_like(values))


def _target_in_condition_space(batch, condition_norm, target_norm):
    physical = batch["target"].float() * float(target_norm.std) + float(target_norm.mean)
    return (physical - float(condition_norm.mean)) / float(condition_norm.std)


def _valid_guide(batch):
    return (batch["guide_mask"] > 0) & (batch["retrieval_mask"][:, :, None] > 0)


def sample_lambdas(size, *, generator, p_zero=0.5, device=None):
    """確率`p_zero`でλ=0、それ以外はλ~U(0,1)(shape (size,))。"""
    device = device or generator.device
    uniform = torch.rand(size, generator=generator, device=device)
    zero = torch.rand(size, generator=generator, device=device) < float(p_zero)
    return torch.where(zero, torch.zeros_like(uniform), uniform)


def mix_guide_batch(batch, lam, *, condition_norm, target_norm):
    """guideを`λ·正解 + (1−λ)·guide`に置き換えたバッチ(新しいdict)を返す。

    lam: shape (B,)。3スロット共通。マスク・similarities・retrieval_maskは変更しない。
    """
    lam = lam.to(batch["guide_values"].dtype)[:, None, None]
    target_c = _target_in_condition_space(batch, condition_norm, target_norm)[:, None, :]
    current_c = _current_condition(batch)[:, None, None]
    mixed_months = _valid_guide(batch) & (batch["target_mask"][:, None, :] > 0)
    values, deltas = batch["guide_values"], batch["guide_deltas"]
    mixed = dict(batch)
    mixed["guide_values"] = torch.where(mixed_months, lam * target_c + (1 - lam) * values, values)
    mixed["guide_deltas"] = torch.where(mixed_months, lam * (target_c - current_c) + (1 - lam) * deltas, deltas)
    return mixed


def residual_base(batch, *, condition_norm):
    """残差方式の基準線(物理値、shape (B,12))。

    月ごとに有効なスロットのdeltasを平均する。どのスロットも無効な月は、その記録の有効な
    スロット・月全体の平均変化量で埋め、有効なguideが1本も無い記録は変化量0(=現状維持)。
    """
    valid = _valid_guide(batch).to(batch["guide_deltas"].dtype)
    deltas = batch["guide_deltas"].float() * valid
    count = valid.sum(dim=1)
    month_mean = deltas.sum(dim=1) / count.clamp_min(1)
    record_mean = deltas.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
    delta = torch.where(count > 0, month_mean, record_mean[:, None])
    current = _current_condition(batch).float() * float(condition_norm.std) + float(condition_norm.mean)
    return current[:, None] + delta * float(condition_norm.std)


def physical_target(batch, target_norm):
    return batch["target"].float() * float(target_norm.std) + float(target_norm.mean)


def to_residual(batch, residual, *, condition_norm, target_norm):
    """`batch["target"]`を正規化済みの残差`(正解 − 基準線 − mean)/std`に置き換えたバッチを返す。
    正解が欠測している月は0(損失はtarget_maskで除外される)。"""
    base = residual_base(batch, condition_norm=condition_norm)
    value = (physical_target(batch, target_norm) - base - float(residual["mean"])) / float(residual["std"])
    out = dict(batch)
    out["target"] = torch.where(batch["target_mask"] > 0, value, torch.zeros_like(value)).to(batch["target"].dtype)
    return out


def from_residual(values, batch, residual, *, condition_norm):
    """DDIMの出力(正規化済みの残差)を物理値に戻す(shape (B,12))。"""
    base = residual_base(batch, condition_norm=condition_norm)
    return values.float() * float(residual["std"]) + float(residual["mean"]) + base


def residual_clip_bounds(batch, residual, bounds, *, condition_norm):
    """物理値に戻したときに`bounds`に収まる、正規化済み残差の要素ごとの上下限((B,12), (B,12))。"""
    base = residual_base(batch, condition_norm=condition_norm)
    mean, std = float(residual["mean"]), float(residual["std"])
    return (float(bounds[0]) - base - mean) / std, (float(bounds[1]) - base - mean) / std


@torch.no_grad()
def fit_residual_target(loader, *, condition_norm, target_norm, device):
    """model_train(λ=0)の残差`正解 − 基準線`の平均・標準偏差(正解が有効な月だけ)。"""
    total, total_sq, count = 0.0, 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        residual = physical_target(batch, target_norm) - residual_base(batch, condition_norm=condition_norm)
        valid = batch["target_mask"] > 0
        values = residual[valid].double()
        total += float(values.sum())
        total_sq += float(values.square().sum())
        count += int(values.numel())
    if count < 2:
        raise ValueError("残差の正規化統計を計算できる正解がありません。")
    mean = total / count
    std = max((total_sq / count - mean * mean) ** 0.5, 1e-6)
    return {"mean": mean, "std": std}
