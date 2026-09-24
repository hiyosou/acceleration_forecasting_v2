"""拡散プロセス本体(コサインスケジュール・prediction_type変換・DDIMサンプリング)。

`acceleration_forecasting_12m/.../diffusion/process.py`を出発点に、
以下の点を変更している。

1. Min-SNR重み付けを削除した。SPEC.md 1.5/1.6の決定により、本プロジェクトの
   主軸はepsilon(Min-SNRなし)であり、不要な複雑さを持ち込まない
   (「実験の枝分かれを防ぐ」というRMA一本化の方針と一貫させる)。
2. per-record損失の計算(`per_record_loss`)と、セグメント単位の重み付け平均
   (SPEC 1.1)を責務として分離した。本クラスは前者のみを担当し、後者は
   `training/loss.py`の`SegmentWeightedMaskedLoss`が担当する。
3. classifier-free guidance(CFG)関連の分岐を削除した(SPEC 1.6: target_mode=absolute
   専用構成でRMAはCFGを使わない、旧実装のtrain.pyのコメント
   "Reference-modulated model does not use classifier-free guidance"と整合)。
"""

from __future__ import annotations

import numpy as np
import torch


def cosine_alpha_bars(steps=1000, offset=0.008):
    times = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    values = torch.cos((times + offset) / (1 + offset) * np.pi / 2).square()
    values = values / values[0]
    betas = 1 - values[1:] / values[:-1]
    betas = torch.clamp(betas, 1e-5, 0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class DiffusionProcess:
    """cosineスケジュールの拡散プロセス。prediction_typeはepsilon/v_prediction/x0_predictionに対応。"""

    def __init__(self, model, steps=1000, prediction_type="epsilon"):
        self.model, self.steps = model, int(steps)
        if prediction_type not in {"epsilon", "v_prediction", "x0_prediction"}:
            raise ValueError("prediction_type must be epsilon, v_prediction, or x0_prediction")
        self.prediction_type = prediction_type
        self.alpha_bars = cosine_alpha_bars(steps)

    def to(self, device):
        self.model.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def per_record_loss(self, batch, noise=None, timesteps=None):
        """マスク付きMSEをレコード単位で平均した損失を返す(shape: (batch,))。

        セグメント単位の重み付け(SPEC 1.1)は含まない — 呼び出し側
        (`training/loss.py`)がこれを`segment_weight`と組み合わせて集約する。
        """
        target, mask = batch["target"], batch["target_mask"]
        batch_size = target.shape[0]
        timesteps = timesteps if timesteps is not None else torch.randint(0, self.steps, (batch_size,), device=target.device)
        noise = noise if noise is not None else torch.randn_like(target)
        alpha = self.alpha_bars[timesteps][:, None]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        if self.prediction_type == "epsilon":
            expected = noise
        elif self.prediction_type == "v_prediction":
            expected = alpha.sqrt() * noise - (1 - alpha).sqrt() * target
        else:
            expected = target
        predicted = self.model(noisy, timesteps, batch)
        squared = (predicted - expected).square() * mask
        per_record = squared.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return per_record

    def diagnostics(self, batch, noise=None, timesteps=None):
        """学習曲線の監視用に、正規化空間でのx0 MAE/RMSEなど補助指標も返す。"""
        target, mask = batch["target"], batch["target_mask"]
        batch_size = target.shape[0]
        timesteps = timesteps if timesteps is not None else torch.randint(0, self.steps, (batch_size,), device=target.device)
        noise = noise if noise is not None else torch.randn_like(target)
        alpha = self.alpha_bars[timesteps][:, None]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        predicted = self.model(noisy, timesteps, batch)
        predicted_clean, _ = self.model_output_to_x0_epsilon(noisy, predicted, alpha)
        clean_error = (predicted_clean - target) * mask
        denom = mask.sum().clamp_min(1)
        return {
            "x0_mae_normalized": (clean_error.abs().sum() / denom).item(),
            "x0_rmse_normalized": (clean_error.square().sum() / denom).sqrt().item(),
        }

    def model_output_to_x0_epsilon(self, values, output, alpha):
        if self.prediction_type == "epsilon":
            epsilon = output
            predicted_clean = (values - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
        elif self.prediction_type == "v_prediction":
            predicted_clean = alpha.sqrt() * values - (1 - alpha).sqrt() * output
            epsilon = (1 - alpha).sqrt() * values + alpha.sqrt() * output
        else:
            predicted_clean = output
            epsilon = (values - alpha.sqrt() * predicted_clean) / (1 - alpha).sqrt().clamp_min(1e-12)
        return predicted_clean, epsilon

    @torch.inference_mode()
    def ddim(self, batch, *, shape, sampling_steps=50, eta=0.0, normalized_clip=None, generator=None, initial_noise_scale=1.0):
        device = batch["history_values"].device
        values = float(initial_noise_scale) * torch.randn(shape, device=device, generator=generator)
        if int(sampling_steps) <= 0 or int(sampling_steps) > self.steps:
            raise ValueError("sampling_steps must be between 1 and diffusion steps")
        sequence = np.linspace(self.steps - 1, 0, int(sampling_steps), dtype=int)
        for position, timestep in enumerate(sequence):
            next_timestep = sequence[position + 1] if position + 1 < len(sequence) else -1
            t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            alpha = self.alpha_bars[int(timestep)]
            next_alpha = self.alpha_bars[int(next_timestep)] if next_timestep >= 0 else torch.tensor(1.0, device=device)
            output = self.model(values, t, batch)
            predicted_clean, epsilon = self.model_output_to_x0_epsilon(values, output, alpha)
            if normalized_clip is not None:
                predicted_clean = predicted_clean.clamp(float(normalized_clip[0]), float(normalized_clip[1]))
            sigma = float(eta) * torch.sqrt((1 - next_alpha) / (1 - alpha) * (1 - alpha / next_alpha))
            direction = torch.sqrt(torch.clamp(1 - next_alpha - sigma.square(), min=0)) * epsilon
            random = torch.randn(values.shape, device=device, generator=generator) if next_timestep >= 0 else 0
            values = next_alpha.sqrt() * predicted_clean + direction + sigma * random
        return values
