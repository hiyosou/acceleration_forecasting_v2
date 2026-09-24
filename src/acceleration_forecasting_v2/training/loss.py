"""セグメント単位重み付き損失。SPEC.md 1.1の数式をそのまま実装する。

    combined_weight_i = segment_weight_i * snr_weight_i
    batch_loss = Σ_i (combined_weight_i * per_record_loss_i) / Σ_i combined_weight_i

`per_record_loss`は`diffusion.process.DiffusionProcess.per_record_loss`が計算する
(マスク付きMSE、レコード単位)。本クラスは重み付き平均を計算するだけで、拡散の
ノイズスケジュール/prediction_type変換とは独立させている(SPEC 1.1参照)。
"""

from __future__ import annotations

import torch


class SegmentWeightedMaskedLoss:
    """per-record損失をsegment_weight(・snr_weight)で重み付き平均する。"""

    def batch_loss(
        self,
        per_record_loss: torch.Tensor,
        segment_weights: torch.Tensor,
        snr_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SPEC.md 1.1の数式に基づくスカラー損失を返す。

        Args:
            per_record_loss: shape (batch,)。
            segment_weights: shape (batch,)。`1/セグメント内anchor数`
                (`datasets.anchors.compute_segment_weights`が算出したもの)。
            snr_weights: shape (batch,)。本プロジェクトはMin-SNRを使わない
                (SPEC 1.5/1.6)ため通常Noneのままでよい(その場合は全て1.0扱い)。
                将来Min-SNR等のタイムステップ由来の重みを追加する場合の拡張点として残す。
        """
        if per_record_loss.shape != segment_weights.shape:
            raise ValueError("per_record_loss and segment_weights must have the same shape")
        combined_weight = segment_weights if snr_weights is None else segment_weights * snr_weights
        weight_sum = combined_weight.sum()
        if float(weight_sum) <= 0:
            raise ValueError("Sum of combined weights must be positive")
        return (combined_weight * per_record_loss).sum() / weight_sum
