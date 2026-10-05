"""RMA(Reference-Modulated Attention) U-Net、一本化・設定固定版。

`acceleration_forecasting_12m/.../models/reference_modulated_unet.py`
(`ReferenceModulatedUNet12`)を出発点に、以下をSPEC.md 1.1/1.3/1.6の決定に
従って変更している。

1. 条件エンコーダの入力を「current(1)+history(5)+history_mask(5)=11次元」から
   「history_values(6)+history_masks(6)=12次元」に統一(SPEC 1.1)。
2. アーキテクチャのハイパーパラメータ(attention_mode/fusion_type/チャンネル構成等)を
   SPEC 1.3で実測に基づき確定した値に固定し、デフォルト引数として埋め込む
   (呼び出し側からの上書きは想定しない、実験の枝分かれを防ぐ)。
3. target_mode=absolute専用(SPEC 1.6)のため、classifier-free guidance用の
   `condition_indicator`/`use_reference`トグルは持たない(常にguideを使う)。
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """`models/unet.py`の`TimeEmbedding`を無変更で移植。"""

    def __init__(self, dimension=128):
        super().__init__()
        self.dimension = int(dimension)
        self.network = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension))

    def forward(self, timesteps):
        half = self.dimension // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
        angles = timesteps.float()[:, None] * frequencies[None]
        values = torch.cat([angles.sin(), angles.cos()], dim=1)
        return self.network(values)


class GuideEncoder12(nn.Module):
    """`models/absolute_attention_unet.py`の`GuideEncoder12`を無変更で移植。"""

    def __init__(self, guide_dim=64, month_dim=16, rank_dim=8, include_similarity=True):
        super().__init__()
        self.include_similarity = bool(include_similarity)
        self.month_embedding = nn.Embedding(12, month_dim)
        self.rank_embedding = nn.Embedding(3, rank_dim)
        self.continuous = nn.Sequential(nn.Linear(3 if self.include_similarity else 2, 32), nn.SiLU())
        self.network = nn.Sequential(
            nn.Linear(32 + month_dim + rank_dim, guide_dim), nn.SiLU(),
            nn.Linear(guide_dim, guide_dim),
        )

    def forward(self, values, deltas, similarities):
        batch, guides, months = values.shape
        features = [values, deltas]
        if self.include_similarity:
            features.append(similarities.unsqueeze(-1).expand(-1, -1, months))
        continuous = self.continuous(torch.stack(features, dim=-1))
        month_ids = torch.arange(months, device=values.device).view(1, 1, months)
        rank_ids = torch.arange(guides, device=values.device).view(1, guides, 1)
        month = self.month_embedding(month_ids).expand(batch, guides, -1, -1)
        rank = self.rank_embedding(rank_ids).expand(batch, -1, months, -1)
        return self.network(torch.cat([continuous, month, rank], dim=-1))


class ConditionalBlock(nn.Module):
    """`models/absolute_attention_unet.py`の`ConditionalBlock`(FiLM変調)を無変更で移植。

    `last_scale_norm`/`last_shift_norm`は`ReferenceModulatedAttention`の
    `last_attention`/`last_context_norm`と同じ診断用の副作用(forwardの計算自体には
    影響しない)。直前1回のforward呼び出し分のFiLM出力(scale/shift)の大きさを
    バッチ単位のスカラーとして保持し、guide条件診断(`inference/self_guide_diagnostics.py`)から
    「attentionが集めた文脈が、実際にU-Netへ加える変調(FiLM)としてどれだけの大きさに
    変換されているか」を覗けるようにする。
    """

    def __init__(self, channels, dropout=0.1, condition_dim=256):
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(8, channels), nn.GroupNorm(8, channels)
        self.conv1, self.conv2 = nn.Conv1d(channels, channels, 3, padding=1), nn.Conv1d(channels, channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.last_scale_norm = None
        self.last_shift_norm = None

    def forward(self, values, condition):
        scale, shift = self.film(condition).chunk(2, dim=-1)
        self.last_scale_norm = scale.detach().norm(dim=-1).mean()
        self.last_shift_norm = shift.detach().norm(dim=-1).mean()
        hidden = self.conv1(F.silu(self.norm1(values)))
        hidden = self.norm2(hidden) * (1 + scale[:, :, None]) + shift[:, :, None]
        return values + self.conv2(self.dropout(F.silu(hidden)))


class ReferenceModulatedAttention(nn.Module):
    """`models/reference_modulated_unet.py`の`ReferenceModulatedAttention`を出発点に、
    `ratd_condition_reconstruction`融合に`context_residual`(guide由来のcontextから
    output_conditionへの直接残差接続)を追加している(2026-09-30、診断で判明した
    ボトルネックへの対応)。

    診断(`inference/self_guide_diagnostics.py`のブロック単位・FiLM単位の内訳)により、
    attentionが集めたcontextはブロックによって最大10%前後変化するのに対し、実際の
    FiLM変調(scale/shift、`condition_fusion`の最終出力)はほぼ変化しない(相対変化
    0.02〜1%程度)ことが分かった。既存の共有MLP(`condition_fusion`)は
    `pooled_values`/`condition`(実運用では支配的な信号)と`pooled_context`(guide由来、
    弱い信号)を混ぜて処理するため、学習中にguide由来の変動への感度がほぼ失われたと
    考えられる。`context_residual`は、既存の代替融合方式`residual_delta`
    (`output_condition = condition + self.condition_output(pooled_context)`)が
    既に持つ「直接残差」の性質を、`condition_fusion`の共有MLPは維持したまま追加で
    与えるもの(両方の利点を両立させる狙い)。ゼロ初期化のため、学習開始時点では
    これまでと数値的に完全に同一の挙動になる。
    """

    def __init__(self, channels, condition_dim=256, reference_dim=64, heads=8, head_dim=8,
                 use_similarity_bias=True, attention_mode="global", fusion_type="residual_delta",
                 context_residual_enabled=True):
        super().__init__()
        if attention_mode not in {"global", "month_aligned"}:
            raise ValueError("attention_mode must be global or month_aligned")
        if fusion_type not in {"residual_delta", "ratd_condition_reconstruction"}:
            raise ValueError("Unsupported reference fusion type")
        self.use_similarity_bias = bool(use_similarity_bias)
        self.attention_mode = attention_mode
        self.fusion_type = fusion_type
        # 2026-10-05: 初期チェックポイント(context_residual導入より前に学習したもの)を
        # 現行コードで読み込めるようにするため、フラグ化した(既定Trueで現行動作を維持)。
        # PROGRESS.md参照。
        self.context_residual_enabled = bool(context_residual_enabled) and fusion_type == "ratd_condition_reconstruction"
        self.heads, self.head_dim = int(heads), int(head_dim)
        inner = self.heads * self.head_dim
        self.query = nn.Linear(channels + condition_dim, inner, bias=False)
        self.key = nn.Linear(reference_dim + condition_dim, inner, bias=False)
        self.value = nn.Linear(reference_dim, inner, bias=False)
        self.feature_output = nn.Linear(inner, channels, bias=False)
        self.condition_output = nn.Linear(inner, condition_dim, bias=False)
        if self.fusion_type == "ratd_condition_reconstruction":
            self.condition_fusion = nn.Sequential(
                nn.Linear(channels + condition_dim + inner, condition_dim), nn.SiLU(),
                nn.Linear(condition_dim, condition_dim),
            )
            if self.context_residual_enabled:
                self.context_residual = nn.Linear(inner, condition_dim, bias=False)
                nn.init.zeros_(self.context_residual.weight)
        if self.use_similarity_bias:
            self.similarity_scale = nn.Parameter(torch.tensor(1.0))
        self.last_attention = None
        self.last_context_norm = None

    @staticmethod
    def month_alignment_mask(length, device):
        if length not in {3, 6, 12}:
            raise ValueError(f"Month-aligned RMA only supports lengths 3, 6 and 12, got {length}")
        months_per_position = 12 // length
        reference_month = torch.arange(12, device=device)
        allowed_by_position = []
        for position in range(length):
            start, end = position * months_per_position, (position + 1) * months_per_position
            allowed_by_position.append((reference_month >= start) & (reference_month < end))
        allowed = torch.stack(allowed_by_position, dim=0)
        return allowed.repeat(1, 3).reshape(length, 3, 12).reshape(length, 36)

    def forward(self, values, condition, references, reference_mask, similarities):
        batch, channels, length = values.shape
        tokens = references.reshape(batch, 36, -1)
        valid = reference_mask.reshape(batch, 36) > 0
        query_input = torch.cat([
            values.transpose(1, 2), condition[:, None, :].expand(-1, length, -1)
        ], dim=-1)
        key_input = torch.cat([
            tokens, condition[:, None, :].expand(-1, tokens.shape[1], -1)
        ], dim=-1)
        q = self.query(query_input).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(key_input).reshape(batch, 36, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(tokens).reshape(batch, 36, self.heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if self.use_similarity_bias:
            bias = similarities[:, :, None].expand(-1, -1, 12).reshape(batch, 1, 1, 36)
            scores = scores + self.similarity_scale * bias
        effective_valid = valid[:, None, None, :]
        if self.attention_mode == "month_aligned":
            allowed = self.month_alignment_mask(length, values.device)
            effective_valid = effective_valid & allowed[None, None, :, :]
        scores = scores.masked_fill(~effective_valid, -1e4)
        weights = torch.softmax(scores, dim=-1) * effective_valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        feature_delta = self.feature_output(context).transpose(1, 2)
        pooled_context = context.mean(dim=1)
        if self.fusion_type == "ratd_condition_reconstruction":
            pooled_values = values.mean(dim=-1)
            output_condition = self.condition_fusion(
                torch.cat([pooled_values, condition, pooled_context], dim=-1)
            )
            if self.context_residual_enabled:
                output_condition = output_condition + self.context_residual(pooled_context)
        else:
            output_condition = condition + self.condition_output(pooled_context)
        self.last_attention = weights.detach()
        self.last_context_norm = context.detach().norm(dim=-1).mean()
        return feature_delta, output_condition


class ReferenceModulatedBlock(nn.Module):
    """`models/reference_modulated_unet.py`の`ReferenceModulatedBlock`を無変更で移植。"""

    def __init__(self, channels, dropout=0.1, use_similarity_bias=True,
                 attention_mode="global", fusion_type="residual_delta", heads=8, head_dim=8,
                 condition_dim=256, reference_dim=64, context_residual_enabled=True):
        super().__init__()
        self.reference_attention = ReferenceModulatedAttention(
            channels, use_similarity_bias=use_similarity_bias, attention_mode=attention_mode,
            fusion_type=fusion_type, heads=heads, head_dim=head_dim,
            condition_dim=condition_dim, reference_dim=reference_dim,
            context_residual_enabled=context_residual_enabled,
        )
        self.residual = ConditionalBlock(channels, dropout, condition_dim=condition_dim)

    def forward(self, values, condition, references, reference_mask, similarities):
        feature_delta, reference_condition = self.reference_attention(
            values, condition, references, reference_mask, similarities
        )
        return self.residual(values + feature_delta, reference_condition)


class ReferenceModulatedUNetV2(nn.Module):
    """RMA U-Net。SPEC.md 1.3で実測に基づき確定した構成をデフォルト引数として固定する。

    条件入力は「history_values(6)+history_masks(6)=12次元」に統一済み(SPEC 1.1)。
    target_mode=absolute専用(SPEC 1.6)のため、CFG用の`condition_indicator`は持たない。
    """

    def __init__(
        self,
        dropout: float = 0.1,
        *,
        history_months: int = 6,
        forecast_months: int = 12,
        # 以下、SPEC.md 1.3で確定した固定値。呼び出し側からの上書きは想定しない
        # (`artifacts_prediction_parameterization/normal_guides/A_epsilon_t1000/model/resolved_config.json`
        # の実測構成に一致させている)。
        reference_similarity_enabled: bool = False,
        reference_attention_mode: Literal["global", "month_aligned"] = "month_aligned",
        reference_fusion_type: Literal["residual_delta", "ratd_condition_reconstruction"] = "ratd_condition_reconstruction",
        base_channels: int = 64,
        channel_multipliers: tuple[int, int, int] = (1, 2, 4),
        block_counts: tuple[int, int, int, int, int] = (2, 2, 2, 2, 2),
        condition_dim: int = 256,
        reference_dim: int = 64,
        attention_heads: int = 8,
        attention_head_dim: int = 8,
        # 2026-10-05追加: context_residual(2026-09-30実験)導入より前に学習した
        # checkpointを現行コードで読み込めるようにするためのフラグ(既定Trueで
        # 現行の本番アーキテクチャと同一)。`inference/predict.py::load_process`が
        # checkpointのstate_dictから自動判定して渡す。PROGRESS.md参照。
        context_residual_enabled: bool = True,
    ):
        super().__init__()
        self.history_months = int(history_months)
        self.forecast_months = int(forecast_months)
        self.reference_similarity_enabled = bool(reference_similarity_enabled)
        self.context_residual_enabled = bool(context_residual_enabled)
        self.reference_attention_mode = reference_attention_mode
        self.reference_fusion_type = reference_fusion_type
        self.base_channels = int(base_channels)
        self.attention_heads = int(attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.channel_multipliers = tuple(int(value) for value in channel_multipliers)
        self.block_counts = tuple(int(value) for value in block_counts)
        self.condition_dim = int(condition_dim)
        self.reference_dim = int(reference_dim)
        if len(self.channel_multipliers) != 3 or any(value <= 0 for value in self.channel_multipliers):
            raise ValueError("channel_multipliers must contain three positive integers")
        if len(self.block_counts) != 5 or any(value <= 0 for value in self.block_counts):
            raise ValueError("block_counts must contain five positive integers")
        if self.condition_dim <= 0 or self.condition_dim % 2:
            raise ValueError("condition_dim must be a positive even integer")
        c1, c2, c3 = (self.base_channels * value for value in self.channel_multipliers)
        if any(channels % 8 for channels in (c1, c2, c3)):
            raise ValueError("all U-Net channels must be divisible by 8 for GroupNorm")

        condition_half = self.condition_dim // 2
        # SPEC 1.1: history_values(history_months) + history_masks(history_months)
        self.condition_encoder = nn.Sequential(
            nn.Linear(2 * self.history_months, condition_half), nn.SiLU(), nn.Linear(condition_half, condition_half)
        )
        self.time_encoder = TimeEmbedding(condition_half)
        self.reference_encoder = GuideEncoder12(
            guide_dim=self.reference_dim, include_similarity=self.reference_similarity_enabled
        )

        self.input = nn.Conv1d(1, c1, 3, padding=1)
        block = lambda channels: ReferenceModulatedBlock(
            channels, dropout, use_similarity_bias=self.reference_similarity_enabled,
            attention_mode=self.reference_attention_mode, fusion_type=self.reference_fusion_type,
            heads=self.attention_heads, head_dim=self.attention_head_dim,
            condition_dim=self.condition_dim, reference_dim=self.reference_dim,
            context_residual_enabled=self.context_residual_enabled,
        )
        self.enc12 = nn.ModuleList([block(c1) for _ in range(self.block_counts[0])])
        self.down6 = nn.Conv1d(c1, c2, 3, stride=2, padding=1)
        self.enc6 = nn.ModuleList([block(c2) for _ in range(self.block_counts[1])])
        self.down3 = nn.Conv1d(c2, c3, 3, stride=2, padding=1)
        self.mid3 = nn.ModuleList([block(c3) for _ in range(self.block_counts[2])])
        self.up6 = nn.ConvTranspose1d(c3, c2, 4, stride=2, padding=1)
        self.merge6 = nn.Conv1d(c2 * 2, c2, 1)
        self.dec6 = nn.ModuleList([block(c2) for _ in range(self.block_counts[3])])
        self.up12 = nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1)
        self.merge12 = nn.Conv1d(c1 * 2, c1, 1)
        self.dec12 = nn.ModuleList([block(c1) for _ in range(self.block_counts[4])])
        self.output = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv1d(c1, 1, 3, padding=1))

    @property
    def reference_blocks(self):
        return [*self.enc12, *self.enc6, *self.mid3, *self.dec6, *self.dec12]

    def _run(self, values, blocks, condition, references, reference_mask, similarities):
        for block in blocks:
            values = block(values, condition, references, reference_mask, similarities)
        return values

    def diagnostic_stats(self):
        norms, rank_weights = [], []
        for block in self.reference_blocks:
            attention = block.reference_attention
            if attention.last_context_norm is not None:
                norms.append(float(attention.last_context_norm))
            if attention.last_attention is not None:
                weights = attention.last_attention.reshape(
                    attention.last_attention.shape[0], attention.last_attention.shape[1],
                    attention.last_attention.shape[2], 3, 12
                ).sum(dim=-1).mean(dim=(0, 1, 2))
                rank_weights.append(weights.cpu())
        rank = torch.stack(rank_weights).mean(dim=0).tolist() if rank_weights else [0.0] * 3
        return {"reference_context_norm": sum(norms) / max(len(norms), 1),
                **{f"attention_rank_{index + 1}": float(value) for index, value in enumerate(rank)}}

    def forward(self, noisy, timesteps, batch):
        """
        Args:
            noisy: shape (batch, forecast_months)。
            timesteps: shape (batch,)。
            batch: "history_values"(batch,history_months), "history_masks"(batch,history_months),
                "guide_values"/"guide_deltas"(batch,3,forecast_months), "guide_similarities"(batch,3),
                "guide_mask"/"retrieval_mask" を含む辞書。

        Returns:
            shape (batch, forecast_months)。
        """
        condition_input = torch.cat([batch["history_values"], batch["history_masks"]], dim=-1)
        condition = torch.cat([self.condition_encoder(condition_input), self.time_encoder(timesteps)], dim=-1)
        references = self.reference_encoder(batch["guide_values"], batch["guide_deltas"], batch["guide_similarities"])
        reference_mask = batch["guide_mask"] * batch["retrieval_mask"].unsqueeze(-1)
        similarities = batch["guide_similarities"]

        x12 = self._run(self.input(noisy[:, None]), self.enc12, condition, references, reference_mask, similarities)
        x6 = self._run(self.down6(x12), self.enc6, condition, references, reference_mask, similarities)
        x3 = self._run(self.down3(x6), self.mid3, condition, references, reference_mask, similarities)
        y6 = self._run(self.merge6(torch.cat([self.up6(x3), x6], dim=1)), self.dec6,
                       condition, references, reference_mask, similarities)
        y12 = self._run(self.merge12(torch.cat([self.up12(y6), x12], dim=1)), self.dec12,
                        condition, references, reference_mask, similarities)
        return self.output(y12).squeeze(1)
