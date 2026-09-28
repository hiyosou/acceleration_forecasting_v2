"""自己ガイド検証(SELF_GUIDE_CHECK.md)。

既に学習済みのcheckpointに対し、**再学習なしで**guideの入力を「正しいまま(normal)」
「入れ替え(shuffled、別dataset_idの他レコードのguide)」「無効化(disabled、
guide_mask/retrieval_maskをゼロ)」の3通りに変えて出力を比較し、
学習済みモデルが実際にguideの情報を予測に反映しているかを診断する。

`SELF_RETRIEVAL_CHECK.md`(検索の配管の検証)とは対象が異なる: こちらは
「学習済みモデルがguide入力を使っているか」というモデル挙動の検証。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from .predict import load_process

_GUIDE_FIELDS = ("guide_values", "guide_deltas", "guide_mask", "guide_similarities", "retrieval_mask")


def _record_seed(seed, trend_id):
    """(seed, trend_id)から決定的な64bit整数seedを作る(バッチ構成やbatch_sizeに依存しない)。"""
    digest = hashlib.sha256(f"{int(seed)}|{trend_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def compute_shuffled_partner_indices(dataset_ids):
    """各レコードiについて、**別のdataset_idを持つ**レコードのインデックスを決定的に選ぶ。

    旧実装の「次のレコード」は同一セグメント内の隣接anchor(月違い、似た値になりやすい)を
    掴む恐れがあるため、dataset_idが異なるものが見つかるまでインデックスを+1ずつずらす
    (循環)。metadata.csvはdataset_idごとに連続してまとまっているため、実際にずらす幅は
    通常小さい。
    """
    dataset_ids = [str(value) for value in dataset_ids]
    n = len(dataset_ids)
    if len(set(dataset_ids)) < 2:
        raise ValueError("shuffled partnerには2件以上の異なるdataset_idが必要です。")
    partners = np.empty(n, dtype=np.int64)
    for i in range(n):
        offset = 1
        while dataset_ids[(i + offset) % n] == dataset_ids[i]:
            offset += 1
            if offset > n:
                raise ValueError("異なるdataset_idを持つ相手が見つかりません。")
        partners[i] = (i + offset) % n
    return partners


class _PairedForecastDataset(Dataset):
    """`ForecastDatasetV2`をラップし、各レコードに「shuffled相手」のguide系フィールドを
    `shuffled_*`キーで併せて返す。2つのDataLoaderを並走させる同期ずれを避けるための設計。
    """

    def __init__(self, dataset: ForecastDatasetV2, partner_indices):
        self.dataset = dataset
        self.partner_indices = np.asarray(partner_indices, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        partner = self.dataset[int(self.partner_indices[index])]
        for field in _GUIDE_FIELDS:
            item[f"shuffled_{field}"] = partner[field]
        return item


def _disable_guide(batch):
    """`guide_mask`/`retrieval_mask`をゼロにしたコピーを返す(attentionで有効なguideトークンが無い状態)。"""
    disabled = dict(batch)
    disabled["guide_mask"] = torch.zeros_like(batch["guide_mask"])
    disabled["retrieval_mask"] = torch.zeros_like(batch["retrieval_mask"])
    return disabled


def _apply_shuffled_guides(batch):
    shuffled = dict(batch)
    for field in _GUIDE_FIELDS:
        shuffled[field] = batch[f"shuffled_{field}"]
    return shuffled


def _per_record_attention_ranks(model):
    """`model.reference_blocks`の`last_attention`(batch,heads,length,36)から、
    `diagnostic_stats()`のバッチ次元までの平均化を経る前の**レコードごとの真の値**を計算する。

    直前1回のforward呼び出し分の状態を読むだけなので、次のforwardで上書きされる前
    (normalフォワードの直後)に呼ぶこと。`reference_blocks`/`last_attention`を持たない
    モデル(テスト用ダミーモデル等)にはNoneを返す。
    """
    blocks = getattr(model, "reference_blocks", None)
    if not blocks:
        return None
    per_block = []
    for block in blocks:
        attention = block.reference_attention.last_attention
        if attention is None:
            continue
        batch, heads, length, _ = attention.shape
        weights = attention.reshape(batch, heads, length, 3, 12).sum(dim=-1).mean(dim=(1, 2))
        per_block.append(weights)
    if not per_block:
        return None
    return torch.stack(per_block, dim=0).mean(dim=0)  # (batch, 3)


def _reference_context_norm(model):
    """`reference_context_norm`はバッチ単位の参考値として`diagnostic_stats()`をそのまま使う。"""
    stats = getattr(model, "diagnostic_stats", None)
    if stats is None:
        return float("nan")
    return float(stats()["reference_context_norm"])


def _physical_mae(dataset, predicted_clean_normalized, target_normalized, mask):
    physical_prediction = dataset.physical_prediction(
        predicted_clean_normalized.detach().cpu().numpy(), bounds=(PHYSICAL_MIN, PHYSICAL_MAX)
    )
    physical_target = dataset.denormalize_target(target_normalized.detach().cpu().numpy())
    mask_np = mask.detach().cpu().numpy()
    diff = np.abs(physical_prediction - physical_target) * mask_np
    denominator = np.clip(mask_np.sum(axis=1), 1, None)
    return diff.sum(axis=1) / denominator


def _masked_output_l1(output_a, output_b, mask):
    diff = (output_a - output_b).abs().detach().cpu().numpy() * mask.detach().cpu().numpy()
    denominator = np.clip(mask.detach().cpu().numpy().sum(axis=1), 1, None)
    return diff.sum(axis=1) / denominator


def _run_one_timestep(process, loader, dataset, metadata, noise_table, timestep, device):
    model = process.model
    alpha = process.alpha_bars[int(timestep)]
    rows = []
    for batch in loader:
        batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
        positions = batch["index"].cpu().numpy()
        noise = torch.from_numpy(noise_table[positions]).to(device)
        target, mask = batch["target"], batch["target_mask"]
        timesteps = torch.full((target.shape[0],), int(timestep), device=device, dtype=torch.long)
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise

        shuffled_batch = _apply_shuffled_guides(batch)
        disabled_batch = _disable_guide(batch)

        with torch.inference_mode():
            output_normal = model(noisy, timesteps, batch)
            # attentionの内部状態はnormalフォワード直後(次のforwardで上書きされる前)にのみ読む。
            attention_ranks = _per_record_attention_ranks(model)
            reference_context_norm = _reference_context_norm(model)
            clean_normal, _ = process.model_output_to_x0_epsilon(noisy, output_normal, alpha)

            output_shuffled = model(noisy, timesteps, shuffled_batch)
            clean_shuffled, _ = process.model_output_to_x0_epsilon(noisy, output_shuffled, alpha)

            output_disabled = model(noisy, timesteps, disabled_batch)
            clean_disabled, _ = process.model_output_to_x0_epsilon(noisy, output_disabled, alpha)

        normal_mae = _physical_mae(dataset, clean_normal, target, mask)
        shuffled_mae = _physical_mae(dataset, clean_shuffled, target, mask)
        disabled_mae = _physical_mae(dataset, clean_disabled, target, mask)
        l1_shuffled = _masked_output_l1(output_normal, output_shuffled, mask)
        l1_disabled = _masked_output_l1(output_normal, output_disabled, mask)
        ranks = (attention_ranks.detach().cpu().numpy() if attention_ranks is not None
                 else np.full((target.shape[0], 3), np.nan))

        trend_ids = metadata["trend_id"].to_numpy()[positions]
        dataset_ids = metadata["dataset_id"].to_numpy()[positions]
        for row in range(target.shape[0]):
            rows.append({
                "trend_id": trend_ids[row], "dataset_id": dataset_ids[row],
                "normal_MAE": float(normal_mae[row]), "shuffled_MAE": float(shuffled_mae[row]),
                "disabled_MAE": float(disabled_mae[row]),
                "guide_advantage_vs_shuffled": float(shuffled_mae[row] - normal_mae[row]),
                "guide_advantage_vs_disabled": float(disabled_mae[row] - normal_mae[row]),
                "normal_vs_shuffled_output_L1": float(l1_shuffled[row]),
                "normal_vs_disabled_output_L1": float(l1_disabled[row]),
                "attention_rank_1": float(ranks[row, 0]), "attention_rank_2": float(ranks[row, 1]),
                "attention_rank_3": float(ranks[row, 2]),
                "reference_context_norm": float(reference_context_norm),
            })
    return pd.DataFrame(rows)


def _percentile_stats(values):
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
    }


def _summarize(frame, timestep, config):
    return {
        "timestep": int(timestep), "record_count": int(len(frame)),
        "prediction_type": config.get("prediction_type"),
        "normal_MAE_mean": float(frame["normal_MAE"].mean()),
        "shuffled_MAE_mean": float(frame["shuffled_MAE"].mean()),
        "disabled_MAE_mean": float(frame["disabled_MAE"].mean()),
        "guide_advantage_vs_shuffled": _percentile_stats(frame["guide_advantage_vs_shuffled"].to_numpy()),
        "guide_advantage_vs_disabled": _percentile_stats(frame["guide_advantage_vs_disabled"].to_numpy()),
        "normal_vs_shuffled_output_L1_mean": float(frame["normal_vs_shuffled_output_L1"].mean()),
        "normal_vs_disabled_output_L1_mean": float(frame["normal_vs_disabled_output_L1"].mean()),
        "reference_context_norm_mean": float(frame["reference_context_norm"].mean()),
        "attention_rank_mean": [float(frame[f"attention_rank_{index}"].mean()) for index in (1, 2, 3)],
    }


def _diagnose_with_process(process, config, dataset_dir, output_dir, *, split="model_validation",
                            timesteps=(900,), seed=42, plot=False, batch_size=64) -> dict:
    """checkpoint読み込みと分離した本体。テストでダミーモデルを注入した`process`を直接渡せる。

    `plot`は未実装のno-op(SELF_GUIDE_CHECK.md Q5: 画像は生成せず数値のみ報告する)。
    """
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    process.model.eval()  # 診断はdropoutの揺らぎを含めない(決定性のため)。
    # `alpha_bars`は`DiffusionProcess`が必ず持つテンソルなので、パラメータを持たない
    # テスト用ダミーモデルでもここでdeviceを取得できる。
    device = process.alpha_bars.device

    dataset = ForecastDatasetV2(dataset_dir / split, dataset_dir, include_targets=True)
    metadata = pd.read_csv(dataset_dir / split / "metadata.csv", encoding="utf-8-sig")
    if len(metadata) != len(dataset):
        raise ValueError("metadata.csvと配列のレコード数が一致しません。")

    partner_indices = compute_shuffled_partner_indices(metadata["dataset_id"].astype(str).tolist())
    paired = _PairedForecastDataset(dataset, partner_indices)
    loader = DataLoader(paired, batch_size=int(batch_size), shuffle=False, num_workers=0)

    trend_ids = metadata["trend_id"].astype(str).tolist()
    noise_table = np.stack([
        np.random.default_rng(_record_seed(seed, trend_id)).standard_normal(12).astype(np.float32)
        for trend_id in trend_ids
    ])

    results = {}
    for timestep in timesteps:
        frame = _run_one_timestep(process, loader, dataset, metadata, noise_table, int(timestep), device)
        frame.to_csv(output_dir / f"condition_usage_per_target_t{int(timestep)}.csv", index=False, encoding="utf-8-sig")
        summary = _summarize(frame, timestep, config)
        (output_dir / f"condition_usage_summary_t{int(timestep)}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results[str(int(timestep))] = summary

    return {
        "split": split, "record_count": int(len(dataset)), "timesteps": [int(value) for value in timesteps],
        "prediction_type": config.get("prediction_type"), "results": results,
    }


def diagnose_guide_conditioning(dataset_dir, checkpoint, output_dir, *, split="model_validation",
                                 timesteps=(900,), device=None, seed=42, plot=False, batch_size=64) -> dict:
    """学習済みcheckpointに対し、guide入力の利用度を診断する(SELF_GUIDE_CHECK.md)。"""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    process, config = load_process(checkpoint, device, use_ema=True)
    return _diagnose_with_process(process, config, dataset_dir, output_dir, split=split,
                                  timesteps=timesteps, seed=seed, plot=plot, batch_size=batch_size)
