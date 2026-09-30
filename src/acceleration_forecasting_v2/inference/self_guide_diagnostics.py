"""自己ガイド検証(SELF_GUIDE_CHECK.md)。

既に学習済みのcheckpointに対し、**再学習なしで**guideの入力を「正しいまま(normal)」
「入れ替え(shuffled、別dataset_idの他レコードのguide)」「無効化(disabled、
guide_mask/retrieval_maskをゼロ)」「自分自身の正解に差し替え(self_target、
理想的な参照系列を与えたら今の実運用モデルはどこまで改善するか)」の4通りに変えて
出力を比較し、学習済みモデルが実際にguideの情報を予測に反映しているかを診断する。

`self_target`は`datasets.self_reference.build_self_reference_dataset`(guideを
自分自身の正解に固定して**再学習する**上限性能測定)とは異なり、**再学習は一切行わず**、
実際の検索結果で学習済みの既存モデルに、推論時だけ理想的なguideを与える。
「モデルはguideを使う能力自体はあるが、実際の検索結果ではその能力を引き出せていない」
という仮説に対し、「能力を引き出す情報さえ与えれば、今のモデルのままでも改善するか」を
確認する(=検索側ではなく生成側の改善余地を切り分けるための診断)。

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


def _build_self_target_batch(dataset, batch, current_values):
    """既存のbatch(実際の検索結果によるguide)から、guideを**そのレコード自身の正解**に
    差し替えたbatchを作る(再学習なし、推論時のみ)。

    `datasets.self_reference._self_reference_arrays`と同じ規約: TOP_K=3スロットのうち
    **1スロットだけ**に自分自身の正解を入れ、残り2スロットは無効(mask=0)のままにする。

    guide_valuesはcondition_norm(history/guideと同じ正規化統計)で正規化する必要がある
    ——targetはtarget_normで正規化されているため、一度物理値に戻してから
    condition_normで正規化し直す(単純にtargetをそのまま流用できない)。
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


_BLOCK_LABELS = (
    "enc12-1", "enc12-2", "enc6-1", "enc6-2", "mid3-1", "mid3-2",
    "dec6-1", "dec6-2", "dec12-1", "dec12-2",
)


def _per_block_diagnostic_stats(model) -> list[dict]:
    """`model.diagnostic_stats()`と同じ計算(reshape→sum(dim=-1).mean(dim=(0,1,2)))を、
    ブロック間で平均せずブロックごとに返す(U-Netのどの深さでnormal/self_targetの反応が
    違うかを見るための内訳)。直前1回のforward呼び出し分の状態を読むだけなので、次のforwardで
    上書きされる前に呼ぶこと。`reference_blocks`を持たないモデル(テスト用ダミーモデル等)には
    `[]`を返す。
    """
    blocks = getattr(model, "reference_blocks", None)
    if not blocks:
        return []
    labels = _BLOCK_LABELS if len(blocks) == len(_BLOCK_LABELS) else [f"block-{index}" for index in range(len(blocks))]
    stats = []
    for label, block in zip(labels, blocks):
        attention = block.reference_attention
        context_norm = float(attention.last_context_norm) if attention.last_context_norm is not None else float("nan")
        if attention.last_attention is not None:
            weights = attention.last_attention.reshape(
                attention.last_attention.shape[0], attention.last_attention.shape[1],
                attention.last_attention.shape[2], 3, 12,
            ).sum(dim=-1).mean(dim=(0, 1, 2))
            rank = [float(value) for value in weights.tolist()]
        else:
            rank = [float("nan")] * 3
        stats.append({
            "block": label, "context_norm": context_norm,
            "attention_rank_1": rank[0], "attention_rank_2": rank[1], "attention_rank_3": rank[2],
        })
    return stats


class _BlockStatsAccumulator:
    """DataLoaderの複数バッチにまたがる、バッチサイズ加重平均のブロック別統計量。

    各バッチの`_per_block_diagnostic_stats`の戻り値(バッチ内で既に平均済みのスカラー)を
    `weight=batch_size`で加重平均することで、split全体での真の(レコード単位の)平均になる。
    """

    _METRIC_KEYS = ("context_norm", "attention_rank_1", "attention_rank_2", "attention_rank_3")

    def __init__(self):
        self._order: list[str] = []
        self._weighted_sums: dict[str, dict[str, float]] = {}
        self._weights: dict[str, float] = {}

    def add(self, per_block_stats: list[dict], weight: int) -> None:
        for entry in per_block_stats:
            label = entry["block"]
            if label not in self._weights:
                self._order.append(label)
                self._weighted_sums[label] = {key: 0.0 for key in self._METRIC_KEYS}
                self._weights[label] = 0.0
            self._weights[label] += weight
            for key in self._METRIC_KEYS:
                self._weighted_sums[label][key] += entry[key] * weight

    def result(self) -> list[dict]:
        rows = []
        for label in self._order:
            total_weight = self._weights[label]
            row = {"block": label}
            for key in self._METRIC_KEYS:
                row[key] = self._weighted_sums[label][key] / total_weight if total_weight else float("nan")
            rows.append(row)
        return rows


def _combine_block_breakdown(normal_rows: list[dict], self_target_rows: list[dict]) -> list[dict]:
    """normal/self_targetのブロック別統計を、blockをキーに横持ち(1ブロック1行)へ結合する。"""
    if not normal_rows or not self_target_rows:
        return []
    self_target_by_block = {row["block"]: row for row in self_target_rows}
    combined = []
    for row in normal_rows:
        label = row["block"]
        partner = self_target_by_block.get(label, {})
        combined.append({
            "block": label,
            "normal_context_norm": row["context_norm"],
            "self_target_context_norm": partner.get("context_norm", float("nan")),
            "normal_attention_rank_1": row["attention_rank_1"],
            "normal_attention_rank_2": row["attention_rank_2"],
            "normal_attention_rank_3": row["attention_rank_3"],
            "self_target_attention_rank_1": partner.get("attention_rank_1", float("nan")),
            "self_target_attention_rank_2": partner.get("attention_rank_2", float("nan")),
            "self_target_attention_rank_3": partner.get("attention_rank_3", float("nan")),
        })
    return combined


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
    current_values = metadata["current_acc_z_max"].to_numpy(dtype=np.float32)
    rows = []
    normal_block_accumulator = _BlockStatsAccumulator()
    self_target_block_accumulator = _BlockStatsAccumulator()
    for batch in loader:
        batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
        positions = batch["index"].cpu().numpy()
        noise = torch.from_numpy(noise_table[positions]).to(device)
        target, mask = batch["target"], batch["target_mask"]
        timesteps = torch.full((target.shape[0],), int(timestep), device=device, dtype=torch.long)
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise

        shuffled_batch = _apply_shuffled_guides(batch)
        disabled_batch = _disable_guide(batch)
        self_target_batch = _build_self_target_batch(dataset, batch, current_values[positions])

        with torch.inference_mode():
            output_normal = model(noisy, timesteps, batch)
            # attentionの内部状態はforward直後(次のforwardで上書きされる前)にのみ読む。
            attention_ranks = _per_record_attention_ranks(model)
            reference_context_norm = _reference_context_norm(model)
            normal_block_accumulator.add(_per_block_diagnostic_stats(model), weight=target.shape[0])
            clean_normal, _ = process.model_output_to_x0_epsilon(noisy, output_normal, alpha)

            output_self_target = model(noisy, timesteps, self_target_batch)
            self_target_attention_ranks = _per_record_attention_ranks(model)
            self_target_block_accumulator.add(_per_block_diagnostic_stats(model), weight=target.shape[0])
            clean_self_target, _ = process.model_output_to_x0_epsilon(noisy, output_self_target, alpha)

            output_shuffled = model(noisy, timesteps, shuffled_batch)
            clean_shuffled, _ = process.model_output_to_x0_epsilon(noisy, output_shuffled, alpha)

            output_disabled = model(noisy, timesteps, disabled_batch)
            clean_disabled, _ = process.model_output_to_x0_epsilon(noisy, output_disabled, alpha)

        normal_mae = _physical_mae(dataset, clean_normal, target, mask)
        self_target_mae = _physical_mae(dataset, clean_self_target, target, mask)
        shuffled_mae = _physical_mae(dataset, clean_shuffled, target, mask)
        disabled_mae = _physical_mae(dataset, clean_disabled, target, mask)
        l1_shuffled = _masked_output_l1(output_normal, output_shuffled, mask)
        l1_disabled = _masked_output_l1(output_normal, output_disabled, mask)
        l1_self_target = _masked_output_l1(output_normal, output_self_target, mask)
        ranks = (attention_ranks.detach().cpu().numpy() if attention_ranks is not None
                 else np.full((target.shape[0], 3), np.nan))
        self_target_ranks = (self_target_attention_ranks.detach().cpu().numpy()
                             if self_target_attention_ranks is not None else np.full((target.shape[0], 3), np.nan))

        trend_ids = metadata["trend_id"].to_numpy()[positions]
        dataset_ids = metadata["dataset_id"].to_numpy()[positions]
        for row in range(target.shape[0]):
            rows.append({
                "trend_id": trend_ids[row], "dataset_id": dataset_ids[row],
                "normal_MAE": float(normal_mae[row]), "shuffled_MAE": float(shuffled_mae[row]),
                "disabled_MAE": float(disabled_mae[row]), "self_target_MAE": float(self_target_mae[row]),
                "guide_advantage_vs_shuffled": float(shuffled_mae[row] - normal_mae[row]),
                "guide_advantage_vs_disabled": float(disabled_mae[row] - normal_mae[row]),
                # 正: 完璧なguide(自分自身の正解)を与えると実際のguideより改善する。
                "self_target_improvement": float(normal_mae[row] - self_target_mae[row]),
                "normal_vs_shuffled_output_L1": float(l1_shuffled[row]),
                "normal_vs_disabled_output_L1": float(l1_disabled[row]),
                "normal_vs_self_target_output_L1": float(l1_self_target[row]),
                "attention_rank_1": float(ranks[row, 0]), "attention_rank_2": float(ranks[row, 1]),
                "attention_rank_3": float(ranks[row, 2]),
                "self_target_attention_rank_1": float(self_target_ranks[row, 0]),
                "self_target_attention_rank_2": float(self_target_ranks[row, 1]),
                "self_target_attention_rank_3": float(self_target_ranks[row, 2]),
                "reference_context_norm": float(reference_context_norm),
            })
    block_breakdown = _combine_block_breakdown(normal_block_accumulator.result(), self_target_block_accumulator.result())
    return pd.DataFrame(rows), block_breakdown


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
        "self_target_MAE_mean": float(frame["self_target_MAE"].mean()),
        "guide_advantage_vs_shuffled": _percentile_stats(frame["guide_advantage_vs_shuffled"].to_numpy()),
        "guide_advantage_vs_disabled": _percentile_stats(frame["guide_advantage_vs_disabled"].to_numpy()),
        "self_target_improvement": _percentile_stats(frame["self_target_improvement"].to_numpy()),
        "normal_vs_shuffled_output_L1_mean": float(frame["normal_vs_shuffled_output_L1"].mean()),
        "normal_vs_disabled_output_L1_mean": float(frame["normal_vs_disabled_output_L1"].mean()),
        "normal_vs_self_target_output_L1_mean": float(frame["normal_vs_self_target_output_L1"].mean()),
        "reference_context_norm_mean": float(frame["reference_context_norm"].mean()),
        "attention_rank_mean": [float(frame[f"attention_rank_{index}"].mean()) for index in (1, 2, 3)],
        "self_target_attention_rank_mean": [
            float(frame[f"self_target_attention_rank_{index}"].mean()) for index in (1, 2, 3)
        ],
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
    block_breakdown_paths = {}
    for timestep in timesteps:
        frame, block_breakdown = _run_one_timestep(process, loader, dataset, metadata, noise_table, int(timestep), device)
        frame.to_csv(output_dir / f"condition_usage_per_target_t{int(timestep)}.csv", index=False, encoding="utf-8-sig")
        summary = _summarize(frame, timestep, config)
        (output_dir / f"condition_usage_summary_t{int(timestep)}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results[str(int(timestep))] = summary

        if block_breakdown:
            block_path = output_dir / f"block_breakdown_t{int(timestep)}.csv"
            pd.DataFrame(block_breakdown).to_csv(block_path, index=False, encoding="utf-8-sig")
            block_breakdown_paths[str(int(timestep))] = str(block_path)
        else:
            block_breakdown_paths[str(int(timestep))] = None

    return {
        "split": split, "record_count": int(len(dataset)), "timesteps": [int(value) for value in timesteps],
        "prediction_type": config.get("prediction_type"), "results": results,
        "block_breakdown": block_breakdown_paths,
    }


def diagnose_guide_conditioning(dataset_dir, checkpoint, output_dir, *, split="model_validation",
                                 timesteps=(900,), device=None, seed=42, plot=False, batch_size=64) -> dict:
    """学習済みcheckpointに対し、guide入力の利用度を診断する(SELF_GUIDE_CHECK.md)。"""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    process, config = load_process(checkpoint, device, use_ema=True)
    return _diagnose_with_process(process, config, dataset_dir, output_dir, split=split,
                                  timesteps=timesteps, seed=seed, plot=plot, batch_size=batch_size)
