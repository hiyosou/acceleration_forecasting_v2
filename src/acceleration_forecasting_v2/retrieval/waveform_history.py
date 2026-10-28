"""6か月履歴の波形埋め込みによるguide検索(2026-10-20、ユーザー提案)。

既存の検索は、波形の埋め込みなら予測開始日の1日分だけ(`GuideIndex.search`)、6か月履歴なら
月最大加速度の値だけ(`GuideIndex.search_by_history`)を使っていた。ここでは6か月分の波形の推移で
比べる:

1. 記録(trend = dataset_id × 計測日)ごとに、その日の走行の埋め込み(L2正規化済み)を平均し、
   L2正規化し直して1本にする。model_train/model_validationはDBの埋め込み、inferenceは
   `waveforms.bin`からその場で埋め込む(`retrieval.pipeline.encode_waveforms`、prepareと同じ)。
2. 各記録の6か月履歴の計測日を`datasets.history.build_history_window`(anchorの履歴・
   `search_by_history`の候補側と同じ選び方)で決め、6か月分の埋め込みの並び(6, dim)を作る。
3. 問い合わせと候補で「kか月前どうし」のコサイン類似度を取り、両方に埋め込みがある月で平均する。
   重なりが`min_history_overlap_months`未満の候補は除外する。
4. 類似度が高い順に`GuideIndex._select`で上位top_k件(eligibility・同日重複の除外は既存と同じ)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import HISTORY_MONTHS
from acceleration_forecasting_v2.datasets.history import build_history_window
from .pipeline import encode_waveforms
from .search import GuideIndex, SearchConfig


def _normalize_rows(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.where(norms > 1e-12, values / np.maximum(norms, 1e-12), 0.0).astype(np.float32)


def trend_embeddings(trend_ids, embeddings):
    """走行ごとの埋め込みを記録(trend_id)ごとに平均してL2正規化する。

    Returns:
        (unique_trend_ids, embeddings (n_trends, dim))。
    """
    trend_ids = np.asarray(trend_ids).astype(str)
    unique, inverse = np.unique(trend_ids, return_inverse=True)
    sums = np.zeros((len(unique), embeddings.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, _normalize_rows(np.asarray(embeddings, dtype=np.float32)))
    return unique, _normalize_rows(sums)


def history_trend_windows(trend_catalog, *, months=HISTORY_MONTHS):
    """各記録の6か月履歴に使う記録のtrend_id(古い月→今月、欠測は"")。

    `trend_catalog`は`trend_id, dataset_id, measurement_date, current_acc_z_max`を持つ。
    同じdataset_id・同じ計測日の記録は1件(trend_catalogの一意性)。
    """
    frame = trend_catalog[["trend_id", "dataset_id", "measurement_date", "current_acc_z_max"]].copy()
    frame["trend_id"] = frame["trend_id"].astype(str)
    frame["measurement_date"] = pd.to_datetime(frame["measurement_date"])
    frame = frame.rename(columns={"current_acc_z_max": "acc_z_max"})
    windows = {}
    for _, group in frame.groupby("dataset_id", sort=False):
        group = group.sort_values("measurement_date", kind="mergesort")
        by_date = dict(zip(group["measurement_date"].dt.strftime("%Y-%m-%d"), group["trend_id"]))
        for trend_id, anchor in zip(group["trend_id"], group["measurement_date"]):
            _, mask, dates = build_history_window(group, anchor, months=months)
            windows[trend_id] = [by_date.get(str(date), "") if mask[k] > 0 else "" for k, date in enumerate(dates)]
    return windows


class WaveformHistoryIndex:
    """`GuideIndex`の候補(走行単位の行)に、6か月履歴の波形埋め込みの並びを対応させた検索器。"""

    def __init__(self, guide_index: GuideIndex, trend_catalog, *, inference_trend_ids=(), inference_embeddings=None,
                 months=HISTORY_MONTHS):
        self.guide_index = guide_index
        ids, embeddings = trend_embeddings(guide_index.trend_ids, guide_index.embeddings)
        if inference_embeddings is not None and len(inference_embeddings):
            extra_ids, extra = trend_embeddings(inference_trend_ids, inference_embeddings)
            keep = ~np.isin(extra_ids, ids)
            ids, embeddings = np.concatenate([ids, extra_ids[keep]]), np.concatenate([embeddings, extra[keep]])
        lookup = {trend_id: position for position, trend_id in enumerate(ids)}
        windows = history_trend_windows(trend_catalog, months=months)
        self.window_lookup = {}
        window_values = np.zeros((len(windows), months, embeddings.shape[1]), dtype=np.float32)
        window_masks = np.zeros((len(windows), months), dtype=bool)
        for row, (trend_id, members) in enumerate(windows.items()):
            self.window_lookup[trend_id] = row
            for k, member in enumerate(members):
                position = lookup.get(member)
                if position is not None:
                    window_values[row, k] = embeddings[position]
                    window_masks[row, k] = True
        self.window_values, self.window_masks = window_values, window_masks
        rows = np.array([self.window_lookup.get(str(trend_id), -1) for trend_id in guide_index.trend_ids])
        if (rows < 0).any():
            raise ValueError("trend_catalogに無い候補の記録があります。")
        self.candidate_rows = rows

    def window(self, trend_id):
        row = self.window_lookup[str(trend_id)]
        return self.window_values[row], self.window_masks[row]

    def similarities(self, query_trend_id, indices, *, min_overlap):
        """候補行`indices`との類似度(両方に埋め込みがある月のコサイン類似度の平均)と、重なりが足りるか。"""
        query_values, query_mask = self.window(query_trend_id)
        rows = self.candidate_rows[indices]
        both = self.window_masks[rows] & query_mask[None, :]
        cosine = np.einsum("nkd,kd->nk", self.window_values[rows], query_values)
        overlap = both.sum(axis=1)
        mean = np.where(both, cosine, 0.0).sum(axis=1) / np.maximum(overlap, 1)
        return mean, overlap >= int(min_overlap)

    def search(self, query_trend_id, *, query_date, query_dataset_id, query_current, query_bin_start_m,
               allowed_splits, config=SearchConfig()):
        index = self.guide_index
        eligible, near, distance = index._eligibility(
            query_date=query_date, query_dataset_id=query_dataset_id, query_current=query_current,
            query_bin_start_m=query_bin_start_m, allowed_splits=allowed_splits, config=config,
        )
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return []
        similarity, enough = self.similarities(query_trend_id, indices, min_overlap=config.min_history_overlap_months)
        if not enough.any():
            return []
        return index._select(indices[enough], similarity[enough], near, distance, query_current, config)


def build_waveform_history_index(guide_index, artifact_dir, *, device=None):
    """`artifact_dir`(trend_catalog.csv・split_manifest.csv・waveforms.bin・autoencoder.pt)から作る。"""
    artifact_dir = Path(artifact_dir)
    trend_catalog = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig")
    inference = manifest.loc[manifest["model_split"] == "inference"]
    embeddings = None
    if len(inference):
        embeddings = encode_waveforms(artifact_dir / "waveforms.bin", len(manifest),
                                      inference["waveform_index"].astype(int).to_numpy(),
                                      artifact_dir / "autoencoder.pt", device=device)
    return WaveformHistoryIndex(guide_index, trend_catalog, inference_trend_ids=inference["trend_id"].astype(str),
                                inference_embeddings=embeddings)
