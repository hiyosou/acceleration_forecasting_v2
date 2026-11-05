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

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import FORECAST_MONTHS, HISTORY_MONTHS
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


def trend_cycle_ages(trend_catalog) -> dict:
    """各記録の「補修(サイクル起点`cycle_start`)からの経過月数」(暦月の差、起点の月=0)。

    `datasets.anchors._months_after`と同じ定義。`cycle_start`列が無い、または空の記録は含めない。
    """
    if "cycle_start" not in trend_catalog.columns:
        return {}
    frame = trend_catalog[["trend_id", "measurement_date", "cycle_start"]].copy()
    start = pd.to_datetime(frame["cycle_start"], errors="coerce")
    date = pd.to_datetime(frame["measurement_date"])
    age = (date.dt.year - start.dt.year) * 12 + date.dt.month - start.dt.month
    frame = frame.assign(age=age).dropna(subset=["age"])
    return dict(zip(frame["trend_id"].astype(str), frame["age"].astype(int)))


class WaveformHistoryIndex:
    """`GuideIndex`の候補(走行単位の行)に、履歴(既定6か月、`months`で変更)の波形埋め込みの並びを
    対応させた検索器。`months`を大きくした窓は、補修より前の月が欠測になるので、補修直後から起点までの
    「伸びていく窓」になる(2026-11、`SearchConfig.same_cycle_age`で経過月数のそろった候補に限れる)。"""

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
        # 補修からの経過月数(窓の行ごと)。`cycle_start`が無ければ全て-1(same_cycle_age・上限は使えない)。
        ages = trend_cycle_ages(trend_catalog)
        self.window_age = np.full(len(windows), -1, dtype=np.int32)
        for trend_id, row in self.window_lookup.items():
            self.window_age[row] = ages.get(trend_id, -1)
        self.candidate_age = self.window_age[rows]
        self._capped_valid_cache = {}

    def capped_valid_months(self, max_months_after_maintenance):
        """候補ごとの有効な将来月数(補修から`max_months_after_maintenance`か月を超える月を除いたもの)。"""
        cap = int(max_months_after_maintenance)
        if cap not in self._capped_valid_cache:
            if (self.candidate_age < 0).any():
                raise ValueError("max_months_after_maintenanceにはtrend_catalogのcycle_start列が必要です。")
            offsets = np.arange(1, FORECAST_MONTHS + 1)
            within = (self.candidate_age[:, None] + offsets[None, :]) <= cap
            masks = np.asarray(self.guide_index.future_masks, dtype=np.float32)
            self._capped_valid_cache[cap] = (masks * within).sum(axis=1).astype(np.int16)
        return self._capped_valid_cache[cap]

    def _cap_guide_future(self, guide, cap):
        """選ばれたガイドの将来を、補修から`cap`か月を超える月で欠測にする(元のリストは書き換えない)。"""
        age = int(self.window_age[self.window_lookup[str(guide["trend_id"])]])
        beyond = (age + np.arange(1, len(guide["future_mask"]) + 1)) > int(cap)
        guide = dict(guide)
        guide["future_mask"] = [0 if drop else value for value, drop in zip(guide["future_mask"], beyond)]
        guide["future_values"] = [None if drop else value for value, drop in zip(guide["future_values"], beyond)]
        guide["valid_months"] = int(sum(bool(value) for value in guide["future_mask"]))
        return guide

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
        cap = config.max_months_after_maintenance
        if cap is not None:
            eligible &= self.capped_valid_months(cap) >= config.min_valid_months
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return []
        similarity, enough = self.similarities(query_trend_id, indices, min_overlap=config.min_history_overlap_months)
        if not enough.any():
            return []
        indices, similarity = indices[enough], similarity[enough]
        if not config.same_cycle_age:
            guides = index._select(indices, similarity, near, distance, query_current, config)
        else:
            guides = self._select_same_age(query_trend_id, indices, similarity, near, distance, query_current, config)
        if cap is not None:
            guides = [self._cap_guide_future(guide, cap) for guide in guides]
        return guides

    def _select_same_age(self, query_trend_id, indices, similarity, near, distance, query_current, config):
        """経過月数が問い合わせと同じ候補から上位を選び、足りなければ差が`cycle_age_fallback_months`以内の候補で補う。"""
        query_age = int(self.window_age[self.window_lookup[str(query_trend_id)]])
        if query_age < 0 or (self.candidate_age[indices] < 0).any():
            raise ValueError("same_cycle_ageにはtrend_catalogのcycle_start列が必要です。")
        gap = np.abs(self.candidate_age[indices] - query_age)
        index = self.guide_index
        selected = index._select(indices[gap == 0], similarity[gap == 0], near, distance, query_current, config)
        for step in range(1, int(config.cycle_age_fallback_months) + 1):
            if len(selected) >= config.top_k:
                break
            used = {guide["measurement_date"] for guide in selected}
            pool = (gap == step) & ~np.isin(index.dates[indices], list(used))
            if not pool.any():
                continue
            rest = replace(config, top_k=config.top_k - len(selected))
            selected += index._select(indices[pool], similarity[pool], near, distance, query_current, rest)
        return selected


def build_waveform_history_index(guide_index, artifact_dir, *, device=None, months=HISTORY_MONTHS,
                                 trend_catalog_path=None):
    """`artifact_dir`(trend_catalog.csv・split_manifest.csv・waveforms.bin・autoencoder.pt)から作る。

    months: 窓の長さ。trend_catalog_path: 既定は`artifact_dir/trend_catalog.csv`。補修からの経過月数
    (`SearchConfig.same_cycle_age`・`max_months_after_maintenance`)を使うときは、`cycle_start`列を持つ
    延長版(`datasets.extension.build_extended_artifacts`の出力)を指定する。
    """
    artifact_dir = Path(artifact_dir)
    trend_catalog = pd.read_csv(trend_catalog_path or artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig")
    inference = manifest.loc[manifest["model_split"] == "inference"]
    embeddings = None
    if len(inference):
        embeddings = encode_waveforms(artifact_dir / "waveforms.bin", len(manifest),
                                      inference["waveform_index"].astype(int).to_numpy(),
                                      artifact_dir / "autoencoder.pt", device=device)
    return WaveformHistoryIndex(guide_index, trend_catalog, inference_trend_ids=inference["trend_id"].astype(str),
                                inference_embeddings=embeddings, months=months)
