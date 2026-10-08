"""guide検索。`acceleration_forecasting_12m/.../retrieval/search.py`を無変更で移植。

split境界を跨いだ漏洩を防ぐロジック(allowed_splits・同一dataset_id除外・
近傍guideの時間順序制約)はSPEC.md 2.3で踏襲を確認済みのため変更していない。

2026-10-08: `search_by_history`(6か月履歴ベクトルの類似度による検索、ユーザー判断)を追加。
raw波形embeddingの代わりに、モデルの生成条件そのもの(history_values、anchors.pyが計算する
のと同じ6点)を検索クエリとして使う検証用。候補プールは従来通り`waveform_records JOIN trends`
(embeddingベース検索と揃える、ユーザー判断)のまま、各候補について自身のdataset_id内での
6か月履歴を`datasets.history.build_history_window`で別途構築して保持する。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.common.constants import HISTORY_MONTHS
from acceleration_forecasting_v2.datasets.history import build_history_window


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 3
    max_current_difference: float | None = 0.5
    min_valid_months: int = 8
    near_distance_m: float = 100.0
    spatial_tolerance_m: float = 1e-6
    # search_by_history専用: 両者ともmask=1の月が何か月以上重ならないと候補から除外するか。
    min_history_overlap_months: int = 3


class GuideIndex:
    def __init__(self, connection: sqlite3.Connection):
        rows = connection.execute("""
            SELECT w.record_id,w.measurement_id,w.measurement_date,w.embedding,w.embedding_dim,
                   t.trend_id,t.dataset_id,t.model_split,t.current_acc_z_max,t.future_values,
                   t.future_mask,t.selected_dates,t.guide_available_date,t.direction,
                   t.bin_start_m,t.bin_end_m,t.cutoff_maintenance_date,
                   t.maintenance_type,t.maintenance_description
            FROM waveform_records w JOIN trends t ON t.trend_id=w.trend_id
        """).fetchall()
        if not rows:
            raise ValueError("Guide database is empty")
        dimension = int(rows[0][4])
        self.embeddings = np.stack([np.frombuffer(row[3], dtype=np.float32, count=dimension) for row in rows])
        self.record_ids = np.asarray([row[0] for row in rows], dtype=object)
        self.measurement_ids = np.asarray([row[1] for row in rows], dtype=object)
        self.dates = np.asarray([str(row[2]) for row in rows], dtype="U10")
        self.trend_ids = np.asarray([row[5] for row in rows], dtype=object)
        self.datasets = np.asarray([row[6] for row in rows], dtype=object)
        self.splits = np.asarray([row[7] for row in rows], dtype=object)
        self.current = np.asarray([row[8] for row in rows], dtype=np.float32)
        self.future_values = [json.loads(row[9]) for row in rows]
        self.future_masks = [json.loads(row[10]) for row in rows]
        self.selected_dates = [json.loads(row[11]) for row in rows]
        self.available_dates = np.asarray([str(row[12]) for row in rows], dtype="U10")
        self.directions = np.asarray([row[13] for row in rows], dtype=object)
        self.bin_starts = np.asarray([row[14] for row in rows], dtype=np.float32)
        self.bin_ends = np.asarray([row[15] for row in rows], dtype=np.float32)
        self.cutoffs = [row[16] or "" for row in rows]
        self.maintenance_types = [row[17] or "" for row in rows]
        self.maintenance_descriptions = [row[18] or "" for row in rows]
        self.valid_months = np.asarray([sum(bool(value) for value in mask) for mask in self.future_masks], dtype=np.int16)
        self.history_values, self.history_masks = self._build_history_vectors(connection, months=HISTORY_MONTHS)

    def _build_history_vectors(self, connection, *, months):
        """各候補行について、自身のdataset_id内での(過去months-1か月+今月)履歴を構築する。

        `datasets.anchors.select_anchors_for_segment`がanchor自身について行っているのと
        同じ計算(`build_history_window`)を、guide候補側にも適用したもの。`trends`テーブル
        (`waveform_records`との結合前の全行)だけから計算でき、embeddingには依存しない。

        `waveform_records`は同一trend_id(=同一dataset_id×measurement_date、同日複数走行)を
        複数行持ちうるが、履歴はtrend_idだけで決まる。実データで候補行数(waveform_records)が
        一意なtrend_id数の3倍以上になるケースがあり、素朴に全行で計算すると非常に遅くなる
        (実測: 335,986行で約1時間)ため、trend_idごとに1回だけ計算してキャッシュする。
        """
        trend_rows = connection.execute("SELECT dataset_id, measurement_date, current_acc_z_max FROM trends").fetchall()
        frame = pd.DataFrame(trend_rows, columns=["dataset_id", "measurement_date", "acc_z_max"])
        frame["measurement_date"] = pd.to_datetime(frame["measurement_date"], errors="coerce")
        groups = {
            str(dataset_id): group.sort_values("measurement_date", kind="mergesort")
            for dataset_id, group in frame.groupby("dataset_id", sort=False)
        }
        empty = (np.full(months, np.nan, dtype=np.float32), np.zeros(months, dtype=np.float32))
        cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        values = np.full((len(self.record_ids), months), np.nan, dtype=np.float32)
        masks = np.zeros((len(self.record_ids), months), dtype=np.float32)
        for row_index in range(len(self.record_ids)):
            trend_id = str(self.trend_ids[row_index])
            cached = cache.get(trend_id)
            if cached is None:
                group = groups.get(str(self.datasets[row_index]))
                if group is None:
                    cached = empty
                else:
                    anchor_date = pd.Timestamp(str(self.dates[row_index]))
                    row_values, row_masks, _ = build_history_window(group, anchor_date, months=months)
                    cached = (row_values, row_masks)
                cache[trend_id] = cached
            values[row_index], masks[row_index] = cached
        return values, masks

    def _eligibility(self, *, query_date, query_dataset_id, query_current, query_bin_start_m, allowed_splits, config):
        allowed_splits = set(str(value) for value in allowed_splits)
        eligible = np.fromiter((str(value) in allowed_splits for value in self.splits), dtype=bool)
        eligible &= self.datasets != str(query_dataset_id)
        eligible &= self.dates != str(query_date)
        eligible &= np.isfinite(self.current)
        if config.max_current_difference is not None:
            eligible &= np.abs(self.current - float(query_current)) <= float(config.max_current_difference) + 1e-12
        eligible &= self.valid_months >= config.min_valid_months
        distance = np.abs(self.bin_starts.astype(np.float64) - float(query_bin_start_m))
        near = distance <= config.near_distance_m + config.spatial_tolerance_m
        eligible &= ~near | (self.available_dates < str(query_date))
        return eligible, near, distance

    def _select(self, indices, best, near, distance, query_current, config):
        order = np.argsort(-best, kind="stable")
        selected, used_dates = [], set()
        for offset in order:
            index = int(indices[int(offset)])
            date = str(self.dates[index])
            if date in used_dates:
                continue
            used_dates.add(date)
            selected.append({
                "record_id": self.record_ids[index], "measurement_id": self.measurement_ids[index],
                "measurement_date": date, "trend_id": self.trend_ids[index],
                "dataset_id": self.datasets[index], "model_split": self.splits[index],
                "current_acc_z_max": float(self.current[index]),
                "current_max_difference": float(self.current[index] - query_current),
                "future_values": self.future_values[index], "future_mask": self.future_masks[index],
                "selected_dates": self.selected_dates[index], "valid_months": int(self.valid_months[index]),
                "guide_available_date": str(self.available_dates[index]),
                "direction": self.directions[index], "bin_start_m": float(self.bin_starts[index]),
                "bin_end_m": float(self.bin_ends[index]), "distance_difference_m": float(distance[index]),
                "spatially_near": bool(near[index]), "temporal_condition_applied": bool(near[index]),
                "similarity": float(best[int(offset)]), "cutoff_maintenance_date": self.cutoffs[index],
                "maintenance_type": self.maintenance_types[index],
                "maintenance_description": self.maintenance_descriptions[index],
            })
            if len(selected) >= config.top_k:
                break
        return selected

    def search(self, query_embeddings, *, query_date, query_dataset_id, query_current,
               query_bin_start_m, allowed_splits, config=SearchConfig()):
        query = np.asarray(query_embeddings, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        norms = np.linalg.norm(query, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 1e-12).any():
            raise ValueError("Query embeddings must be finite and non-zero")
        query = query / norms
        eligible, near, distance = self._eligibility(
            query_date=query_date, query_dataset_id=query_dataset_id, query_current=query_current,
            query_bin_start_m=query_bin_start_m, allowed_splits=allowed_splits, config=config,
        )
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return []
        similarities = query @ self.embeddings[indices].T
        best = similarities.max(axis=0)
        return self._select(indices, best, near, distance, query_current, config)

    def search_by_history(self, query_history_values, query_history_mask, *, query_date, query_dataset_id,
                           query_current, query_bin_start_m, allowed_splits, config=SearchConfig()):
        """6か月履歴ベクトル(anchors.pyが計算するhistory_values/history_maskと同じもの)の
        類似度でguideを検索する(2026-10-08、ユーザー判断による`search`の代替検証)。

        類似度は、両者とも有効(mask=1)な月だけを使った負のRMSE(両者とも値が存在する月の
        二乗誤差平均の平方根の符号反転、値が近いほど大きい)。重なりが
        `config.min_history_overlap_months`未満の候補は除外する。`search`と同じ
        allowed_splits・同一dataset_id除外・current_acc_z_max近傍・空間/時間リーク制約を適用する
        (変更したのは類似度の入力だけ)。
        """
        query_values = np.asarray(query_history_values, dtype=np.float32)
        query_mask = np.asarray(query_history_mask, dtype=np.float32) > 0
        if query_values.shape != query_mask.shape or query_values.ndim != 1:
            raise ValueError("query_history_values/query_history_mask must be 1-D arrays of the same shape")
        if not query_mask.any():
            raise ValueError("query_history_mask has no valid (mask=1) month")
        eligible, near, distance = self._eligibility(
            query_date=query_date, query_dataset_id=query_dataset_id, query_current=query_current,
            query_bin_start_m=query_bin_start_m, allowed_splits=allowed_splits, config=config,
        )
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return []
        candidate_values = self.history_values[indices]
        candidate_mask = self.history_masks[indices] > 0
        both_valid = candidate_mask & query_mask[None, :]
        overlap_count = both_valid.sum(axis=1)
        enough_overlap = overlap_count >= config.min_history_overlap_months
        if not enough_overlap.any():
            return []
        # near/distanceは全行ベースのまま(_selectが絞り込み後のindexで引き直す)なので、
        # candidate側だけenough_overlapで絞り込めば十分(indices自体を絞り込めば連動する)。
        indices = indices[enough_overlap]
        candidate_values = candidate_values[enough_overlap]
        both_valid = both_valid[enough_overlap]
        overlap_count = overlap_count[enough_overlap]
        squared_error = np.where(both_valid, (candidate_values - query_values[None, :]) ** 2, 0.0)
        rmse = np.sqrt(squared_error.sum(axis=1) / overlap_count)
        best = -rmse
        return self._select(indices, best, near, distance, query_current, config)
