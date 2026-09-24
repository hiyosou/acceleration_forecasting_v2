"""datasets.build の単体テスト(SPEC.md 2.1データ契約、target_mode=absoluteの決定に対応)。"""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from acceleration_forecasting_v2.retrieval.database import (
    initialize_database,
    insert_trends,
    insert_waveform_record,
)
from acceleration_forecasting_v2.retrieval.search import GuideIndex, SearchConfig
from acceleration_forecasting_v2.datasets.anchors import AnchorCandidate, compute_segment_weights
from acceleration_forecasting_v2.datasets.build import (
    ARRAY_NAMES,
    build_raw_split,
    fit_condition_and_target_normalization,
)


def _unit_vector(seed, dim=8):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _guide_trend_row(trend_id, dataset_id, date, value, *, bin_start_m=9000.0, current=None):
    # current(query側のmax_current_differenceフィルタに使われる値)と、future_value
    # (未来の推移そのもの)を独立に指定できるようにする。current未指定ならvalueと同じ。
    future_values = [value] * 12
    future_mask = [1] * 12
    return {
        "trend_id": trend_id, "dataset_id": dataset_id, "model_split": "model_train",
        "measurement_date": date, "direction": "U", "bin_start_m": bin_start_m, "bin_end_m": bin_start_m + 100.0,
        "current_acc_z_max": value if current is None else current,
        "future_values": json.dumps(future_values), "future_mask": json.dumps(future_mask),
        "selected_dates": json.dumps([None] * 12), "guide_available_date": "2000-01-01",
        "cutoff_maintenance_date": "", "maintenance_type": "", "maintenance_description": "",
    }


def _build_index_with_one_guide(tmp_path, guide_value=2.0, *, guide_current=None):
    connection = initialize_database(tmp_path / "guide.sqlite")
    row = _guide_trend_row("guide-1", "seg-guide", "2023-01-01", guide_value, current=guide_current)
    insert_trends(connection, pd.DataFrame([row]))
    insert_waveform_record(connection, {
        "record_id": "rec-guide-1", "measurement_id": "meas-guide-1", "measurement_date": row["measurement_date"],
        "direction": "U", "bin_start_m": row["bin_start_m"], "bin_end_m": row["bin_end_m"], "mean_velocity_kmh": 60.0,
        "source_csv_path": "dummy.csv", "waveform_sha256": "abc", "dataset_id": "seg-guide", "trend_id": "guide-1",
    }, _unit_vector(1))
    connection.commit()
    connection.close()
    return GuideIndex(sqlite3.connect(str(tmp_path / "guide.sqlite")))


def _anchor(trend_id="a1", dataset_id="seg-query", date="2023-06-15", current=1.0, forecast_months=12):
    return AnchorCandidate(
        trend_id=trend_id, dataset_id=dataset_id, measurement_date=pd.Timestamp(date),
        bin_start_m=2000.0, bin_end_m=2100.0, direction="U", current_acc_z_max=current,
        history_values=np.array([1, 1, 1, 1, 1, current], dtype=np.float32),
        history_mask=np.ones(6, dtype=np.float32),
        future_values=np.full(forecast_months, current, dtype=np.float32),
        future_mask=np.ones(forecast_months, dtype=np.float32),
        valid_history_months=6, valid_future_months=forecast_months,
    )


def test_build_raw_split_produces_all_array_names_with_matching_row_count(tmp_path):
    # guideのcurrent値をanchorと同じにして、max_current_differenceフィルタを実際に通し、
    # guideが1件見つかった状態でshapeを検証する(見つからなくてもshape自体は同じに
    # なってしまうため、意図的に「実際に見つかる」設定にしてテストを意味のあるものにする)。
    index = _build_index_with_one_guide(tmp_path, guide_value=2.0, guide_current=1.0)
    anchors = [_anchor()]
    segment_weights = compute_segment_weights(anchors)
    result = build_raw_split(
        anchors, split="model_train", guide_index=index,
        query_vectors_by_trend_id={"a1": [_unit_vector(1)]},
        allowed_guide_splits={"model_train"}, segment_weights=segment_weights,
    )
    assert result.metadata.iloc[0]["guide_count"] == 1
    assert set(result.arrays.keys()) == set(ARRAY_NAMES)
    for name in ARRAY_NAMES:
        assert result.arrays[name].shape[0] == 1
    assert result.arrays["guide_values"].shape == (1, 3, 12)
    assert result.arrays["guide_softmax_weights"].shape == (1, 3, 12)
    assert result.arrays["history_values"].shape == (1, 6)
    assert result.arrays["target_values"].shape == (1, 12)
    assert len(result.metadata) == 1


def test_build_raw_split_target_values_are_absolute_not_residual(tmp_path):
    # target_mode=absolute の決定: target_values は anchor.future_values そのものであり、
    # guide_baselineを引いた残差ではない。
    # guideのcurrent値はanchorと近くしてmax_current_differenceフィルタを通す一方、
    # guideの「未来の推移」だけをanchorのtarget(5.0)と大きく異ならせる(99.0)。
    index = _build_index_with_one_guide(tmp_path, guide_value=99.0, guide_current=5.0)
    anchor = _anchor(current=5.0)  # future_values は _anchor 内で current=5.0 埋めになっている
    segment_weights = compute_segment_weights([anchor])
    result = build_raw_split(
        [anchor], split="model_train", guide_index=index,
        query_vectors_by_trend_id={"a1": [_unit_vector(1)]},
        allowed_guide_splits={"model_train"}, segment_weights=segment_weights,
    )
    np.testing.assert_allclose(result.arrays["target_values"][0], 5.0)
    # guide(99.0)がベースラインに使われていても、target_valuesには一切混ざらない。
    assert not np.allclose(result.arrays["target_values"][0], 99.0)
    # guide_baselinesは評価用の参考値として引き続き計算される(こちらはguideの値に近づく)。
    np.testing.assert_allclose(result.arrays["guide_baselines"][0], 99.0, atol=1e-3)


def test_build_raw_split_segment_weight_attached_per_anchor():
    anchors = [_anchor(trend_id="a1", dataset_id="segX"), _anchor(trend_id="a2", dataset_id="segX", date="2023-07-15"),
               _anchor(trend_id="a3", dataset_id="segY", date="2023-08-15")]
    weights = compute_segment_weights(anchors)
    result = build_raw_split(
        anchors, split="model_train", guide_index=_empty_guide_index_stub(),
        query_vectors_by_trend_id={}, allowed_guide_splits={"model_train"}, segment_weights=weights,
    )
    np.testing.assert_allclose(result.arrays["segment_weights"], [0.5, 0.5, 1.0])
    assert result.metadata["dataset_id"].tolist() == ["segX", "segX", "segY"]


def test_build_raw_split_no_query_vectors_falls_back_to_current_value_baseline():
    # queryの埋め込みが無い(guide検索できない)場合、guide_countは0になり、
    # guide_baselineはsoftmax_guide_baselineのフォールバック(current値)になる。
    anchor = _anchor(current=3.5)
    weights = compute_segment_weights([anchor])
    result = build_raw_split(
        [anchor], split="model_train", guide_index=_empty_guide_index_stub(),
        query_vectors_by_trend_id={}, allowed_guide_splits={"model_train"}, segment_weights=weights,
    )
    assert result.metadata.iloc[0]["guide_count"] == 0
    np.testing.assert_allclose(result.arrays["guide_masks"][0], 0.0)
    np.testing.assert_allclose(result.arrays["guide_baselines"][0], 3.5)


class _NoOpGuideIndex:
    def search(self, *args, **kwargs):
        raise AssertionError("query_vectorsが空ならsearchは呼ばれないはず")


def _empty_guide_index_stub():
    return _NoOpGuideIndex()


def test_fit_condition_and_target_normalization_uses_only_masked_values():
    from acceleration_forecasting_v2.datasets.build import RawSplit

    arrays = {
        "history_values": np.array([[10.0, 20.0]], dtype=np.float32),
        "history_masks": np.array([[1.0, 0.0]], dtype=np.float32),  # 20.0はmask=0なので除外される
        "guide_values": np.array([[[30.0]]], dtype=np.float32),
        "guide_masks": np.array([[[1.0]]], dtype=np.float32),
        "target_values": np.array([[100.0, 200.0]], dtype=np.float32),
        "target_masks": np.array([[1.0, 1.0]], dtype=np.float32),
    }
    raw = RawSplit(arrays=arrays, metadata=pd.DataFrame())
    condition_norm, target_norm = fit_condition_and_target_normalization(raw)
    # condition_norm は 10.0(history, mask=1) と 30.0(guide, mask=1) のみから計算される。
    assert condition_norm.mean == pytest.approx((10.0 + 30.0) / 2)
    assert target_norm.mean == pytest.approx((100.0 + 200.0) / 2)
