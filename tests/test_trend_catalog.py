"""retrieval.trends.TrendCatalog の単体テスト(SPEC.md 1.2 打ち切り仕様に対応)。"""

import json

import pandas as pd
import pytest

from acceleration_forecasting_v2.retrieval.trends import REQUIRED_TREND_COLUMNS, TrendCatalog


def _row(dataset_id, date, acc_z_max, *, segment_index=1, previous_maintenance_date="", direction="U", bin_start_m=2000.0, bin_end_m=2100.0):
    return {
        "dataset_id": dataset_id,
        "direction": direction,
        "bin_start_m": bin_start_m,
        "bin_end_m": bin_end_m,
        "segment_index": segment_index,
        "measurement_date": date,
        "acc_z_max": acc_z_max,
        "measured_at": date,
        "previous_maintenance_date": previous_maintenance_date,
        "maintenance_type": "",
        "maintenance_description": "",
    }


def _monthly_rows(dataset_id, start="2023-01-15", months=14, value=1.0, **kwargs):
    # 月ごとに厳密に "start + iか月" の日付を生成する(pd.date_range(freq="MS")は
    # startが月初でない場合に開始月をスキップしてしまうため、意図的に使わない)。
    start_date = pd.Timestamp(start)
    dates = [start_date + pd.DateOffset(months=index) for index in range(months)]
    return [_row(dataset_id, date.strftime("%Y-%m-%d"), value, **kwargs) for date in dates]


def test_build_trend_no_maintenance_full_history():
    rows = _monthly_rows("segA", months=14, value=2.0)
    catalog = TrendCatalog(pd.DataFrame(rows))
    anchor = catalog.get_anchor(rows[0]["measurement_date"], "U", 2000.0, 2100.0)
    assert anchor is not None
    trend = catalog.build_trend(anchor, forecast_months=12)
    mask = json.loads(trend["future_mask"])
    values = json.loads(trend["future_values"])
    assert len(mask) == 12
    assert sum(mask) == 12
    assert all(value == pytest.approx(2.0) for value in values)
    assert trend["cutoff_maintenance_date"] == ""


def test_build_trend_truncates_at_next_maintenance():
    # segAは2023-01始まり、segBが2023-06に開始(=segAの補修打ち切り月)する2セグメント構成。
    # segBは日付をsegAとずらす(20日)ことで、同一(日付,方向,区間)によるTrendCatalogの
    # 重複排除でsegAの行に上書きされない(=同じ場所で2つのdataset_idが同時刻に存在する
    # ことは実データ上ありえないため、意図的にセグメント間で日付を分けている)。
    rows_a = _monthly_rows("segA", start="2023-01-15", months=14, value=1.0, segment_index=1)
    rows_b = _monthly_rows(
        "segB", start="2023-06-20", months=6, value=9.0, segment_index=2,
        previous_maintenance_date="2023-06-01",
    )
    catalog = TrendCatalog(pd.DataFrame(rows_a + rows_b))
    anchor = catalog.get_anchor(rows_a[0]["measurement_date"], "U", 2000.0, 2100.0)
    trend = catalog.build_trend(anchor, forecast_months=12)
    mask = json.loads(trend["future_mask"])
    values = json.loads(trend["future_values"])
    # anchor=2023-01-15 -> 対象月は2023-02〜2024-01。cutoff=2023-06-01以降は打ち切り。
    assert mask == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert values[4:] == [None] * 8
    assert trend["cutoff_maintenance_date"] == "2023-06-01"


def test_build_trend_masked_value_is_none_not_zero():
    rows = _monthly_rows("segC", months=2, value=3.0)
    catalog = TrendCatalog(pd.DataFrame(rows))
    anchor = catalog.get_anchor(rows[0]["measurement_date"], "U", 2000.0, 2100.0)
    trend = catalog.build_trend(anchor, forecast_months=12)
    values = json.loads(trend["future_values"])
    mask = json.loads(trend["future_mask"])
    for value, flag in zip(values, mask):
        if flag == 0:
            assert value is None
        else:
            assert value is not None


def test_build_trend_id_is_deterministic():
    rows = _monthly_rows("segD", months=12, value=1.5)
    catalog = TrendCatalog(pd.DataFrame(rows))
    anchor = catalog.get_anchor(rows[0]["measurement_date"], "U", 2000.0, 2100.0)
    trend_1 = catalog.build_trend(anchor, forecast_months=12)
    trend_2 = catalog.build_trend(anchor, forecast_months=12)
    assert trend_1["trend_id"] == trend_2["trend_id"]
    assert len(trend_1["trend_id"]) == 24


def test_get_anchor_returns_none_for_missing_key():
    rows = _monthly_rows("segE", months=3, value=1.0)
    catalog = TrendCatalog(pd.DataFrame(rows))
    assert catalog.get_anchor("2099-01-01", "U", 2000.0, 2100.0) is None


def test_required_trend_columns_are_documented():
    assert REQUIRED_TREND_COLUMNS == {
        "dataset_id", "direction", "bin_start_m", "bin_end_m",
        "segment_index", "measurement_date", "acc_z_max",
    }
