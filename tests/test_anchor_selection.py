"""datasets.anchors の単体テスト(SPEC.md 決定「セグメント内の全ての月を拾う」に対応)。

TrendCatalog(ステップ2で実装済み・テスト済み)を使って現実的なtrend_catalog行を
組み立ててから select_anchors_for_segment に渡す。これにより「未来値の構築」と
「anchor選定」の結合部分も同時に確認できる。
"""

import pandas as pd
import pytest

from acceleration_forecasting_v2.retrieval.trends import TrendCatalog
from acceleration_forecasting_v2.datasets.anchors import compute_segment_weights, select_anchors_for_segment


def _build_segment_rows(dataset_id, months, *, start="2022-01-15", value=1.0, forecast_months=12):
    start_date = pd.Timestamp(start)
    dates = [start_date + pd.DateOffset(months=index) for index in range(months)]
    rows = [{
        "dataset_id": dataset_id, "direction": "U", "bin_start_m": 2000.0, "bin_end_m": 2100.0,
        "segment_index": 1, "measurement_date": date.strftime("%Y-%m-%d"), "acc_z_max": value,
        "measured_at": date.strftime("%Y-%m-%d"), "previous_maintenance_date": "",
        "maintenance_type": "", "maintenance_description": "",
    } for date in dates]
    catalog = TrendCatalog(pd.DataFrame(rows))
    trend_records = []
    for date in dates:
        anchor = catalog.get_anchor(date.strftime("%Y-%m-%d"), "U", 2000.0, 2100.0)
        trend_records.append(catalog.build_trend(anchor, forecast_months=forecast_months))
    return pd.DataFrame(trend_records)


def test_select_anchors_collects_all_eligible_months_not_just_one():
    # 20か月連続でデータがある1セグメント。history_months=6なので「今月」自体も
    # 有効な履歴1か月分として数えられる点に注意(過去の実測がn か月あるanchorの
    # valid_history_months = min(n,5)+1)。min_history=3を満たすのは3か月目以降
    # (今月+過去2か月=3)、min_target=8(12か月中8か月以上)を満たすのは
    # 残り20-i か月分の未来のうち8以上が必要 -> 12か月目以前。
    # よって3〜12か月目の10件が選ばれるはず(旧実装なら1件のみだった)。
    segment_rows = _build_segment_rows("segA", months=20)
    selected = select_anchors_for_segment(
        segment_rows, split="model_train", min_history_months=3, min_target_months=8,
    )
    assert len(selected) == 10
    dates = [anchor.measurement_date.strftime("%Y-%m-%d") for anchor in selected]
    assert dates == sorted(dates)
    assert dates[0] == "2022-03-15"  # 3か月目
    assert dates[-1] == "2022-12-15"  # 12か月目


def test_select_anchors_respects_min_target_months_threshold():
    segment_rows = _build_segment_rows("segB", months=20)
    lenient = select_anchors_for_segment(segment_rows, split="model_train", min_history_months=3, min_target_months=1)
    strict = select_anchors_for_segment(segment_rows, split="model_train", min_history_months=3, min_target_months=12)
    assert len(lenient) > len(strict)
    assert len(strict) == 6  # 履歴>=3(3か月目以降)かつ未来12か月全て埋まる(3〜8か月目)


def test_select_anchors_inference_split_ignores_future_requirement():
    segment_rows = _build_segment_rows("segC", months=20)
    inference = select_anchors_for_segment(
        segment_rows, split="inference", min_history_months=3, min_target_months=8,
    )
    train = select_anchors_for_segment(
        segment_rows, split="model_train", min_history_months=3, min_target_months=8,
    )
    # inferenceは未来を要求しないため、末尾の月(未来が足りない月)も含まれ、trainより多い。
    assert len(inference) > len(train)
    assert len(inference) == 18  # 履歴>=3を満たす3〜20か月目


def test_select_anchors_empty_segment_returns_empty_list():
    empty = pd.DataFrame(columns=["dataset_id", "direction", "bin_start_m", "bin_end_m", "measurement_date",
                                   "current_acc_z_max", "future_values", "future_mask", "trend_id"])
    assert select_anchors_for_segment(empty, split="model_train", min_history_months=3, min_target_months=8) == []


def test_select_anchors_no_eligible_candidate_returns_empty_list():
    # 3か月しかデータが無いセグメント。3か月目は履歴条件(今月+過去2か月=3)は
    # 満たすが、その先の実測が1件も無いため valid_future_months=0 となり、
    # min_target_months=1(train/validationは未来も必須)を満たせず、
    # 結果として1件も選ばれない。
    segment_rows = _build_segment_rows("segD", months=3)
    selected = select_anchors_for_segment(segment_rows, split="model_train", min_history_months=3, min_target_months=1)
    assert selected == []


def test_select_anchors_rejects_future_values_length_mismatch_with_forecast_months():
    # trend_catalogのforecast_months(future_valuesの長さ)と、呼び出し側が指定する
    # forecast_monthsが食い違っている場合、値を静かに切り詰めたりせず明確に失敗する。
    segment_rows = _build_segment_rows("segG", months=6, forecast_months=6)
    with pytest.raises(ValueError, match="forecast_months"):
        select_anchors_for_segment(
            segment_rows, split="model_train", min_history_months=1, min_target_months=1,
            forecast_months=12,
        )


def test_compute_segment_weights_matches_anchor_counts():
    segment_a = _build_segment_rows("segE", months=20)
    segment_b = _build_segment_rows("segF", months=6, start="2023-01-15")
    anchors_a = select_anchors_for_segment(segment_a, split="model_train", min_history_months=3, min_target_months=8)
    anchors_b = select_anchors_for_segment(segment_b, split="model_train", min_history_months=3, min_target_months=1)
    weights = compute_segment_weights(anchors_a + anchors_b)
    assert weights["segE"] == pytest.approx(1.0 / len(anchors_a))
    assert weights["segF"] == pytest.approx(1.0 / len(anchors_b))
    # 検算: 各セグメントの (重み × 件数) の合計は必ず1になる。
    assert weights["segE"] * len(anchors_a) == pytest.approx(1.0)
