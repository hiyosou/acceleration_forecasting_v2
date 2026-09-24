"""datasets.history.build_history_window の単体テスト(SPEC.md 1.1 / 1.2 対応)。"""

import numpy as np
import pandas as pd
import pytest

from acceleration_forecasting_v2.datasets.history import build_history_window


def _section_history(dates_and_values):
    return pd.DataFrame({
        "measurement_date": pd.to_datetime([date for date, _ in dates_and_values]),
        "acc_z_max": [value for _, value in dates_and_values],
    })


def test_history_window_six_months_all_valid_and_ordered_oldest_to_newest():
    anchor = "2023-07-15"
    rows = [
        ("2023-02-15", 1.0), ("2023-03-15", 2.0), ("2023-04-15", 3.0),
        ("2023-05-15", 4.0), ("2023-06-15", 5.0), (anchor, 6.0),
    ]
    values, mask, dates = build_history_window(_section_history(rows), anchor, months=6)
    assert values.shape == (6,)
    assert mask.shape == (6,)
    assert values.dtype == np.float32
    assert mask.dtype == np.float32
    np.testing.assert_allclose(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert (mask == 1.0).all()
    assert dates[-1] == "2023-07-15"


def test_history_window_partial_missing_month_has_nan_and_zero_mask():
    anchor = "2023-07-15"
    rows = [
        ("2023-02-15", 1.0),
        # 2023-03 は欠測。
        ("2023-04-15", 3.0), ("2023-05-15", 4.0), ("2023-06-15", 5.0), (anchor, 6.0),
    ]
    values, mask, dates = build_history_window(_section_history(rows), anchor, months=6)
    assert mask.tolist() == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert np.isnan(values[1])
    assert dates[1] == ""


def test_history_window_excludes_dates_on_or_after_anchor():
    anchor = "2023-07-15"
    rows = [
        ("2023-06-15", 5.0),
        ("2023-07-15", 6.0),  # anchor自身(今月分として使われる)
        ("2023-07-20", 999.0),  # anchor以降 -> 過去5か月の探索対象からは除外される
    ]
    values, mask, dates = build_history_window(_section_history(rows), anchor, months=2)
    # months=2 -> [過去1か月, 今月]。過去1か月の対象月は2023-06。
    assert mask.tolist() == [1.0, 1.0]
    assert values[0] == pytest.approx(5.0)
    assert values[1] == pytest.approx(6.0)  # 999.0(anchorより後の日付)は混入しない


def test_history_window_current_month_missing_raises():
    anchor = "2023-07-15"
    rows = [("2023-06-15", 5.0)]  # anchor自身の行が無い
    with pytest.raises(ValueError, match="anchor_date"):
        build_history_window(_section_history(rows), anchor, months=2)


def test_history_window_current_month_row_exists_but_value_nan():
    # 行自体は存在するが acc_z_max が非有限(NaN)の場合。「行が無い」場合(raise)とは
    # 区別しなければならない(存在するが欠測扱いは mask=0 で継続、raiseしない)。
    anchor = "2023-07-15"
    rows = _section_history([("2023-06-15", 5.0), (anchor, 1.0)])
    rows.loc[rows["measurement_date"] == pd.Timestamp(anchor), "acc_z_max"] = np.nan
    values, mask, dates = build_history_window(rows, anchor, months=2)
    assert mask[-1] == 0.0
    assert np.isnan(values[-1])
    assert dates[-1] == ""


def test_history_window_prefers_closer_candidate_when_two_are_within_search_days():
    # 対象月は2023-04-01(anchor=2023-05-01のちょうど1か月前)。
    # 2023-04-03(差3日)と2023-04-12(差11日)は両方search_days=15の範囲内だが、
    # より近い前者を優先する(旧実装のタイブレークをそのまま踏襲)。
    anchor = "2023-05-01"
    rows = [
        ("2023-04-03", 20.0),  # 対象月2023-04-01からの差=3日 -> より近い候補
        ("2023-04-12", 30.0),  # 対象月2023-04-01からの差=11日 -> 範囲内だが遠い
        (anchor, 40.0),
    ]
    values, mask, dates = build_history_window(_section_history(rows), anchor, months=2, search_days=15)
    assert mask[0] == 1.0
    assert values[0] == pytest.approx(20.0)
    assert dates[0] == "2023-04-03"


def test_history_window_length_matches_months_argument():
    anchor = "2023-07-15"
    rows = [(anchor, 6.0), ("2023-06-15", 5.0), ("2023-05-15", 4.0), ("2023-04-15", 3.0)]
    values, mask, dates = build_history_window(_section_history(rows), anchor, months=4)
    assert len(values) == len(mask) == len(dates) == 4


def test_history_window_rejects_non_positive_months():
    with pytest.raises(ValueError):
        build_history_window(_section_history([("2023-07-15", 1.0)]), "2023-07-15", months=0)
