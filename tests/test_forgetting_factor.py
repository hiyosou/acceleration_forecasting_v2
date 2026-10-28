import numpy as np
import pandas as pd
import pytest

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_v2.evaluation.forgetting_factor import (
    DEFAULT_LAMBDAS,
    cycle_measurements,
    forgetting_factor_line,
    forgetting_factor_lines,
)


def test_default_lambdas_are_one_to_point_nine_in_hundredths():
    assert DEFAULT_LAMBDAS == (1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9)


def test_lambda_one_matches_ordinary_least_squares_on_a_day_axis():
    dates = pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-11", "2024-03-02", "2024-03-30"])
    values = np.array([1.2, 1.5, 1.4, 1.9, 2.0])
    grid = pd.to_datetime(["2024-04-01", "2024-06-01"])
    x = (dates - dates.max()).days.to_numpy(float)
    slope, intercept = np.polyfit(x, values, 1)
    expected = intercept + slope * (grid - dates.max()).days.to_numpy(float)
    np.testing.assert_allclose(forgetting_factor_line(dates, values, grid, 1.0), expected)


def test_weights_follow_measurement_count_not_elapsed_days():
    # 重み λ^(N−i) は「何回前の計測か」で決まる。日付の間隔を変えても重みの並びは同じなので、
    # 日付を一様に伸ばした入力どうしでは、当てはめた直線の各計測日での値が一致する。
    values = np.array([1.0, 1.6, 1.2, 1.8])
    dense = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"])
    sparse = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-21", "2024-01-31"])
    a = forgetting_factor_line(dense, values, dense, 0.5)
    b = forgetting_factor_line(sparse, values, sparse, 0.5)
    np.testing.assert_allclose(a, b)


def test_small_lambda_approaches_the_line_through_the_two_latest_points():
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    values = np.array([3.0, 1.0, 1.5, 2.0])
    line = forgetting_factor_line(dates, values, dates[-2:], 1e-6)
    np.testing.assert_allclose(line, values[-2:], atol=1e-4)


def test_single_point_gives_a_flat_line_and_values_are_clipped_to_physical_bounds():
    grid = pd.to_datetime(["2024-01-01", "2025-01-01"])
    np.testing.assert_allclose(forgetting_factor_line(pd.to_datetime(["2024-01-01"]), [1.7], grid, 0.9), [1.7, 1.7])
    steep = forgetting_factor_line(pd.to_datetime(["2024-01-01", "2024-01-02"]), [1.0, 3.0], grid, 1.0)
    assert steep.min() >= PHYSICAL_MIN and steep.max() <= PHYSICAL_MAX
    with pytest.raises(ValueError):
        forgetting_factor_line(pd.DatetimeIndex([]), [], grid, 0.9)


def test_forgetting_factor_lines_stacks_one_row_per_lambda():
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
    grid = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    lines = forgetting_factor_lines(dates, [1.0, 1.2, 1.1], grid, DEFAULT_LAMBDAS)
    assert lines.shape == (len(DEFAULT_LAMBDAS), len(grid))


def test_cycle_measurements_keeps_the_same_dataset_id_up_to_the_anchor_in_date_order():
    catalog = pd.DataFrame({
        "dataset_id": ["A", "A", "A", "B", "A"],
        "measurement_date": ["2024-03-01", "2024-01-01", "2024-05-01", "2024-02-01", "2024-02-01"],
        "current_acc_z_max": [1.3, 1.1, 9.9, 7.7, np.nan],
    })
    frame = cycle_measurements(catalog, "A", "2024-03-01")
    assert frame["measurement_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-03-01"]
    assert frame["value"].tolist() == [1.1, 1.3]
