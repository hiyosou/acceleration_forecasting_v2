"""retrieval.waveforms の単体テスト。"""

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.retrieval.waveforms import (
    extract_waveform,
    group_frame_by_bins,
    parse_measurement,
)
from acceleration_forecasting_v2.retrieval.constants import SAMPLES_PER_BIN, STEP_M


def _make_bin_frame(bin_start_m, samples=SAMPLES_PER_BIN, step_m=STEP_M, velocity=60.0, acc=0.5):
    distances = bin_start_m + np.arange(samples) * step_m
    return pd.DataFrame({
        "distance_corrected[m]": distances,
        "acc_z[m/s2]": np.full(samples, acc, dtype=float),
        "velocity[km/h]": np.full(samples, velocity, dtype=float),
    })


def test_extract_waveform_valid_segment_returns_waveform_and_velocity():
    frame = _make_bin_frame(2000.0)
    extracted, reason = extract_waveform(frame, 2000.0, 2100.0)
    assert reason is None
    assert extracted["waveform"].shape == (SAMPLES_PER_BIN,)
    assert extracted["waveform"].dtype == np.float32
    assert extracted["mean_velocity_kmh"] == 60.0


def test_extract_waveform_wrong_sample_count_is_rejected():
    frame = _make_bin_frame(2000.0, samples=SAMPLES_PER_BIN - 1)
    extracted, reason = extract_waveform(frame, 2000.0, 2100.0)
    assert extracted is None
    assert reason == "sample_count"


def test_extract_waveform_speed_out_of_range_is_rejected():
    frame = _make_bin_frame(2000.0, velocity=10.0)
    extracted, reason = extract_waveform(frame, 2000.0, 2100.0)
    assert extracted is None
    assert reason == "mean_speed_out_of_range"


def test_extract_waveform_nonfinite_acceleration_is_rejected():
    frame = _make_bin_frame(2000.0)
    frame.loc[0, "acc_z[m/s2]"] = np.nan
    extracted, reason = extract_waveform(frame, 2000.0, 2100.0)
    assert extracted is None
    assert reason == "waveform_nonfinite"


def test_extract_waveform_missing_columns_reports_which_are_missing():
    frame = _make_bin_frame(2000.0).drop(columns=["velocity[km/h]"])
    extracted, reason = extract_waveform(frame, 2000.0, 2100.0)
    assert extracted is None
    assert reason == "missing_columns:velocity[km/h]"


def test_parse_measurement_valid_filename():
    result = parse_measurement("NAGANO2_20230115_120000_U_1.csv")
    assert result == {
        "measurement_id": "20230115_120000_U_1",
        "measurement_date": "2023-01-15",
        "direction": "U",
    }


def test_parse_measurement_invalid_filename_returns_none():
    assert parse_measurement("not_a_measurement.csv") is None


def test_group_frame_by_bins_assigns_correct_bin_index():
    frame = pd.concat([_make_bin_frame(2000.0), _make_bin_frame(2100.0)], ignore_index=True)
    grouped = group_frame_by_bins(frame, 2000.0, 2200.0, 100.0)
    assert set(grouped.keys()) == {0, 1}
    assert len(grouped[0]) == SAMPLES_PER_BIN
    assert len(grouped[1]) == SAMPLES_PER_BIN
    assert grouped[0]["distance_corrected[m]"].min() >= 2000.0
    assert grouped[1]["distance_corrected[m]"].min() >= 2100.0
