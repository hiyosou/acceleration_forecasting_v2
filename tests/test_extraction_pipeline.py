"""extract_manifest_and_trends のend-to-end統合テスト(小規模フィクスチャ)。

SPEC.md 成果物4 で挙げた「新旧retrieval出力差分の検証」の意図を、実データの
D:\\railwaydata を使わずに検証できる最小構成で満たす。CSV読み込み→100m区間の
波形抽出→trend_catalogとの突き合わせ→manifest/waveforms.bin/trend_catalog.csv
書き出しまでを一気通貫で確認する。
"""

import numpy as np
import pandas as pd

from acceleration_forecasting_v2.retrieval.constants import BIN_WIDTH_M, SAMPLES_PER_BIN, START_M, STEP_M
from acceleration_forecasting_v2.retrieval.waveforms import extract_manifest_and_trends, open_waveforms


BASE_COLUMNS = [
    "updown", "seq", "distance[m]", "velocity[km/h]",
    "acc_x[m/s2]", "acc_y[m/s2]", "acc_z[m/s2]",
    "gyro_x[°/sec]", "gyro_y[°/sec]", "gyro_z[°/sec]",
]


def _write_waveform_csv(path, *, distance_start=START_M, acc_z_value=0.4, velocity=60.0):
    distances = distance_start + np.arange(SAMPLES_PER_BIN) * STEP_M
    frame = pd.DataFrame({
        "updown": ["U"] * SAMPLES_PER_BIN,
        "seq": np.arange(SAMPLES_PER_BIN),
        "distance[m]": distances,
        "velocity[km/h]": np.full(SAMPLES_PER_BIN, velocity),
        "acc_x[m/s2]": np.zeros(SAMPLES_PER_BIN),
        "acc_y[m/s2]": np.zeros(SAMPLES_PER_BIN),
        "acc_z[m/s2]": np.full(SAMPLES_PER_BIN, acc_z_value),
        "gyro_x[°/sec]": np.zeros(SAMPLES_PER_BIN),
        "gyro_y[°/sec]": np.zeros(SAMPLES_PER_BIN),
        "gyro_z[°/sec]": np.zeros(SAMPLES_PER_BIN),
        "distance_corrected[m]": distances,
    })
    frame.to_csv(path, header=False, index=False, encoding="cp932")


def _write_trend_csv(path, *, dataset_id="ds1", date="2023-01-15", acc_z_max=0.4, segment_index=1):
    pd.DataFrame([{
        "dataset_id": dataset_id, "direction": "U", "bin_start_m": START_M, "bin_end_m": START_M + 100.0,
        "segment_index": segment_index, "measurement_date": date, "acc_z_max": acc_z_max,
        "measured_at": date, "previous_maintenance_date": "", "maintenance_type": "", "maintenance_description": "",
    }]).to_csv(path, index=False, encoding="utf-8-sig")


def test_extract_manifest_and_trends_end_to_end(tmp_path):
    waveform_dir = tmp_path / "waveforms"
    trend_dir = tmp_path / "trends"
    artifact_dir = tmp_path / "artifacts"
    waveform_dir.mkdir()
    trend_dir.mkdir()

    _write_waveform_csv(waveform_dir / "NAGANO2_20230115_120000_U_1.csv")
    _write_trend_csv(trend_dir / "trend_ds1.csv")

    # end_m を1区間分に絞り、実データにある残り308区間ぶんの「データが無いだけの
    # sample_count除外」がテストの主張をノイズで埋もれさせないようにする。
    summary = extract_manifest_and_trends(
        waveform_dir, trend_dir, artifact_dir, end_m=START_M + BIN_WIDTH_M, progress=False,
    )

    assert summary["eligible_records"] == 1
    assert summary["dataset_count"] == 1
    assert summary["excluded"] == 0

    manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig")
    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["dataset_id"] == "ds1"
    assert row["direction"] == "U"
    assert row["measurement_date"] == "2023-01-15"
    assert row["bin_start_m"] == START_M

    trends = pd.read_csv(artifact_dir / "trend_catalog.csv", encoding="utf-8-sig")
    assert len(trends) == 1
    assert trends.iloc[0]["trend_id"] == row["trend_id"]
    assert trends.iloc[0]["current_acc_z_max"] == 0.4

    waveforms = open_waveforms(artifact_dir / "waveforms.bin", len(manifest))
    assert waveforms.shape == (1, SAMPLES_PER_BIN)
    np.testing.assert_allclose(waveforms[0], 0.4, atol=1e-6)


def test_extract_manifest_and_trends_excludes_out_of_range_speed(tmp_path):
    waveform_dir = tmp_path / "waveforms"
    trend_dir = tmp_path / "trends"
    artifact_dir = tmp_path / "artifacts"
    waveform_dir.mkdir()
    trend_dir.mkdir()

    _write_waveform_csv(waveform_dir / "NAGANO2_20230115_120000_U_1.csv", velocity=10.0)
    _write_trend_csv(trend_dir / "trend_ds1.csv")

    summary = extract_manifest_and_trends(
        waveform_dir, trend_dir, artifact_dir, end_m=START_M + BIN_WIDTH_M, progress=False,
    )

    assert summary["eligible_records"] == 0
    assert summary["excluded"] == 1
    diagnostics = pd.read_csv(artifact_dir / "exclusion_diagnostics.csv", encoding="utf-8-sig")
    assert diagnostics.iloc[0]["reason"] == "mean_speed_out_of_range"


def test_extract_manifest_and_trends_excludes_when_no_matching_trend(tmp_path):
    waveform_dir = tmp_path / "waveforms"
    trend_dir = tmp_path / "trends"
    artifact_dir = tmp_path / "artifacts"
    waveform_dir.mkdir()
    trend_dir.mkdir()

    _write_waveform_csv(waveform_dir / "NAGANO2_20230115_120000_U_1.csv")
    # trend_dir を空のままにする(=anchorが見つからない)。

    summary = extract_manifest_and_trends(
        waveform_dir, trend_dir, artifact_dir, end_m=START_M + BIN_WIDTH_M, progress=False,
    )

    assert summary["eligible_records"] == 0
    assert summary["excluded"] == 1
    diagnostics = pd.read_csv(artifact_dir / "exclusion_diagnostics.csv", encoding="utf-8-sig")
    assert diagnostics.iloc[0]["reason"] == "trend_not_found"
