"""生の振動計測CSVを読み込む処理。

`my_project/position_correction_common.py`(ルートパイプライン共通、パッケージ化
されていないプレーンスクリプト)の `load_vibration_csv` 相当を、`uv` 管理の
独立パッケージである本リポジトリに合わせてこのファイルへ複製したもの。

複製した理由と伴うトレードオフ: SPEC.md 成果物5 ステップ2の実装中に発見した
「SPEC.mdだけでは一意に決められない」論点として壁打ちで確認済み(選択肢1を採用)。
ルート側で `load_vibration_csv` の実装が変わった場合、本ファイルは自動追従しない。
"""

from __future__ import annotations

import pandas as pd


BASE_COLUMNS = [
    "updown",
    "seq",
    "distance[m]",
    "velocity[km/h]",
    "acc_x[m/s2]",
    "acc_y[m/s2]",
    "acc_z[m/s2]",
    "gyro_x[°/sec]",
    "gyro_y[°/sec]",
    "gyro_z[°/sec]",
]

OPTIONAL_COLUMNS = [
    "distance_corrected[m]",
    "position_correction_delta[m]",
]


def _read_csv_with_fallback(filepath):
    errors = []
    for encoding in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(filepath, header=None, encoding=encoding)
        except Exception as exc:  # noqa: BLE001 - フォールバック探索のため捕捉
            errors.append(exc)
    raise errors[-1]


def load_vibration_csv(filepath):
    """1本の振動計測CSVを読み込み、列名付きDataFrameを返す。

    ヘッダー行の有無を自動判定し(先頭行が数値でなければヘッダー行とみなして
    読み飛ばす)、`distance[m]`が欠測の行を除外する。読み込めない/空の場合は
    `None`を返す。
    """
    df = _read_csv_with_fallback(filepath)
    if df.empty:
        return None

    if isinstance(df.iloc[0, 2], str):
        probe = df.iloc[0, 2].replace(".", "", 1).replace("-", "", 1)
        if not probe.isdigit():
            df = df.iloc[1:].reset_index(drop=True)
    if df.empty or df.shape[1] < len(BASE_COLUMNS):
        return None

    column_names = list(BASE_COLUMNS)
    for index in range(df.shape[1] - len(BASE_COLUMNS)):
        if index < len(OPTIONAL_COLUMNS):
            column_names.append(OPTIONAL_COLUMNS[index])
        else:
            column_names.append(f"extra_col_{index + 1}")
    df = df.iloc[:, : len(column_names)].copy()
    df.columns = column_names

    numeric_columns = column_names[1:]
    df.loc[:, numeric_columns] = df.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")

    df = df.dropna(subset=["distance[m]"]).reset_index(drop=True)
    return df
