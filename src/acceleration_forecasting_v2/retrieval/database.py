"""検索用SQLite DBのスキーマと入出力。

旧実装は2段階(`acceleration_retrieval/database.py` でmodel_split無しのDBを作り、
`acceleration_forecasting_12m/.../retrieval/database.py` でmodel_split付きに
作り直す)だったが、本リポジトリはSPEC.md 2.1の通り1段階で完結させる
(`trends`テーブルに最初から`model_split`を持たせる)。SCHEMA自体は後者を踏襲。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE trends(
    trend_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    model_split TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    bin_start_m REAL NOT NULL,
    bin_end_m REAL NOT NULL,
    current_acc_z_max REAL NOT NULL,
    future_values TEXT NOT NULL,
    future_mask TEXT NOT NULL,
    selected_dates TEXT NOT NULL,
    guide_available_date TEXT NOT NULL,
    cutoff_maintenance_date TEXT,
    maintenance_type TEXT,
    maintenance_description TEXT
);
CREATE TABLE waveform_records(
    record_id TEXT PRIMARY KEY,
    measurement_id TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    bin_start_m REAL NOT NULL,
    bin_end_m REAL NOT NULL,
    mean_velocity_kmh REAL NOT NULL,
    source_csv_path TEXT NOT NULL,
    waveform_sha256 TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    trend_id TEXT NOT NULL REFERENCES trends(trend_id),
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL
);
CREATE INDEX idx_trends_split ON trends(model_split);
CREATE INDEX idx_trends_dataset ON trends(dataset_id);
CREATE INDEX idx_waveform_trend ON waveform_records(trend_id);
CREATE INDEX idx_waveform_date ON waveform_records(measurement_date);
"""

TREND_COLUMNS = [
    "trend_id",
    "dataset_id",
    "model_split",
    "measurement_date",
    "direction",
    "bin_start_m",
    "bin_end_m",
    "current_acc_z_max",
    "future_values",
    "future_mask",
    "selected_dates",
    "guide_available_date",
    "cutoff_maintenance_date",
    "maintenance_type",
    "maintenance_description",
]


def connect(path):
    return sqlite3.connect(str(path))


def connect_read_only(path):
    path = Path(path).resolve()
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def initialize_database(path, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    connection = connect(path)
    connection.executescript(SCHEMA)
    return connection


def insert_trends(connection, trend_frame):
    """`trends`にレコードを一括挿入する。`trend_frame`は`model_split`列を持つ必要がある。

    `guide_available_date`が無ければ、anchor月+FORECAST_MONTHSか月+15日(旧実装の
    18か月版と同じ余裕日数の考え方)を仮の利用可能日として補う。
    """
    trend_frame = trend_frame.copy()
    if "guide_available_date" not in trend_frame.columns:
        from .constants import FUTURE_MONTHS

        anchor_month = pd.to_datetime(trend_frame["measurement_date"], errors="raise").dt.to_period("M").dt.to_timestamp()
        trend_frame["guide_available_date"] = (
            anchor_month + pd.DateOffset(months=FUTURE_MONTHS) + pd.Timedelta(days=15)
        ).dt.strftime("%Y-%m-%d")
    connection.executemany(
        f"INSERT OR REPLACE INTO trends ({','.join(TREND_COLUMNS)}) VALUES ({','.join('?' for _ in TREND_COLUMNS)})",
        [tuple(None if pd.isna(row[column]) else row[column] for column in TREND_COLUMNS) for _, row in trend_frame.iterrows()],
    )


def insert_waveform_record(connection, row, embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    connection.execute(
        """
        INSERT INTO waveform_records (
            record_id, measurement_id, measurement_date, direction,
            bin_start_m, bin_end_m, mean_velocity_kmh, source_csv_path,
            waveform_sha256, dataset_id, trend_id, embedding, embedding_dim
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["record_id"],
            row["measurement_id"],
            row["measurement_date"],
            row["direction"],
            float(row["bin_start_m"]),
            float(row["bin_end_m"]),
            float(row["mean_velocity_kmh"]),
            row["source_csv_path"],
            row["waveform_sha256"],
            row["dataset_id"],
            row["trend_id"],
            embedding.tobytes(order="C"),
            int(embedding.size),
        ),
    )


def store_metadata(connection, values):
    connection.executemany(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        [
            (str(key), json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value)
            for key, value in values.items()
        ],
    )
