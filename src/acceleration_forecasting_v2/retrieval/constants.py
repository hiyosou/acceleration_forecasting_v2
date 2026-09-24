from pathlib import Path

from acceleration_forecasting_v2.common.constants import (
    DEFAULT_ARTIFACTS,
    EMBEDDING_DIM,
    FORECAST_MONTHS,
)

ACCELERATION_COLUMN = "acc_z[m/s2]"
VELOCITY_COLUMN = "velocity[km/h]"
DISTANCE_COLUMN = "distance_corrected[m]"

# ルートパイプラインの出力を直接読む(acceleration_retrieval の成果物には依存しない)。
DEFAULT_WAVEFORM_DIR = Path(r"D:\railwaydata\方向別振動データ_位置補正_0.2m")
DEFAULT_TREND_DIR = Path(
    r"D:\railwaydata\方向別振動データ_位置補正_0.2m_100m最大値_datasets"
)
DEFAULT_ARTIFACT_DIR = DEFAULT_ARTIFACTS / "retrieval"

START_M = 2000.0
END_M = 33000.0
BIN_WIDTH_M = 100.0
STEP_M = 0.2
SAMPLES_PER_BIN = 500

MIN_MEAN_SPEED_KMH = 50.0
MAX_MEAN_SPEED_KMH = 75.0
# 旧 acceleration_retrieval は18か月分(姉妹プロジェクト向け)を保持していたが、
# 本リポジトリは12か月固定の出力ホライズンしか使わないため FORECAST_MONTHS に合わせる。
FUTURE_MONTHS = FORECAST_MONTHS
SEARCH_DAYS = 15
EMBEDDING_DIM = EMBEDDING_DIM

# 2段階 dataset_id 単位分割(SPEC.md 1.4 / 2.3 で踏襲を確定した比率)。
DEVELOPMENT_RATIO = 0.8
MODEL_TRAIN_RATIO_WITHIN_DEVELOPMENT = 0.9
RANDOM_SEED = 42
