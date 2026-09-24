from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS = REPOSITORY_ROOT / "artifacts"

# SPEC.md 成果物2 / 前提の要件定義表に対応する確定値。
# 6か月統一履歴(今月+過去5か月)、出力ホライズンは変更なし12か月固定。
HISTORY_MONTHS = 6
MIN_HISTORY_MONTHS = 3
FORECAST_MONTHS = 12
MIN_TARGET_MONTHS = 8
MIN_GUIDE_MONTHS = 8
EMBEDDING_DIM = 256
TOP_K = 3
PHYSICAL_MIN = 0.1
PHYSICAL_MAX = 6.0
