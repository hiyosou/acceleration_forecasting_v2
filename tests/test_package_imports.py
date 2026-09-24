"""Scaffold smoke test: SPEC.md 実装順序案 #1 の完了判定基準に対応する。"""

import importlib


PACKAGE_AND_SUBPACKAGES = (
    "acceleration_forecasting_v2",
    "acceleration_forecasting_v2.common",
    "acceleration_forecasting_v2.common.constants",
    "acceleration_forecasting_v2.retrieval",
    "acceleration_forecasting_v2.datasets",
    "acceleration_forecasting_v2.diffusion",
    "acceleration_forecasting_v2.models",
    "acceleration_forecasting_v2.training",
    "acceleration_forecasting_v2.inference",
    "acceleration_forecasting_v2.evaluation",
)


def test_package_and_subpackages_import_without_error():
    for module_name in PACKAGE_AND_SUBPACKAGES:
        importlib.import_module(module_name)


def test_constants_match_spec():
    from acceleration_forecasting_v2.common import constants

    assert constants.HISTORY_MONTHS == 6
    assert constants.FORECAST_MONTHS == 12
    assert constants.MIN_TARGET_MONTHS == 8
    assert constants.MIN_GUIDE_MONTHS == 8
    assert constants.TOP_K == 3
