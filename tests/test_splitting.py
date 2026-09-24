"""retrieval.splitting の単体テスト(SPEC.md 2.3 チェック項目1に対応)。"""

import pandas as pd
import pytest

from acceleration_forecasting_v2.retrieval.splitting import assign_model_split, split_summary


def _manifest(dataset_count=20, records_per_dataset=3):
    rows = []
    for dataset_index in range(dataset_count):
        for record_index in range(records_per_dataset):
            rows.append({"dataset_id": f"seg{dataset_index:03d}", "record_id": f"seg{dataset_index:03d}-{record_index}"})
    return pd.DataFrame(rows)


def test_assign_model_split_each_dataset_id_appears_in_exactly_one_split():
    manifest = assign_model_split(_manifest(), seed=1)
    per_dataset_splits = manifest.groupby("dataset_id")["model_split"].nunique()
    assert (per_dataset_splits == 1).all()


def test_assign_model_split_is_reproducible_with_same_seed():
    first = assign_model_split(_manifest(), seed=7)
    second = assign_model_split(_manifest(), seed=7)
    pd.testing.assert_series_equal(
        first.set_index("record_id")["model_split"].sort_index(),
        second.set_index("record_id")["model_split"].sort_index(),
    )


def test_assign_model_split_different_seed_can_change_assignment():
    first = assign_model_split(_manifest(dataset_count=30), seed=1)
    second = assign_model_split(_manifest(dataset_count=30), seed=2)
    first_map = first.drop_duplicates("dataset_id").set_index("dataset_id")["model_split"]
    second_map = second.drop_duplicates("dataset_id").set_index("dataset_id")["model_split"]
    assert not first_map.equals(second_map)


def test_assign_model_split_ratio_is_approximately_80_20_then_90_10():
    manifest = assign_model_split(_manifest(dataset_count=100), seed=1)
    dataset_splits = manifest.drop_duplicates("dataset_id")["model_split"]
    counts = dataset_splits.value_counts()
    # development_ratio=0.8 -> inference は概ね20件、development(80件)を9:1でtrain/valid。
    assert counts["inference"] == pytest.approx(20, abs=2)
    assert counts["model_train"] == pytest.approx(72, abs=3)
    assert counts["model_validation"] == pytest.approx(8, abs=3)


def test_assign_model_split_requires_at_least_two_dataset_ids():
    with pytest.raises(ValueError):
        assign_model_split(_manifest(dataset_count=1))


def test_split_summary_counts_match_manifest():
    manifest = assign_model_split(_manifest(dataset_count=10, records_per_dataset=2), seed=3)
    summary = split_summary(manifest)
    assert sum(summary["record_count"].values()) == len(manifest)
    assert sum(summary["dataset_count"].values()) == manifest["dataset_id"].nunique()
