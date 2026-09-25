"""train / validation / inference 分割のリーク検証(SPEC.md 成果物4 `test_split_leakage.py`、実装順序案ステップ13)。

個々の部品(検索・分割・正規化)の単体テストは各テストファイルにあるが、ここでは
`prepare_datasets`で**実際に組み立てた3splitの成果物そのもの**を検査する。
"""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from synthetic_artifacts import build_artifacts

from acceleration_forecasting_v2.datasets.normalization import Normalization
from acceleration_forecasting_v2.datasets.prepare import GUIDE_SOURCE_SPLITS, SPLITS, prepare_datasets
from acceleration_forecasting_v2.datasets.verify import verify_dataset_leakage
from acceleration_forecasting_v2.retrieval import autoencoder_training


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("leak")
    artifact_dir, dataset_dir = root / "artifacts", root / "dataset"
    manifest = build_artifacts(artifact_dir, n_datasets=12, months=22, seed=0)
    summary = prepare_datasets(artifact_dir, dataset_dir, device="cpu")
    metadata = {split: pd.read_csv(dataset_dir / split / "metadata.csv", encoding="utf-8-sig") for split in SPLITS}
    return artifact_dir, dataset_dir, manifest, metadata, summary


def test_all_three_splits_are_built(prepared):
    _, dataset_dir, _, metadata, summary = prepared
    assert set(summary["splits"]) == set(SPLITS)
    for split in SPLITS:
        assert len(metadata[split]) > 0
        assert (dataset_dir / split / "target_values.npy").is_file()


def test_dataset_id_appears_in_exactly_one_split(prepared):
    _, _, manifest, metadata, _ = prepared
    sets = {split: set(metadata[split]["dataset_id"]) for split in SPLITS}
    assert sets["model_train"].isdisjoint(sets["model_validation"])
    assert sets["model_train"].isdisjoint(sets["inference"])
    assert sets["model_validation"].isdisjoint(sets["inference"])
    # 各splitに入っているdataset_idは、manifestで割り当てられたsplitと一致する。
    assigned = manifest.drop_duplicates("dataset_id").set_index("dataset_id")["model_split"]
    for split, ids in sets.items():
        assert all(assigned[dataset_id] == split for dataset_id in ids)


@pytest.mark.parametrize("split", SPLITS)
def test_guides_only_come_from_allowed_splits(prepared, split):
    metadata = prepared[3][split]
    used = set()
    for value in metadata["guide_model_splits"]:
        used.update(json.loads(value))
    assert used, "guideが1件も見つからないと検証にならない"
    assert used <= GUIDE_SOURCE_SPLITS[split]


def test_train_and_validation_guides_never_come_from_validation_or_inference(prepared):
    for split in ("model_train", "model_validation"):
        used = set()
        for value in prepared[3][split]["guide_model_splits"]:
            used.update(json.loads(value))
        assert "model_validation" not in used
        assert "inference" not in used


def test_inference_guides_never_come_from_other_inference_records(prepared):
    used = set()
    for value in prepared[3]["inference"]["guide_model_splits"]:
        used.update(json.loads(value))
    assert "inference" not in used


@pytest.mark.parametrize("split", SPLITS)
def test_guides_never_come_from_the_same_dataset_id(prepared, split):
    for _, row in prepared[3][split].iterrows():
        assert row["dataset_id"] not in json.loads(row["guide_dataset_ids"])


def test_guide_trend_ids_belong_to_the_recorded_dataset_and_split(prepared):
    _, _, manifest, metadata, _ = prepared
    by_trend = manifest.drop_duplicates("trend_id").set_index("trend_id")
    for split in SPLITS:
        for _, row in metadata[split].iterrows():
            for trend_id, dataset_id, guide_split in zip(json.loads(row["guide_trend_ids"]),
                                                         json.loads(row["guide_dataset_ids"]),
                                                         json.loads(row["guide_model_splits"])):
                assert by_trend.loc[trend_id, "dataset_id"] == dataset_id
                assert by_trend.loc[trend_id, "model_split"] == guide_split


def test_inference_records_are_absent_from_the_guide_database(prepared):
    artifact_dir, _, manifest, metadata, _ = prepared
    connection = sqlite3.connect(str(artifact_dir / "vector_database.sqlite"))
    splits = {row[0] for row in connection.execute("SELECT DISTINCT model_split FROM trends")}
    database_trends = {row[0] for row in connection.execute("SELECT trend_id FROM trends")}
    connection.close()
    assert splits == {"model_train", "model_validation"}
    assert database_trends.isdisjoint(set(metadata["inference"]["trend_id"]))


def test_segment_weights_sum_to_one_per_dataset_within_each_split(prepared):
    for split in SPLITS:
        frame = prepared[3][split]
        totals = frame.groupby("dataset_id")["segment_weight"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-6)


def test_normalization_statistics_use_only_model_train_targets(prepared):
    _, dataset_dir, _, _, _ = prepared
    target_norm = Normalization.load(dataset_dir / "target_normalization.json")
    values = np.load(dataset_dir / "model_train" / "target_values.npy")
    masks = np.load(dataset_dir / "model_train" / "target_masks.npy")
    train_values = values[masks > 0]
    assert target_norm.count == train_values.size
    assert target_norm.mean == pytest.approx(float(train_values.mean()), rel=1e-5)
    all_values = np.concatenate([
        np.load(dataset_dir / split / "target_values.npy")[np.load(dataset_dir / split / "target_masks.npy") > 0]
        for split in SPLITS
    ])
    assert target_norm.count < all_values.size  # 他splitの値は含まれていない


def test_verify_dataset_leakage_reports_no_violation_on_a_correct_build(prepared):
    _, dataset_dir, manifest, _, _ = prepared
    assert verify_dataset_leakage(dataset_dir, manifest) == []


def _tampered_copy(prepared, tmp_path):
    import shutil

    _, dataset_dir, manifest, _, _ = prepared
    copy = tmp_path / "dataset"
    shutil.copytree(dataset_dir, copy)
    return copy, manifest


def _rewrite_first_guide_field(dataset_dir, split, field, value):
    path = dataset_dir / split / "metadata.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    row = frame.index[frame["guide_count"] > 0][0]
    frame[field] = frame[field].astype(object)
    values = json.loads(frame.at[row, field])
    values[0] = value
    frame.at[row, field] = json.dumps(values)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def test_verify_detects_a_guide_from_a_forbidden_split(prepared, tmp_path):
    copy, manifest = _tampered_copy(prepared, tmp_path)
    _rewrite_first_guide_field(copy, "model_train", "guide_model_splits", "inference")
    assert any("許可されていないsplit" in text for text in verify_dataset_leakage(copy, manifest))


def test_verify_detects_a_guide_from_the_same_dataset(prepared, tmp_path):
    copy, manifest = _tampered_copy(prepared, tmp_path)
    path = copy / "model_train" / "metadata.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    row = frame.index[frame["guide_count"] > 0][0]
    values = json.loads(frame.at[row, "guide_dataset_ids"])
    values[0] = str(frame.at[row, "dataset_id"])
    frame.at[row, "guide_dataset_ids"] = json.dumps(values)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    assert any("同一dataset_id" in text for text in verify_dataset_leakage(copy, manifest))


def test_verify_detects_a_dataset_id_present_in_two_splits(prepared, tmp_path):
    copy, manifest = _tampered_copy(prepared, tmp_path)
    train_path, valid_path = copy / "model_train" / "metadata.csv", copy / "model_validation" / "metadata.csv"
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    valid = pd.read_csv(valid_path, encoding="utf-8-sig")
    valid.loc[0, "dataset_id"] = train.loc[0, "dataset_id"]
    valid.to_csv(valid_path, index=False, encoding="utf-8-sig")
    assert any("両方に存在する" in text for text in verify_dataset_leakage(copy, manifest))


def test_verify_detects_wrong_segment_weights_and_wrong_normalization(prepared, tmp_path):
    copy, manifest = _tampered_copy(prepared, tmp_path)
    path = copy / "model_train" / "metadata.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["segment_weight"] = 0.5
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    Normalization(mean=123.0, std=1.0, count=3, source="tampered").save(copy / "target_normalization.json")
    text = " ".join(verify_dataset_leakage(copy, manifest))
    assert "segment_weightの合計が1でない" in text
    assert "正規化統計" in text


def test_prepare_rejects_manifest_without_model_split(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    manifest = build_artifacts(artifact_dir, n_datasets=4, months=16)
    manifest.drop(columns=["model_split"]).to_csv(artifact_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="model_split"):
        prepare_datasets(artifact_dir, tmp_path / "dataset", device="cpu")


def test_max_datasets_per_split_limits_the_segments_used(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    build_artifacts(artifact_dir, n_datasets=12, months=22)
    prepare_datasets(artifact_dir, tmp_path / "dataset", device="cpu", max_datasets_per_split=1)
    for split in ("model_train", "model_validation"):
        frame = pd.read_csv(tmp_path / "dataset" / split / "metadata.csv", encoding="utf-8-sig")
        assert frame["dataset_id"].nunique() == 1


# --- Autoencoder(retrieval側)のリーク検証 -------------------------------------------

def test_autoencoder_training_never_touches_inference_waveforms(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    manifest = build_artifacts(artifact_dir, n_datasets=12, months=6, inference_offset=1000.0)
    seen = []
    original = autoencoder_training.MemmapWaveformDataset

    class Spy(original):
        def __init__(self, waveform_path, total_count, indices, mean, std):
            seen.append(np.asarray(indices).copy())
            super().__init__(waveform_path, total_count, indices, mean, std)

    monkeypatch.setattr(autoencoder_training, "MemmapWaveformDataset", Spy)
    autoencoder_training.train_autoencoder(
        artifact_dir / "split_manifest.csv", artifact_dir / "waveforms.bin", tmp_path / "ae",
        device="cpu", epochs=1, batch_size=8, embedding_dim=8,
    )
    inference_indices = set(manifest.loc[manifest["model_split"] == "inference", "waveform_index"])
    train_indices = set(manifest.loc[manifest["model_split"] == "model_train", "waveform_index"])
    assert seen, "MemmapWaveformDatasetが使われていない"
    for indices in seen:
        assert set(indices.tolist()).isdisjoint(inference_indices)
    normalization = json.loads((tmp_path / "ae" / "normalization.json").read_text(encoding="utf-8"))
    assert normalization["fitted_record_count"] == len(train_indices)
    assert normalization["split"] == "model_train"
    # inference波形には+1000を足してある。それが統計に混ざっていれば平均は大きくずれる。
    assert abs(normalization["mean"]) < 1.0


# --- 推論セットの間引き(評価用セット) -----------------------------------------------

def test_thin_evenly_keeps_everything_when_under_the_limit_or_unlimited():
    from acceleration_forecasting_v2.datasets.prepare import thin_evenly
    items = list(range(7))
    assert thin_evenly(items, None) == items
    assert thin_evenly(items, 7) == items
    assert thin_evenly(items, 20) == items


def test_thin_evenly_is_ordered_deterministic_and_spans_the_whole_range():
    from acceleration_forecasting_v2.datasets.prepare import thin_evenly
    items = list(range(100))
    picked = thin_evenly(items, 10)
    assert picked == thin_evenly(items, 10)
    assert picked == sorted(picked) and len(picked) == 10
    assert picked[0] == 0 and picked[-1] == 99
    gaps = np.diff(picked)
    assert gaps.max() - gaps.min() <= 1  # ほぼ等間隔


def test_inference_thinning_limits_records_per_segment_to_evaluable_ones(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    build_artifacts(artifact_dir, n_datasets=12, months=22)
    full = prepare_datasets(artifact_dir, tmp_path / "full", device="cpu")
    thin = prepare_datasets(artifact_dir, tmp_path / "thin", device="cpu", inference_max_per_segment=2)
    frame = pd.read_csv(tmp_path / "thin" / "inference" / "metadata.csv", encoding="utf-8-sig")
    assert frame.groupby("dataset_id").size().max() <= 2
    assert (frame["valid_future_months"] >= 8).all()  # 評価に使えるanchorだけが残る
    full_frame = pd.read_csv(tmp_path / "full" / "inference" / "metadata.csv", encoding="utf-8-sig")
    assert (full_frame["valid_future_months"] < 8).any()  # 間引き無しには正解の足りないanchorも含まれる
    assert thin["splits"]["inference"]["anchors"] < full["splits"]["inference"]["anchors"]
    # trainとvalidationは間引きの影響を受けない。
    for split in ("model_train", "model_validation"):
        assert thin["splits"][split] == full["splits"][split]
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv", encoding="utf-8-sig")
    assert verify_dataset_leakage(tmp_path / "thin", manifest) == []
