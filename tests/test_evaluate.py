"""evaluation.evaluate の単体テスト(evaluate_guide_fidelity、生成データとguideとの誤差)。"""

import numpy as np
import pandas as pd
import pytest

from acceleration_forecasting_v2.evaluation.evaluate import evaluate_guide_fidelity


def _write_guide_fidelity_fixture(tmp_path):
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "inference"
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame({"trend_id": ["t0", "t1", "t2"], "dataset_id": ["d0", "d1", "d2"]})
    metadata.to_csv(split_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    n = 3
    guide_values = np.zeros((n, 3, 12), dtype=np.float32)
    guide_masks = np.zeros((n, 3, 12), dtype=np.float32)
    retrieval_masks = np.zeros((n, 3), dtype=np.float32)

    # t0: guideスロット0のみ有効、全月有効、値1.0。
    guide_values[0, 0] = 1.0
    guide_masks[0, 0] = 1.0
    retrieval_masks[0, 0] = 1.0

    # t1: guideスロット0・1が有効(値1.0/3.0、全月有効)。
    guide_values[1, 0] = 1.0
    guide_masks[1, 0] = 1.0
    retrieval_masks[1, 0] = 1.0
    guide_values[1, 1] = 3.0
    guide_masks[1, 1] = 1.0
    retrieval_masks[1, 1] = 1.0

    # t2: 有効なguideが1本も無い(評価対象から除外されるべき)。

    # 正解(target): t0は2.0、t1は1.5(全月有効) — generated_vs_guideとは別の値にして、
    # guide_vs_target_MAEが独立に正しく計算されていることを検証できるようにする。
    target_values = np.zeros((n, 12), dtype=np.float32)
    target_masks = np.ones((n, 12), dtype=np.float32)
    target_values[0] = 2.0
    target_values[1] = 1.5

    np.save(split_dir / "guide_values.npy", guide_values)
    np.save(split_dir / "guide_masks.npy", guide_masks)
    np.save(split_dir / "retrieval_masks.npy", retrieval_masks)
    np.save(split_dir / "target_values.npy", target_values)
    np.save(split_dir / "target_masks.npy", target_masks)

    prediction_dir = tmp_path / "pred"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    samples = np.zeros((n, 5, 12), dtype=np.float32)
    samples[0] = 1.5  # t0: |1.5-1.0| = 0.5
    samples[1] = 2.0  # t1: |2.0-1.0|=1.0, |2.0-3.0|=1.0 → 平均1.0
    samples[2] = 9.9  # t2は除外されるため値は無関係
    np.savez_compressed(prediction_dir / "samples.npz",
                        trend_ids=np.array(["t0", "t1", "t2"], dtype="U"), samples=samples)
    return dataset_dir, prediction_dir


def test_evaluate_guide_fidelity_averages_across_valid_guide_slots(tmp_path):
    dataset_dir, prediction_dir = _write_guide_fidelity_fixture(tmp_path)
    summary = evaluate_guide_fidelity(dataset_dir, prediction_dir, tmp_path / "out", split="inference")

    assert summary["record_count"] == 2
    assert summary["skipped_no_valid_guide"] == 1
    assert summary["generated_vs_guide_MAE"]["mean"] == pytest.approx(0.75)
    # guide_vs_target: t0=|1.0-2.0|=1.0, t1=mean(|1.0-1.5|,|3.0-1.5|)=mean(0.5,1.5)=1.0 → 平均1.0。
    assert summary["guide_vs_target_MAE"]["mean"] == pytest.approx(1.0)

    frame = pd.read_csv(tmp_path / "out" / "generated_vs_guide_fidelity_per_record.csv", encoding="utf-8-sig")
    assert set(frame["trend_id"].astype(str)) == {"t0", "t1"}
    row_t0 = frame.loc[frame["trend_id"].astype(str) == "t0"].iloc[0]
    assert row_t0["generated_vs_guide_MAE"] == pytest.approx(0.5)
    assert row_t0["guide_slot_count"] == 1
    assert row_t0["guide_vs_target_MAE"] == pytest.approx(1.0)
    row_t1 = frame.loc[frame["trend_id"].astype(str) == "t1"].iloc[0]
    assert row_t1["generated_vs_guide_MAE"] == pytest.approx(1.0)
    assert row_t1["guide_slot_count"] == 2
    assert row_t1["guide_vs_target_MAE"] == pytest.approx(1.0)


def test_evaluate_guide_fidelity_raises_when_no_record_has_a_valid_guide(tmp_path):
    dataset_dir, prediction_dir = _write_guide_fidelity_fixture(tmp_path)
    np.save(dataset_dir / "inference" / "retrieval_masks.npy", np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        evaluate_guide_fidelity(dataset_dir, prediction_dir, tmp_path / "out2", split="inference")


def test_evaluate_guide_fidelity_only_averages_over_valid_guide_months(tmp_path):
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "inference"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trend_id": ["t0"], "dataset_id": ["d0"]}).to_csv(
        split_dir / "metadata.csv", index=False, encoding="utf-8-sig",
    )

    guide_values = np.zeros((1, 3, 12), dtype=np.float32)
    guide_masks = np.zeros((1, 3, 12), dtype=np.float32)
    retrieval_masks = np.zeros((1, 3), dtype=np.float32)
    guide_values[0, 0, :6] = 1.0
    guide_masks[0, 0, :6] = 1.0
    retrieval_masks[0, 0] = 1.0
    np.save(split_dir / "guide_values.npy", guide_values)
    np.save(split_dir / "guide_masks.npy", guide_masks)
    np.save(split_dir / "retrieval_masks.npy", retrieval_masks)

    # target_maskはguide_maskと重ならない後半だけ有効にする
    # (guide_vs_target_MAEは「guide_mask AND target_mask」の交差が空なので対象外になるはず)。
    target_values = np.full((1, 12), 9.0, dtype=np.float32)
    target_masks = np.zeros((1, 12), dtype=np.float32)
    target_masks[0, 6:] = 1.0
    np.save(split_dir / "target_values.npy", target_values)
    np.save(split_dir / "target_masks.npy", target_masks)

    prediction_dir = tmp_path / "pred"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    samples = np.zeros((1, 1, 12), dtype=np.float32)
    samples[0, 0, :6] = 1.5
    samples[0, 0, 6:] = 100.0  # 無効な後半、無視されるべき
    np.savez_compressed(prediction_dir / "samples.npz", trend_ids=np.array(["t0"], dtype="U"), samples=samples)

    summary = evaluate_guide_fidelity(dataset_dir, prediction_dir, tmp_path / "out3", split="inference")
    assert summary["generated_vs_guide_MAE"]["mean"] == pytest.approx(0.5)

    frame = pd.read_csv(tmp_path / "out3" / "generated_vs_guide_fidelity_per_record.csv", encoding="utf-8-sig")
    # guide_mask(前半)とtarget_mask(後半)が重ならないため、guide_vs_target_MAEは欠損(NaN)になる。
    assert pd.isna(frame.iloc[0]["guide_vs_target_MAE"])
