"""inference.self_guide_diagnostics の検証(SELF_GUIDE_CHECK.md ステップ3)。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from test_training import _raw_split, _train_kwargs

from acceleration_forecasting_v2.datasets.build import write_dataset
from acceleration_forecasting_v2.diffusion.process import DiffusionProcess
from acceleration_forecasting_v2.inference.self_guide_diagnostics import (
    _diagnose_with_process,
    _disable_guide,
    compute_shuffled_partner_indices,
    diagnose_guide_conditioning,
)
from acceleration_forecasting_v2.training.train import train


def _raw_split_with_datasets(n, *, seed=0, n_datasets=4, **kwargs):
    """`test_training._raw_split`にdataset_id列を追加する(shuffled相手選びに必要)。"""
    raw = _raw_split(n, seed=seed, **kwargs)
    metadata = raw.metadata.copy()
    metadata["dataset_id"] = [f"seg{index % n_datasets:02d}" for index in range(n)]
    return replace(raw, metadata=metadata)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("self_guide")
    dataset_dir = root / "dataset"
    write_dataset({
        "model_train": _raw_split_with_datasets(32, seed=1, n_datasets=6),
        "model_validation": _raw_split_with_datasets(16, seed=2, n_datasets=4),
    }, dataset_dir)
    train(dataset_dir, root / "run", **_train_kwargs(epochs=2))
    return dataset_dir, root / "run" / "best_model.pt"


# --- compute_shuffled_partner_indices --------------------------------------------

def test_shuffled_partner_always_has_a_different_dataset_id():
    dataset_ids = [f"seg{index % 4:02d}" for index in range(20)]
    partners = compute_shuffled_partner_indices(dataset_ids)
    for i, partner in enumerate(partners):
        assert dataset_ids[partner] != dataset_ids[i]


def test_shuffled_partner_handles_one_large_group_and_one_singleton():
    # 1件だけ別dataset_id(末尾)、残り全部同じ: 大半のレコードは末尾まで循環して辿り着く必要がある。
    dataset_ids = ["A"] * 9 + ["B"]
    partners = compute_shuffled_partner_indices(dataset_ids)
    assert all(dataset_ids[partner] == "B" for partner in partners[:9])
    assert dataset_ids[partners[9]] == "A"


def test_shuffled_partner_rejects_a_single_dataset_id():
    with pytest.raises(ValueError, match="dataset_id"):
        compute_shuffled_partner_indices(["A"] * 5)


# --- _disable_guide ---------------------------------------------------------------

def test_disable_guide_zeroes_only_the_mask_fields():
    batch = {
        "guide_values": torch.ones(2, 3, 12), "guide_mask": torch.ones(2, 3, 12),
        "retrieval_mask": torch.ones(2, 3), "history_values": torch.ones(2, 6),
    }
    disabled = _disable_guide(batch)
    assert torch.count_nonzero(disabled["guide_mask"]) == 0
    assert torch.count_nonzero(disabled["retrieval_mask"]) == 0
    # guide_values自体やguide以外のフィールドは変更しない。
    assert torch.equal(disabled["guide_values"], batch["guide_values"])
    assert torch.equal(disabled["history_values"], batch["history_values"])
    assert disabled is not batch  # 元のbatchを書き換えていない
    assert torch.count_nonzero(batch["guide_mask"]) == 2 * 3 * 12


# --- 出力の形状・複数タイムステップ --------------------------------------------------

def test_output_shape_and_columns_match_the_split_record_count(trained, tmp_path):
    dataset_dir, checkpoint = trained
    result = diagnose_guide_conditioning(dataset_dir, checkpoint, tmp_path / "out", split="model_validation",
                                        timesteps=(900,), device="cpu", batch_size=4)
    assert result["record_count"] == 16
    frame = pd.read_csv(tmp_path / "out" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    assert len(frame) == 16
    expected_columns = {
        "trend_id", "dataset_id", "normal_MAE", "shuffled_MAE", "disabled_MAE",
        "guide_advantage_vs_shuffled", "guide_advantage_vs_disabled",
        "normal_vs_shuffled_output_L1", "normal_vs_disabled_output_L1",
        "attention_rank_1", "attention_rank_2", "attention_rank_3", "reference_context_norm",
    }
    assert expected_columns <= set(frame.columns)
    assert np.isfinite(frame[list(expected_columns - {"trend_id", "dataset_id"})].to_numpy()).all()
    assert (tmp_path / "out" / "condition_usage_summary_t900.json").is_file()


def test_multiple_timesteps_produce_separate_outputs(trained, tmp_path):
    dataset_dir, checkpoint = trained
    result = diagnose_guide_conditioning(dataset_dir, checkpoint, tmp_path / "out", split="model_validation",
                                        timesteps=(100, 900), device="cpu", batch_size=4)
    assert result["timesteps"] == [100, 900]
    for timestep in (100, 900):
        assert (tmp_path / "out" / f"condition_usage_per_target_t{timestep}.csv").is_file()
        assert (tmp_path / "out" / f"condition_usage_summary_t{timestep}.json").is_file()
    frame_100 = pd.read_csv(tmp_path / "out" / "condition_usage_per_target_t100.csv", encoding="utf-8-sig")
    frame_900 = pd.read_csv(tmp_path / "out" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    # ノイズの少ないt=100と多いt=900では、少なくとも一部のレコードで数値が異なるはず。
    assert not np.allclose(frame_100["normal_MAE"].to_numpy(), frame_900["normal_MAE"].to_numpy())


def test_reference_context_norm_is_constant_within_a_batch_but_can_vary_across_batches(trained, tmp_path):
    dataset_dir, checkpoint = trained
    diagnose_guide_conditioning(dataset_dir, checkpoint, tmp_path / "out", split="model_validation",
                                timesteps=(900,), device="cpu", batch_size=4)
    frame = pd.read_csv(tmp_path / "out" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    # batch_size=4、16件なので4バッチ。各バッチ内では同一値になるはず。
    for start in range(0, 16, 4):
        chunk = frame["reference_context_norm"].iloc[start:start + 4]
        assert chunk.nunique() == 1


def test_determinism_across_runs_with_the_same_seed(trained, tmp_path):
    dataset_dir, checkpoint = trained
    diagnose_guide_conditioning(dataset_dir, checkpoint, tmp_path / "a", split="model_validation",
                                timesteps=(900,), device="cpu", batch_size=4, seed=7)
    diagnose_guide_conditioning(dataset_dir, checkpoint, tmp_path / "b", split="model_validation",
                                timesteps=(900,), device="cpu", batch_size=4, seed=7)
    a = pd.read_csv(tmp_path / "a" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    b = pd.read_csv(tmp_path / "b" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_the_injected_noise(trained, tmp_path):
    dataset_dir, checkpoint = trained
    diagnose_guide_conditioning(dataset_dir, checkpoint, tmp_path / "a", split="model_validation",
                                timesteps=(900,), device="cpu", batch_size=4, seed=1)
    diagnose_guide_conditioning(dataset_dir, checkpoint, tmp_path / "b", split="model_validation",
                                timesteps=(900,), device="cpu", batch_size=4, seed=2)
    a = pd.read_csv(tmp_path / "a" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    b = pd.read_csv(tmp_path / "b" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    assert not np.allclose(a["normal_MAE"].to_numpy(), b["normal_MAE"].to_numpy())


# --- 極端ケース(検出力): ダミーモデルを直接process経由で注入 ------------------------

class _GuideIgnoringModel(torch.nn.Module):
    """guide関連キーを一切参照しないダミーモデル。normal/shuffled/disabledで出力が完全一致するはず。"""

    def forward(self, noisy, timesteps, batch):
        return noisy * 0.0 + batch["history_values"].mean(dim=1, keepdim=True).expand(-1, 12) * 0.1


class _GuideOnlyModel(torch.nn.Module):
    """guide_valuesのマスク加重平均だけを返すダミーモデル。guide無効化で出力が大きく変わるはず。"""

    def forward(self, noisy, timesteps, batch):
        mask = batch["guide_mask"] * batch["retrieval_mask"].unsqueeze(-1)
        weighted_sum = (batch["guide_values"] * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1e-6)
        return weighted_sum / denominator


def _dataset_dir_for_dummy_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("self_guide_dummy")
    dataset_dir = root / "dataset"
    write_dataset({
        "model_train": _raw_split_with_datasets(8, seed=1, n_datasets=4),
        "model_validation": _raw_split_with_datasets(12, seed=2, n_datasets=4),
    }, dataset_dir)
    return dataset_dir


def test_a_model_that_ignores_guide_shows_zero_advantage(tmp_path_factory, tmp_path):
    dataset_dir = _dataset_dir_for_dummy_models(tmp_path_factory)
    process = DiffusionProcess(_GuideIgnoringModel(), steps=1000, prediction_type="epsilon")
    result = _diagnose_with_process(process, {"prediction_type": "epsilon"}, dataset_dir, tmp_path / "out",
                                    split="model_validation", timesteps=(900,), batch_size=4)
    frame = pd.read_csv(tmp_path / "out" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    assert (frame["normal_vs_shuffled_output_L1"] == 0.0).all()
    assert (frame["normal_vs_disabled_output_L1"] == 0.0).all()
    assert (frame["guide_advantage_vs_shuffled"] == 0.0).all()
    assert (frame["guide_advantage_vs_disabled"] == 0.0).all()
    summary = result["results"]["900"]
    assert summary["guide_advantage_vs_disabled"]["mean"] == pytest.approx(0.0)


def test_a_model_that_only_uses_guide_shows_a_large_disabled_gap(tmp_path_factory, tmp_path):
    dataset_dir = _dataset_dir_for_dummy_models(tmp_path_factory)
    process = DiffusionProcess(_GuideOnlyModel(), steps=1000, prediction_type="epsilon")
    result = _diagnose_with_process(process, {"prediction_type": "epsilon"}, dataset_dir, tmp_path / "out",
                                    split="model_validation", timesteps=(900,), batch_size=4)
    frame = pd.read_csv(tmp_path / "out" / "condition_usage_per_target_t900.csv", encoding="utf-8-sig")
    # guideを無効化するとguide_valuesの寄与が消え、出力がゼロ付近まで大きく変わるはず。
    assert (frame["normal_vs_disabled_output_L1"] > 0.5).all()
    assert frame["normal_vs_shuffled_output_L1"].mean() > 0.0
    summary = result["results"]["900"]
    assert summary["normal_vs_disabled_output_L1_mean"] > 0.5
    # attention_rank/reference_context_normはRMA専用のためNaN(ダミーモデルにはreference_blocksが無い)。
    assert frame["attention_rank_1"].isna().all()
    assert frame["reference_context_norm"].isna().all()
