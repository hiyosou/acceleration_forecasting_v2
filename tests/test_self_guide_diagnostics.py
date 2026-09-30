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
    _BLOCK_LABELS,
    _BlockStatsAccumulator,
    _combine_block_breakdown,
    _combine_film_breakdown,
    _diagnose_with_process,
    _disable_guide,
    _merge_breakdowns,
    _per_block_diagnostic_stats,
    _per_block_film_stats,
    compute_shuffled_partner_indices,
    diagnose_guide_conditioning,
)
from acceleration_forecasting_v2.training.train import train


def _raw_split_with_datasets(n, *, seed=0, n_datasets=4, **kwargs):
    """`test_training._raw_split`にdataset_id列・current_acc_z_max列を追加する
    (それぞれshuffled相手選び・self_target構築に必要)。
    """
    raw = _raw_split(n, seed=seed, **kwargs)
    metadata = raw.metadata.copy()
    metadata["dataset_id"] = [f"seg{index % n_datasets:02d}" for index in range(n)]
    metadata["current_acc_z_max"] = raw.arrays["history_values"][:, -1].astype(np.float32)
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
        "trend_id", "dataset_id", "normal_MAE", "shuffled_MAE", "disabled_MAE", "self_target_MAE",
        "guide_advantage_vs_shuffled", "guide_advantage_vs_disabled", "self_target_improvement",
        "normal_vs_shuffled_output_L1", "normal_vs_disabled_output_L1", "normal_vs_self_target_output_L1",
        "attention_rank_1", "attention_rank_2", "attention_rank_3",
        "self_target_attention_rank_1", "self_target_attention_rank_2", "self_target_attention_rank_3",
        "reference_context_norm",
    }
    assert expected_columns <= set(frame.columns)
    assert np.isfinite(frame[list(expected_columns - {"trend_id", "dataset_id"})].to_numpy()).all()
    assert (tmp_path / "out" / "condition_usage_summary_t900.json").is_file()

    block_path = result["block_breakdown"]["900"]
    assert block_path is not None
    block_frame = pd.read_csv(block_path, encoding="utf-8-sig")
    assert len(block_frame) == 10
    expected_block_columns = {
        "block", "normal_context_norm", "self_target_context_norm",
        "normal_attention_rank_1", "normal_attention_rank_2", "normal_attention_rank_3",
        "self_target_attention_rank_1", "self_target_attention_rank_2", "self_target_attention_rank_3",
        "normal_scale_norm", "self_target_scale_norm", "normal_shift_norm", "self_target_shift_norm",
    }
    assert expected_block_columns == set(block_frame.columns)
    assert np.isfinite(block_frame.drop(columns=["block"]).to_numpy()).all()


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
    for timestep in (100, 900):
        assert (tmp_path / "out" / f"block_breakdown_t{timestep}.csv").is_file()
    assert set(result["block_breakdown"]) == {"100", "900"}


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
    assert (frame["normal_vs_self_target_output_L1"] == 0.0).all()
    assert (frame["guide_advantage_vs_shuffled"] == 0.0).all()
    assert (frame["guide_advantage_vs_disabled"] == 0.0).all()
    assert (frame["self_target_improvement"] == 0.0).all()
    summary = result["results"]["900"]
    assert summary["guide_advantage_vs_disabled"]["mean"] == pytest.approx(0.0)
    assert summary["self_target_improvement"]["mean"] == pytest.approx(0.0)


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
    assert frame["self_target_attention_rank_1"].isna().all()
    assert frame["reference_context_norm"].isna().all()
    # self_targetのguideの中身(条件正規化した自分自身の正解)は合成データのnormal guide
    # (target+微小ノイズ)と異なる値になるため、guideをそのまま出力するこのダミーモデルでは
    # normalとself_targetで出力が変わるはず(=self_targetが正しく別の入力として渡っている証拠)。
    assert (frame["normal_vs_self_target_output_L1"] > 0.0).all()


def test_self_target_batch_fills_only_slot_zero_with_condition_normalized_target(tmp_path_factory, tmp_path):
    from acceleration_forecasting_v2.inference.self_guide_diagnostics import _build_self_target_batch

    dataset_dir = _dataset_dir_for_dummy_models(tmp_path_factory)
    from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2

    dataset = ForecastDatasetV2(dataset_dir / "model_validation", dataset_dir, include_targets=True)
    batch = {key: (value.unsqueeze(0) if torch.is_tensor(value) else value) for key, value in dataset[0].items()}
    current_values = np.array([1.0], dtype=np.float32)

    self_target = _build_self_target_batch(dataset, batch, current_values)
    expected_guide_values = dataset.condition_norm.normalize(dataset.denormalize_target(batch["target"].numpy()))
    np.testing.assert_allclose(self_target["guide_values"][:, 0].numpy(), expected_guide_values, atol=1e-4)
    np.testing.assert_array_equal(self_target["guide_mask"][:, 0].numpy(), batch["target_mask"].numpy())
    assert self_target["guide_mask"][:, 1:].sum() == 0
    assert self_target["retrieval_mask"][0, 0] == 1.0
    assert self_target["retrieval_mask"][0, 1:].sum() == 0
    assert self_target["guide_similarities"][0, 0] == 1.0
    # guide以外のフィールド(history_values等)は変更しない。
    assert torch.equal(self_target["history_values"], batch["history_values"])


# --- ブロック単位の内訳(block_breakdown) ------------------------------------------

def test_per_block_diagnostic_stats_returns_empty_for_models_without_reference_blocks():
    assert _per_block_diagnostic_stats(_GuideIgnoringModel()) == []
    assert _per_block_diagnostic_stats(_GuideOnlyModel()) == []


def test_per_block_diagnostic_stats_matches_labels_and_aggregate(trained):
    from torch.utils.data import DataLoader

    from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
    from acceleration_forecasting_v2.inference.predict import load_process

    dataset_dir, checkpoint = trained
    process, _ = load_process(checkpoint, "cpu", use_ema=True)
    model = process.model
    model.eval()

    dataset = ForecastDatasetV2(dataset_dir / "model_validation", dataset_dir, include_targets=True)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    target = batch["target"]
    alpha = process.alpha_bars[900]
    noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * torch.randn_like(target)
    timesteps = torch.full((target.shape[0],), 900, dtype=torch.long)

    with torch.inference_mode():
        model(noisy, timesteps, batch)
        per_block = _per_block_diagnostic_stats(model)
        aggregate = model.diagnostic_stats()

    assert len(per_block) == 10
    assert [entry["block"] for entry in per_block] == list(_BLOCK_LABELS)
    # 既存の信頼できる集計値(diagnostic_stats、全ブロック平均)の正しい分解になっているはず。
    assert np.mean([entry["context_norm"] for entry in per_block]) == pytest.approx(
        aggregate["reference_context_norm"], rel=1e-5
    )
    for index in (1, 2, 3):
        assert np.mean([entry[f"attention_rank_{index}"] for entry in per_block]) == pytest.approx(
            aggregate[f"attention_rank_{index}"], rel=1e-5
        )


def test_block_stats_accumulator_uses_batch_size_weighted_mean():
    accumulator = _BlockStatsAccumulator()
    accumulator.add(
        [{"block": "b", "context_norm": 0.0, "attention_rank_1": 0.0, "attention_rank_2": 0.0, "attention_rank_3": 0.0}],
        weight=1,
    )
    accumulator.add(
        [{"block": "b", "context_norm": 4.0, "attention_rank_1": 4.0, "attention_rank_2": 4.0, "attention_rank_3": 4.0}],
        weight=3,
    )
    result = accumulator.result()
    assert len(result) == 1 and result[0]["block"] == "b"
    # 正しい加重平均は(0*1+4*3)/4=3.0。単純平均(2.0)ならこのテストで検出できる。
    assert result[0]["context_norm"] == pytest.approx(3.0)
    assert result[0]["attention_rank_1"] == pytest.approx(3.0)


def test_block_stats_accumulator_returns_empty_when_never_fed():
    assert _BlockStatsAccumulator().result() == []


def test_combine_block_breakdown_keeps_each_blocks_values_distinct():
    normal_rows = [
        {"block": "a", "context_norm": 1.0, "attention_rank_1": 0.1, "attention_rank_2": 0.2, "attention_rank_3": 0.3},
        {"block": "b", "context_norm": 5.0, "attention_rank_1": 0.5, "attention_rank_2": 0.6, "attention_rank_3": 0.7},
    ]
    self_target_rows = [
        {"block": "a", "context_norm": 2.0, "attention_rank_1": 0.9, "attention_rank_2": 0.0, "attention_rank_3": 0.0},
        {"block": "b", "context_norm": 6.0, "attention_rank_1": 0.9, "attention_rank_2": 0.0, "attention_rank_3": 0.0},
    ]
    combined = _combine_block_breakdown(normal_rows, self_target_rows)
    assert [row["block"] for row in combined] == ["a", "b"]
    assert combined[0]["normal_context_norm"] == 1.0 and combined[0]["self_target_context_norm"] == 2.0
    assert combined[1]["normal_context_norm"] == 5.0 and combined[1]["self_target_context_norm"] == 6.0
    assert combined[0]["self_target_context_norm"] != combined[1]["self_target_context_norm"]


def test_combine_block_breakdown_returns_empty_if_either_side_is_empty():
    row = [{"block": "a", "context_norm": 1.0, "attention_rank_1": 0.0, "attention_rank_2": 0.0, "attention_rank_3": 0.0}]
    assert _combine_block_breakdown([], row) == []
    assert _combine_block_breakdown(row, []) == []


def test_self_target_attention_is_exactly_zero_for_masked_slots_in_every_block(trained, tmp_path):
    # self_targetはスロット1・2を完全にマスクするため(softmax後に有効マスクを乗算する実装上)、
    # 学習内容に関わらず全ブロックで厳密に0になる非flakyな構造的性質。
    dataset_dir, checkpoint = trained
    result = diagnose_guide_conditioning(dataset_dir, checkpoint, tmp_path / "out", split="model_validation",
                                        timesteps=(900,), device="cpu", batch_size=4)
    frame = pd.read_csv(result["block_breakdown"]["900"], encoding="utf-8-sig")
    assert len(frame) == 10
    assert list(frame["block"]) == list(_BLOCK_LABELS)
    np.testing.assert_allclose(frame["self_target_attention_rank_2"].to_numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(frame["self_target_attention_rank_3"].to_numpy(), 0.0, atol=1e-6)
    assert (frame["self_target_attention_rank_1"] > 0.9).all()


def test_block_breakdown_is_none_and_no_file_for_models_without_reference_blocks(tmp_path_factory, tmp_path):
    dataset_dir = _dataset_dir_for_dummy_models(tmp_path_factory)
    process = DiffusionProcess(_GuideIgnoringModel(), steps=1000, prediction_type="epsilon")
    result = _diagnose_with_process(process, {"prediction_type": "epsilon"}, dataset_dir, tmp_path / "out",
                                    split="model_validation", timesteps=(900,), batch_size=4)
    assert result["block_breakdown"]["900"] is None
    assert not (tmp_path / "out" / "block_breakdown_t900.csv").is_file()


# --- FiLM(condition_fusion由来のscale/shift)の内訳 ---------------------------------

def test_per_block_film_stats_returns_empty_for_models_without_reference_blocks():
    assert _per_block_film_stats(_GuideIgnoringModel()) == []
    assert _per_block_film_stats(_GuideOnlyModel()) == []


def test_per_block_film_stats_matches_labels_and_is_finite(trained):
    from torch.utils.data import DataLoader

    from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
    from acceleration_forecasting_v2.inference.predict import load_process

    dataset_dir, checkpoint = trained
    process, _ = load_process(checkpoint, "cpu", use_ema=True)
    model = process.model
    model.eval()

    dataset = ForecastDatasetV2(dataset_dir / "model_validation", dataset_dir, include_targets=True)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    target = batch["target"]
    alpha = process.alpha_bars[900]
    noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * torch.randn_like(target)
    timesteps = torch.full((target.shape[0],), 900, dtype=torch.long)

    with torch.inference_mode():
        model(noisy, timesteps, batch)
        per_block = _per_block_film_stats(model)

    assert len(per_block) == 10
    assert [entry["block"] for entry in per_block] == list(_BLOCK_LABELS)
    values = np.array([[entry["scale_norm"], entry["shift_norm"]] for entry in per_block])
    assert np.isfinite(values).all()
    assert (values > 0).all()  # normは非負、実際のFiLM出力があれば通常0にはならない


def test_combine_film_breakdown_keeps_each_blocks_values_distinct():
    normal_rows = [
        {"block": "a", "scale_norm": 1.0, "shift_norm": 0.1},
        {"block": "b", "scale_norm": 5.0, "shift_norm": 0.5},
    ]
    self_target_rows = [
        {"block": "a", "scale_norm": 2.0, "shift_norm": 0.2},
        {"block": "b", "scale_norm": 6.0, "shift_norm": 0.6},
    ]
    combined = _combine_film_breakdown(normal_rows, self_target_rows)
    assert [row["block"] for row in combined] == ["a", "b"]
    assert combined[0]["normal_scale_norm"] == 1.0 and combined[0]["self_target_scale_norm"] == 2.0
    assert combined[1]["normal_scale_norm"] == 5.0 and combined[1]["self_target_scale_norm"] == 6.0
    assert combined[0]["self_target_scale_norm"] != combined[1]["self_target_scale_norm"]


def test_combine_film_breakdown_returns_empty_if_either_side_is_empty():
    row = [{"block": "a", "scale_norm": 1.0, "shift_norm": 0.1}]
    assert _combine_film_breakdown([], row) == []
    assert _combine_film_breakdown(row, []) == []


def test_merge_breakdowns_adds_film_columns_without_dropping_attention_columns():
    attention_breakdown = [
        {"block": "a", "normal_context_norm": 1.0, "self_target_context_norm": 2.0},
        {"block": "b", "normal_context_norm": 3.0, "self_target_context_norm": 4.0},
    ]
    film_breakdown = [
        {"block": "a", "normal_scale_norm": 0.1, "self_target_scale_norm": 0.2},
        {"block": "b", "normal_scale_norm": 0.3, "self_target_scale_norm": 0.4},
    ]
    merged = _merge_breakdowns(attention_breakdown, film_breakdown)
    assert merged[0] == {
        "block": "a", "normal_context_norm": 1.0, "self_target_context_norm": 2.0,
        "normal_scale_norm": 0.1, "self_target_scale_norm": 0.2,
    }
    assert merged[1]["block"] == "b" and merged[1]["normal_scale_norm"] == 0.3


def test_merge_breakdowns_returns_empty_when_attention_breakdown_is_empty():
    assert _merge_breakdowns([], [{"block": "a", "normal_scale_norm": 0.1}]) == []


def test_film_breakdown_is_absent_for_models_without_reference_blocks(tmp_path_factory, tmp_path):
    dataset_dir = _dataset_dir_for_dummy_models(tmp_path_factory)
    process = DiffusionProcess(_GuideOnlyModel(), steps=1000, prediction_type="epsilon")
    result = _diagnose_with_process(process, {"prediction_type": "epsilon"}, dataset_dir, tmp_path / "out",
                                    split="model_validation", timesteps=(900,), batch_size=4)
    assert result["block_breakdown"]["900"] is None
