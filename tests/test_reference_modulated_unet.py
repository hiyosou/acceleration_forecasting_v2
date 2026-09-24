"""models.reference_modulated_unet.ReferenceModulatedUNetV2 の単体テスト(SPEC.md 1.1/1.3/1.6対応)。"""

import torch

from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2


def _dummy_batch(batch_size=4, history_months=6, forecast_months=12, top_k=3):
    return {
        "history_values": torch.randn(batch_size, history_months),
        "history_masks": torch.ones(batch_size, history_months),
        "guide_values": torch.randn(batch_size, top_k, forecast_months),
        "guide_deltas": torch.randn(batch_size, top_k, forecast_months),
        "guide_similarities": torch.rand(batch_size, top_k),
        "guide_mask": torch.ones(batch_size, top_k, forecast_months),
        "retrieval_mask": torch.ones(batch_size, top_k),
    }


def test_forward_output_shape_matches_forecast_months():
    model = ReferenceModulatedUNetV2()
    batch = _dummy_batch(batch_size=4)
    noisy = torch.randn(4, 12)
    timesteps = torch.randint(0, 1000, (4,))
    output = model(noisy, timesteps, batch)
    assert output.shape == (4, 12)


def test_forward_produces_finite_output():
    model = ReferenceModulatedUNetV2()
    batch = _dummy_batch(batch_size=2)
    output = model(torch.randn(2, 12), torch.randint(0, 1000, (2,)), batch)
    assert torch.isfinite(output).all()


def test_gradients_flow_to_condition_encoder_and_reference_encoder():
    model = ReferenceModulatedUNetV2()
    batch = _dummy_batch(batch_size=2)
    output = model(torch.randn(2, 12), torch.randint(0, 1000, (2,)), batch)
    output.sum().backward()
    assert model.condition_encoder[0].weight.grad is not None
    assert (model.condition_encoder[0].weight.grad.abs().sum() > 0).item()
    assert model.reference_encoder.network[0].weight.grad is not None


def test_default_config_matches_spec_1_3_recommended_values():
    model = ReferenceModulatedUNetV2()
    assert model.reference_similarity_enabled is False
    assert model.reference_attention_mode == "month_aligned"
    assert model.reference_fusion_type == "ratd_condition_reconstruction"
    assert model.base_channels == 64
    assert model.channel_multipliers == (1, 2, 4)
    assert model.block_counts == (2, 2, 2, 2, 2)
    assert model.condition_dim == 256
    assert model.reference_dim == 64
    assert model.attention_heads == 8
    assert model.attention_head_dim == 8


def test_condition_encoder_input_dim_is_twice_history_months():
    # SPEC 1.1: history_values(6)+history_masks(6)=12次元(旧実装の11次元current+history統合から変更)。
    model = ReferenceModulatedUNetV2(history_months=6)
    assert model.condition_encoder[0].in_features == 12
    model_alt = ReferenceModulatedUNetV2(history_months=4)
    assert model_alt.condition_encoder[0].in_features == 8


def test_rejects_invalid_channel_multipliers_length():
    import pytest
    with pytest.raises(ValueError, match="channel_multipliers"):
        ReferenceModulatedUNetV2(channel_multipliers=(1, 2))


def test_rejects_invalid_block_counts_length():
    import pytest
    with pytest.raises(ValueError, match="block_counts"):
        ReferenceModulatedUNetV2(block_counts=(2, 2, 2))


def test_rejects_odd_condition_dim():
    import pytest
    with pytest.raises(ValueError, match="condition_dim"):
        ReferenceModulatedUNetV2(condition_dim=255)


def test_month_aligned_attention_mode_works_at_all_three_resolution_levels():
    # U-Netの3段階(12->6->3)がmonth_alignment_maskの許容長{3,6,12}とちょうど一致することを
    # 実際にforward通しで確認する(month_alignedを使うと決めた1.3節の決定が、この
    # アーキテクチャの解像度設計と矛盾しないことの検算)。
    model = ReferenceModulatedUNetV2(reference_attention_mode="month_aligned")
    batch = _dummy_batch(batch_size=1)
    output = model(torch.randn(1, 12), torch.randint(0, 1000, (1,)), batch)
    assert output.shape == (1, 12)


def test_parameter_count_is_reasonable_compared_to_historical_reference():
    # SPEC.md 1.3で根拠にした旧A_epsilon_t1000構成の実測パラメータ数は約488万
    # (11次元条件入力・全く同じアーキテクチャ)。条件入力が11→12次元に変わった分の
    # 差は無視できるほど小さいため、同程度のオーダーに収まることを確認する。
    model = ReferenceModulatedUNetV2()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 4_000_000 <= parameter_count <= 6_000_000


def test_reference_blocks_property_lists_all_ten_blocks():
    model = ReferenceModulatedUNetV2()
    assert len(model.reference_blocks) == sum(model.block_counts)


def test_non_default_configuration_paths_still_produce_valid_output():
    # デフォルトはreference_similarity_enabled=False/month_aligned/ratd_condition_reconstruction
    # だが、クラス自体は他の組み合わせも受け付ける設計になっている(SPEC 1.3で「CLIには
    # 露出させない」と決めただけで、クラスの引数自体は残している)。デフォルト以外の
    # 分岐(similarity_scaleパラメータ・globalアテンション・residual_delta融合)が
    # 実際に動くことを確認する。
    model = ReferenceModulatedUNetV2(
        reference_similarity_enabled=True,
        reference_attention_mode="global",
        reference_fusion_type="residual_delta",
    )
    batch = _dummy_batch(batch_size=2)
    output = model(torch.randn(2, 12), torch.randint(0, 1000, (2,)), batch)
    assert output.shape == (2, 12)
    assert torch.isfinite(output).all()
    output.sum().backward()
    first_block = model.reference_blocks[0]
    assert first_block.reference_attention.similarity_scale.grad is not None


def test_diagnostic_stats_available_after_forward_pass():
    model = ReferenceModulatedUNetV2()
    batch = _dummy_batch(batch_size=2)
    model(torch.randn(2, 12), torch.randint(0, 1000, (2,)), batch)
    stats = model.diagnostic_stats()
    assert "reference_context_norm" in stats
    assert "attention_rank_1" in stats
