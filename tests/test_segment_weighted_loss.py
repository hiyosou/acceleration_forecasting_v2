"""training.loss.SegmentWeightedMaskedLoss の単体テスト(SPEC.md 1.1の数式・手計算例に対応)。"""

import torch
import pytest

from acceleration_forecasting_v2.training.loss import SegmentWeightedMaskedLoss


def test_batch_loss_matches_spec_1_1_hand_computed_example():
    # SPEC.md 1.1の例: セグメントAに10件(loss=0.20固定)、セグメントBに1件(loss=0.50)。
    # segment_weight = 1/セグメント内件数 -> A:0.1(×10件), B:1.0(×1件)。
    # 期待値(2段階平均と同じ) = (0.20+0.50)/2 = 0.35
    per_record_loss = torch.tensor([0.20] * 10 + [0.50])
    segment_weights = torch.tensor([0.1] * 10 + [1.0])
    loss = SegmentWeightedMaskedLoss().batch_loss(per_record_loss, segment_weights)
    assert loss.item() == pytest.approx(0.35, abs=1e-6)


def test_batch_loss_equals_plain_mean_when_all_segment_weights_equal():
    per_record_loss = torch.tensor([0.1, 0.2, 0.3, 0.4])
    segment_weights = torch.ones(4)
    loss = SegmentWeightedMaskedLoss().batch_loss(per_record_loss, segment_weights)
    assert loss.item() == pytest.approx(per_record_loss.mean().item(), abs=1e-6)


def test_batch_loss_combines_snr_weight_multiplicatively():
    per_record_loss = torch.tensor([1.0, 1.0])
    segment_weights = torch.tensor([1.0, 1.0])
    snr_weights = torch.tensor([1.0, 3.0])
    loss = SegmentWeightedMaskedLoss().batch_loss(per_record_loss, segment_weights, snr_weights=snr_weights)
    # combined_weight = [1.0, 3.0] -> (1*1.0 + 3*1.0) / (1+3) = 1.0 (両方loss=1.0なので不変)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)

    per_record_loss_varied = torch.tensor([0.0, 2.0])
    loss_varied = SegmentWeightedMaskedLoss().batch_loss(per_record_loss_varied, segment_weights, snr_weights=snr_weights)
    # combined_weight=[1,3] -> (1*0.0 + 3*2.0)/4 = 1.5
    assert loss_varied.item() == pytest.approx(1.5, abs=1e-6)


def test_batch_loss_none_snr_weights_behaves_like_all_ones():
    per_record_loss = torch.tensor([0.0, 2.0])
    segment_weights = torch.tensor([1.0, 3.0])
    without_snr = SegmentWeightedMaskedLoss().batch_loss(per_record_loss, segment_weights, snr_weights=None)
    with_ones = SegmentWeightedMaskedLoss().batch_loss(per_record_loss, segment_weights, snr_weights=torch.ones(2))
    assert without_snr.item() == pytest.approx(with_ones.item(), abs=1e-6)


def test_batch_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        SegmentWeightedMaskedLoss().batch_loss(torch.zeros(3), torch.zeros(2))


def test_batch_loss_rejects_zero_weight_sum():
    with pytest.raises(ValueError, match="positive"):
        SegmentWeightedMaskedLoss().batch_loss(torch.zeros(2), torch.zeros(2))


def test_batch_loss_is_differentiable_with_respect_to_per_record_loss():
    per_record_loss = torch.tensor([0.2, 0.5], requires_grad=True)
    segment_weights = torch.tensor([0.5, 1.0])
    loss = SegmentWeightedMaskedLoss().batch_loss(per_record_loss, segment_weights)
    loss.backward()
    assert per_record_loss.grad is not None
    assert torch.isfinite(per_record_loss.grad).all()
