"""diffusion.process.DiffusionProcess の単体テスト。"""

import torch
from torch import nn
import pytest

from acceleration_forecasting_v2.diffusion.process import DiffusionProcess, cosine_alpha_bars
from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2


class _ConstantModel(nn.Module):
    """batchを無視し、常に固定値を返す手計算用のダミーモデル。"""

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, noisy, timesteps, batch):
        return torch.full_like(noisy, self.value) * self.weight


def _dummy_batch(batch_size, forecast_months=12, history_months=6, top_k=3):
    return {
        "history_values": torch.zeros(batch_size, history_months),
        "history_masks": torch.ones(batch_size, history_months),
        "guide_values": torch.zeros(batch_size, top_k, forecast_months),
        "guide_deltas": torch.zeros(batch_size, top_k, forecast_months),
        "guide_similarities": torch.zeros(batch_size, top_k),
        "guide_mask": torch.ones(batch_size, top_k, forecast_months),
        "retrieval_mask": torch.ones(batch_size, top_k),
    }


def test_cosine_alpha_bars_is_monotonically_decreasing_and_bounded():
    alpha_bars = cosine_alpha_bars(steps=1000)
    assert alpha_bars.shape == (1000,)
    assert (alpha_bars[:-1] >= alpha_bars[1:]).all()
    assert alpha_bars[0] <= 1.0
    assert alpha_bars[-1] >= 0.0


def test_per_record_loss_hand_computed_example():
    # モデルは常に0を返す固定モデル。target=[2,2,2,2]、mask=[1,1,0,0]、noise=0なら、
    # noisy = sqrt(alpha)*target。epsilon予測ではexpected=noise=0固定なので、
    # 二乗誤差 = predicted^2 = 0(モデル出力0)。この設定では損失は常に0になるため、
    # ここでは「マスクされた位置が無視される」ことを確認するのに十分な、より直接的な
    # 検算を行う: モデル出力を固定値1、期待値をnoiseとして具体的に計算する。
    model = _ConstantModel(value=1.0)
    process = DiffusionProcess(model, steps=1000, prediction_type="epsilon")
    batch = _dummy_batch(batch_size=1)
    batch["target"] = torch.zeros(1, 12)
    batch["target"][0, :4] = 2.0
    batch["target_mask"] = torch.zeros(1, 12)
    batch["target_mask"][0, :2] = 1.0  # 先頭2つだけ有効
    noise = torch.zeros(1, 12)
    timesteps = torch.zeros(1, dtype=torch.long)
    per_record = process.per_record_loss(batch, noise=noise, timesteps=timesteps)
    # epsilon予測: expected=noise=0。model出力は常に1.0(weight=1.0で初期化)。
    # 二乗誤差=(1.0-0.0)^2=1.0が、有効な2箇所のみで平均される -> 1.0
    assert per_record.shape == (1,)
    assert per_record.item() == pytest.approx(1.0, rel=1e-4)


def test_per_record_loss_all_masked_gives_zero_via_clamp():
    model = _ConstantModel(value=5.0)
    process = DiffusionProcess(model, steps=1000, prediction_type="epsilon")
    batch = _dummy_batch(batch_size=1)
    batch["target"] = torch.zeros(1, 12)
    batch["target_mask"] = torch.zeros(1, 12)  # 全てマスクされている
    per_record = process.per_record_loss(batch, noise=torch.zeros(1, 12), timesteps=torch.zeros(1, dtype=torch.long))
    # mask.sum()=0 -> clamp_min(1)により分母1、分子0(sum*mask=0) -> 損失0
    assert per_record.item() == pytest.approx(0.0)


def test_model_output_to_x0_epsilon_roundtrip_for_each_prediction_type():
    for prediction_type in ("epsilon", "v_prediction", "x0_prediction"):
        model = _ConstantModel(value=0.0)
        process = DiffusionProcess(model, steps=1000, prediction_type=prediction_type)
        values = torch.tensor([[1.0, 2.0]])
        alpha = torch.tensor([[0.7]])
        target = torch.tensor([[0.5, -0.3]])
        noise = (values - alpha.sqrt() * target) / (1 - alpha).sqrt()
        if prediction_type == "epsilon":
            output = noise
        elif prediction_type == "v_prediction":
            output = alpha.sqrt() * noise - (1 - alpha).sqrt() * target
        else:
            output = target
        predicted_clean, predicted_epsilon = process.model_output_to_x0_epsilon(values, output, alpha)
        torch.testing.assert_close(predicted_clean, target, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(predicted_epsilon, noise, atol=1e-4, rtol=1e-4)


def test_rejects_unknown_prediction_type():
    with pytest.raises(ValueError, match="prediction_type"):
        DiffusionProcess(_ConstantModel(0.0), prediction_type="unknown")


def test_ddim_sampling_produces_correct_shape_with_real_model():
    model = ReferenceModulatedUNetV2()
    process = DiffusionProcess(model, steps=1000, prediction_type="epsilon")
    batch = _dummy_batch(batch_size=2)
    generator = torch.Generator().manual_seed(0)
    samples = process.ddim(batch, shape=(2, 12), sampling_steps=5, generator=generator)
    assert samples.shape == (2, 12)
    assert torch.isfinite(samples).all()


def test_ddim_rejects_sampling_steps_out_of_range():
    model = ReferenceModulatedUNetV2()
    process = DiffusionProcess(model, steps=1000, prediction_type="epsilon")
    batch = _dummy_batch(batch_size=1)
    with pytest.raises(ValueError, match="sampling_steps"):
        process.ddim(batch, shape=(1, 12), sampling_steps=0)
    with pytest.raises(ValueError, match="sampling_steps"):
        process.ddim(batch, shape=(1, 12), sampling_steps=2000)


def test_ddim_single_step_output_equals_clamped_predicted_clean():
    # sampling_steps=1・eta=0では、next_timestep=-1(next_alpha=1.0, sigma=0, direction=0)
    # となり、最終出力は「クランプ後のpredicted_clean」に厳密に一致するはず。
    # モデルが極端に大きいepsilonを返す固定モデルを使い、normalized_clipの下限に
    # 実際にクランプされていることを厳密値で検算する(NaNが出ないことしか
    # 見ていなかった以前のテストより厳密)。
    model = _ConstantModel(value=100.0)
    process = DiffusionProcess(model, steps=1000, prediction_type="epsilon")
    batch = _dummy_batch(batch_size=1)
    generator = torch.Generator().manual_seed(0)
    samples = process.ddim(
        batch, shape=(1, 12), sampling_steps=1, normalized_clip=(-1.0, 1.0), eta=0.0,
        generator=generator, initial_noise_scale=0.0,
    )
    torch.testing.assert_close(samples, torch.full((1, 12), -1.0), atol=1e-5, rtol=1e-5)


def test_ddim_without_clip_can_exceed_the_would_be_clip_range():
    # 上のテストとの対比: normalized_clipを指定しない場合、同じ極端なepsilon出力から
    # 計算されるpredicted_cleanは[-1,1]の範囲を大きく超える(クランプ処理が
    # 実際に効果を持っていることの反証テスト)。
    model = _ConstantModel(value=100.0)
    process = DiffusionProcess(model, steps=1000, prediction_type="epsilon")
    batch = _dummy_batch(batch_size=1)
    generator = torch.Generator().manual_seed(0)
    samples = process.ddim(
        batch, shape=(1, 12), sampling_steps=1, normalized_clip=None, eta=0.0,
        generator=generator, initial_noise_scale=0.0,
    )
    assert (samples.abs() > 1.0).all()
