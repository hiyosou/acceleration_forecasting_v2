"""energy scoreで予測分布の形を直す微調整(2026-11、案B)。

D4 ext24で学習済みのモデル(EMA重み)を初期値に、損失を
    L = L_eps(今と同じ、全バッチ、セグメント重み) + γ · ES(部分バッチ、セグメント重み)
にして学習し直す。ESは、実際に短いDDIM(eta=0、勾配は全ステップを通す)で作った予測サンプルの
分布を、正解と比べるエネルギースコア(CRPSの多変量版)である:

    ES(レコード) = mean_k d(x_k, y) − 0.5 · mean_{k≠j} d(x_k, x_j)
    d(a, b)      = sqrt( Σ_有効な月 (a − b)² / 有効な月数 + 1e-8 )   (正規化空間、target_maskの月だけ)

`training/train.py`の部品(`validation_loss`・`SegmentWeightedMaskedLoss`・`EMA`・`dataset_build_id`)を
再利用し、出力は`train`と同じ形式(best_model.pt・last_model.pt・training_history.csv・resolved_config.json)
にして`inference.predict`がそのまま読めるようにする。epsilon予測・absoluteのみ。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from acceleration_forecasting_v2.diffusion.process import DEFAULT_START_TIMESTEP, DiffusionProcess
from acceleration_forecasting_v2.inference.predict import _expand, load_process
from .ema import EMA
from .loss import SegmentWeightedMaskedLoss
from .train import _device, _move, dataset_build_id, validation_loss

DISTANCE_EPSILON = 1e-8


def _masked_distance(a, b, mask, count):
    """a, b: (..., T)、mask: (..., T)、count: (...)。有効な月だけの二乗平均の平方根(+1e-8でsqrtの勾配を保つ)。"""
    return ((a - b).square() * mask).sum(dim=-1).div(count).add(DISTANCE_EPSILON).sqrt()


def energy_score(samples, target, mask):
    """samples (B,K,T)、target (B,T)、mask (B,T) → レコードごとのenergy score (B,)(小さいほど良い)。

    K=1のときは広がりの項は0。有効な月が1つもないレコードは0。
    """
    if samples.dim() != 3 or target.shape != mask.shape or samples.shape[0] != target.shape[0] \
            or samples.shape[2] != target.shape[1]:
        raise ValueError("samples (B,K,T), target (B,T), mask (B,T) の形が合いません")
    mask = mask.to(samples.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    k = samples.shape[1]
    to_target = _masked_distance(samples, target[:, None, :], mask[:, None, :], count[:, None]).mean(dim=1)
    if k > 1:
        pair_mask = mask[:, None, None, :]
        pairwise = _masked_distance(samples[:, :, None, :], samples[:, None, :, :], pair_mask, count[:, None, None])
        off_diagonal = 1.0 - torch.eye(k, device=samples.device, dtype=samples.dtype)
        spread = (pairwise * off_diagonal).sum(dim=(1, 2)) / (k * (k - 1))
    else:
        spread = torch.zeros_like(to_target)
    return (to_target - 0.5 * spread) * (mask.sum(dim=1) > 0).to(samples.dtype)


def _sample_energy(process, batch, *, samples, steps, clip, start_timestep, generator):
    """batch(B件)を`samples`回ずつ複製して微分可能なDDIMで予測を作り、レコードごとのESを返す。float32。"""
    size = batch["target"].shape[0]
    expanded = _expand(batch, samples)
    noise = torch.randn((size * samples, batch["target"].shape[1]), device=batch["target"].device,
                        generator=generator)
    values = process.ddim_differentiable(expanded, shape=tuple(noise.shape), sampling_steps=steps,
                                         normalized_clip=clip, start_timestep=start_timestep, noise=noise)
    return energy_score(values.reshape(size, samples, -1), batch["target"], batch["target_mask"])


@torch.inference_mode()
def validation_energy(process, loader, device, *, samples, steps, clip, start_timestep, seed):
    """固定の検証レコード(loader)でのsegment_weight付き平均ES。ノイズは固定seedで毎回同じ。float32。"""
    process.model.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    weighted, weights = 0.0, 0.0
    for batch in loader:
        batch = _move(batch, device)
        es = _sample_energy(process, batch, samples=samples, steps=steps, clip=clip,
                            start_timestep=start_timestep, generator=generator)
        weighted += float((batch["segment_weight"] * es).sum())
        weights += float(batch["segment_weight"].sum())
    return weighted / max(weights, 1e-12)


def _fixed_subset(dataset, count, seed):
    count = min(int(count), len(dataset))
    indices = np.sort(np.random.default_rng(int(seed)).choice(len(dataset), size=count, replace=False))
    return Subset(dataset, [int(i) for i in indices])


def finetune_energy(dataset_dir, output_dir, *, init_checkpoint, energy_weight, energy_batch=16, energy_samples=4,
                    energy_steps=4, learning_rate=2e-5, weight_decay=1e-4, max_epochs=20, patience=5,
                    validation_records=512, validation_samples=8, batch_size=128, accumulation_steps=2,
                    ema_decay=0.999, min_delta=1e-5, energy_start_timestep=None, seed=42, device=None, resume=True, progress=True):
    """D4 ext24のcheckpoint(EMA重み)をenergy scoreで微調整する。energy_weight=0は対照(ESを足さない同じ手順)。

    best_model.pt: エポック1以降で検証ESが最小のもの(エポック0=初期値は参考として記録するだけで、bestにしない)。
    検証ESは固定の`validation_records`件(seed固定で選ぶ)・K=`validation_samples`・EMA重みで測る。
    training_history.csvの列: epoch・train_loss_eps・train_energy・validation_loss・validation_energy・
    best_validation_energy・elapsed_seconds(epoch 0は初期値の検証値だけ)。
    """
    energy_weight = float(energy_weight)
    if energy_weight < 0:
        raise ValueError("energy_weight must be non-negative")
    if min(int(energy_batch), int(energy_samples), int(energy_steps), int(validation_samples)) < 1:
        raise ValueError("energy_batch / energy_samples / energy_steps / validation_samples must be positive")
    torch.manual_seed(seed)
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    init_checkpoint = Path(init_checkpoint).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(device)
    start_timestep = DEFAULT_START_TIMESTEP if energy_start_timestep is None else int(energy_start_timestep)

    # model・EMAとも初期checkpointのEMA重みで初期化する。model_configは引き継ぎ、energy_finetuneを足す。
    process, config = load_process(init_checkpoint, device, use_ema=True)
    if config["prediction_type"] != "epsilon" or config.get("target_mode", "absolute") != "absolute":
        raise ValueError("energy微調整はepsilon予測のabsoluteでだけ使えます")
    model = process.model
    diffusion_steps = int(config["diffusion_steps"])
    model_config = {**config, "energy_finetune": {
        "weight": energy_weight, "batch": int(energy_batch), "samples": int(energy_samples),
        "steps": int(energy_steps), "init": str(init_checkpoint)}}

    build_id = dataset_build_id(dataset_dir)
    train_data = ForecastDatasetV2(dataset_dir / "model_train", dataset_dir)
    valid_data = ForecastDatasetV2(dataset_dir / "model_validation", dataset_dir)
    if int(config.get("cycle_feature_dim", 0)) > 0 and (train_data.cycle_features is None
                                                         or valid_data.cycle_features is None):
        raise ValueError("このcheckpointにはデータセットの cycle_features.npy が必要です。")
    normalized_clip = tuple(float(v) for v in train_data.target_norm.normalize(
        np.asarray((PHYSICAL_MIN, PHYSICAL_MAX), np.float32)))
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=shuffle_generator, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    energy_loader = DataLoader(_fixed_subset(valid_data, validation_records, seed),
                               batch_size=max(1, 256 // int(validation_samples)), shuffle=False, num_workers=0)

    ema = EMA(model, ema_decay)
    validation_process = DiffusionProcess(ema.model, diffusion_steps, prediction_type="epsilon").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = SegmentWeightedMaskedLoss()
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    autocast_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16

    def measure():
        loss = validation_loss(validation_process, valid_loader, device, seed)["loss"]
        energy = validation_energy(validation_process, energy_loader, device, samples=int(validation_samples),
                                   steps=int(energy_steps), clip=normalized_clip, start_timestep=start_timestep,
                                   seed=seed + 7)
        return loss, energy

    def log(message):
        print(message, file=sys.stderr, flush=True)

    last_path = output_dir / "last_model.pt"
    history_path = output_dir / "training_history.csv"
    start_epoch, best, best_epoch, stale, history = 1, float("inf"), 0, 0, []
    started = time.perf_counter()
    elapsed_offset = 0.0
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("dataset_build_id") != build_id or checkpoint.get("model_config") != model_config:
            raise ValueError("Checkpoint dataset/model configuration does not match")
        model.load_state_dict(checkpoint["model_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_validation_energy"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale_epochs"])
        history = pd.read_csv(history_path).to_dict("records") if history_path.is_file() else []
        elapsed_offset = float(history[-1]["elapsed_seconds"]) if history else 0.0
    else:
        (output_dir / "best_model.pt").unlink(missing_ok=True)  # やり直すときは前回のbestを残さない
        # エポック0: 初期値(EMA=初期重み)の検証値。参考として記録するだけで、bestにはしない。
        loss0, energy0 = measure()
        history.append({"epoch": 0, "train_loss_eps": np.nan, "train_energy": np.nan, "validation_loss": loss0,
                        "validation_energy": energy0, "best_validation_energy": np.nan,
                        "elapsed_seconds": time.perf_counter() - started})
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
        log(f"epoch 0 (init): validation_loss={loss0:.5f} validation_energy={energy0:.5f}")

    def save(path, epoch):
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "best_validation_loss": best,
            "best_validation_energy": best, "best_epoch": best_epoch, "stale_epochs": stale,
            "dataset_build_id": build_id, "model_config": model_config, "seed": seed,
        }, path)

    if not last_path.is_file() or start_epoch == 1:
        save(last_path, 0)

    for epoch in range(start_epoch, int(max_epochs) + 1):
        if stale >= patience:
            break
        # エポックごとに乱数を決める(resumeしても、そのエポックの並び・ノイズが同じになる)。
        shuffle_generator.manual_seed(seed + epoch)
        torch.manual_seed(seed * 1000 + 900 + epoch)  # dropout用
        noise_generator = torch.Generator(device=device).manual_seed(seed * 1000 + epoch)
        energy_generator = torch.Generator(device=device).manual_seed(seed * 1000 + 500 + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        eps_sum, energy_sum, eps_weight, energy_weight_sum = 0.0, 0.0, 0.0, 0.0
        for batch_index, batch in enumerate(train_loader):
            batch = _move(batch, device)
            size = batch["target"].shape[0]
            timesteps = torch.randint(0, diffusion_steps, (size,), device=device, generator=noise_generator)
            noise = torch.randn(batch["target"].shape, device=device, generator=noise_generator)
            model.train()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_cuda):
                per_record = process.per_record_loss(batch, noise=noise, timesteps=timesteps)
                eps_loss = loss_fn.batch_loss(per_record, batch["segment_weight"])
            scaler.scale(eps_loss / accumulation_steps).backward()
            weights = batch["segment_weight"]
            eps_sum += float((weights * per_record.detach().float()).sum())
            eps_weight += float(weights.sum())
            if energy_weight > 0:
                count = min(int(energy_batch), size)
                part = {key: value[:count] if torch.is_tensor(value) else value for key, value in batch.items()}
                # 推論と同じ決定的なネットワーク(dropoutなし)でサンプルを作り、勾配は全ステップを通す。float32。
                model.eval()
                es = _sample_energy(process, part, samples=int(energy_samples), steps=int(energy_steps),
                                    clip=normalized_clip, start_timestep=start_timestep, generator=energy_generator)
                es_loss = loss_fn.batch_loss(es, part["segment_weight"])
                scaler.scale(energy_weight * es_loss / accumulation_steps).backward()
                model.train()
                energy_sum += float((part["segment_weight"] * es.detach()).sum())
                energy_weight_sum += float(part["segment_weight"].sum())
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
        train_eps = eps_sum / max(eps_weight, 1e-12)
        train_energy = energy_sum / energy_weight_sum if energy_weight_sum > 0 else float("nan")

        loss, energy = measure()
        improved = energy < best - min_delta
        if improved:
            best, best_epoch, stale = energy, epoch, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss_eps": train_eps, "train_energy": train_energy,
                        "validation_loss": loss, "validation_energy": energy, "best_validation_energy": best,
                        "elapsed_seconds": elapsed_offset + time.perf_counter() - started})
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
        save(last_path, epoch)
        if improved:
            save(output_dir / "best_model.pt", epoch)
        log(f"epoch {epoch}: train_eps={train_eps:.5f} train_energy={train_energy:.5f} validation_loss={loss:.5f} "
            f"validation_energy={energy:.5f} best={best:.5f}(epoch {best_epoch}) stale={stale} "
            f"elapsed={history[-1]['elapsed_seconds']:.0f}s")

    if not (output_dir / "best_model.pt").is_file():
        # 1エポックも改善しなかった(または回していない)ときは、最後の状態をbestとして残す(predictが読めるように)。
        save(output_dir / "best_model.pt", int(history[-1]["epoch"]))
    resolved = {
        "dataset_dir": str(dataset_dir), "dataset_build_id": build_id, "device": str(device),
        "epochs_completed": int(history[-1]["epoch"]), "best_validation_loss": best, "best_validation_energy": best,
        "best_epoch": best_epoch, "initial_validation_loss": history[0]["validation_loss"],
        "initial_validation_energy": history[0]["validation_energy"], "model_config": model_config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "seed": int(seed),
        "learning_rate": float(learning_rate), "energy_start_timestep": start_timestep,
        "validation_records": min(int(validation_records), len(valid_data)),
        "validation_samples": int(validation_samples),
        "training_elapsed_seconds": float(history[-1]["elapsed_seconds"]),
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved
