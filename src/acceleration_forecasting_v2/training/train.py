"""学習ループ。`acceleration_forecasting_12m/.../training/train.py`を出発点に、
SPEC.md 1.1/1.5/1.6の決定に合わせて次の点を変更した。

- モデルはRMA(`ReferenceModulatedUNetV2`)のみ。attention_type等の分岐と、
  アーキテクチャ引数の露出(base_channels等)を削除した(SPEC 1.3: 固定)。
- 損失はセグメント単位重み付き(`SegmentWeightedMaskedLoss`)。**学習lossだけでなく
  validation lossにも同じ重みを適用する**(early stopping/best checkpoint選定の
  基準が、anchor数の多いセグメントに偏らないようにするため。SPEC 1.1)。
- Min-SNR・classifier-free guidance関連を削除(SPEC 1.5/1.6)。
- prediction_typeは"epsilon"/"v_prediction"を切り替えて2回学習を回す
  (SPEC 1.5: 1回の学習=1つのprediction_type)。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from acceleration_forecasting_v2.diffusion.process import DiffusionProcess
from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2
from .ema import EMA
from .loss import SegmentWeightedMaskedLoss


def _device(value):
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _learning_rate_factor(epoch, epochs, warmup=5, minimum_ratio=0.01):
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def dataset_build_id(dataset_dir) -> str:
    """データセットの同一性を判定するID(resume時に別データでの再開を検出するため)。

    正規化統計2ファイルの中身と、各splitのレコード数から算出する。
    """
    dataset_dir = Path(dataset_dir)
    digest = hashlib.sha256()
    for name in ("condition_normalization.json", "target_normalization.json"):
        digest.update((dataset_dir / name).read_bytes())
    for split in ("model_train", "model_validation"):
        digest.update(str(len(ForecastDatasetV2(dataset_dir / split, dataset_dir))).encode())
    return digest.hexdigest()


@torch.inference_mode()
def validation_loss(process, loader, device, seed=42):
    """検証セット全体でのsegment_weight付き平均損失(と、参考のレコード単位単純平均)を返す。

    バッチごとに重み付き平均を取って平均する方式ではなく、検証セット全体の
    Σ(w·loss)/Σw を計算する(バッチ構成に結果が依存しないようにするため)。
    """
    process.model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    weighted_sum, weight_sum, record_sum, count = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = _move(batch, device)
        size = batch["target"].shape[0]
        timesteps = torch.randint(0, process.steps, (size,), device=device, generator=generator)
        noise = torch.randn(batch["target"].shape, device=device, generator=generator)
        per_record = process.per_record_loss(batch, noise=noise, timesteps=timesteps)
        weights = batch["segment_weight"]
        weighted_sum += float((weights * per_record).sum())
        weight_sum += float(weights.sum())
        record_sum += float(per_record.sum())
        count += size
    if weight_sum <= 0:
        raise ValueError("検証セットの重みの合計が0です。")
    return {"loss": weighted_sum / weight_sum, "record_loss": record_sum / max(count, 1)}


def train(dataset_dir, output_dir, *, device=None, epochs=200, batch_size=128,
          accumulation_steps=2, learning_rate=1e-4, weight_decay=1e-4,
          dropout=0.1, patience=20, min_delta=1e-4, ema_decay=0.999,
          seed=42, resume=True, progress=True, prediction_type="epsilon",
          diffusion_steps=1000):
    """RMAを学習する。dataset_dirは`write_dataset`が書き出したディレクトリ。"""
    if prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError("prediction_type must be epsilon or v_prediction (SPEC.md 1.5)")
    if int(diffusion_steps) <= 0:
        raise ValueError("diffusion_steps must be positive")
    torch.manual_seed(seed)
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_id = dataset_build_id(dataset_dir)
    train_data = ForecastDatasetV2(dataset_dir / "model_train", dataset_dir)
    valid_data = ForecastDatasetV2(dataset_dir / "model_validation", dataset_dir)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    device = _device(device)

    model_config = {"dropout": float(dropout), "forecast_months": 12, "target_mode": "absolute",
                    "prediction_type": prediction_type, "diffusion_steps": int(diffusion_steps)}
    model = ReferenceModulatedUNetV2(dropout).to(device)
    process = DiffusionProcess(model, diffusion_steps, prediction_type=prediction_type).to(device)
    ema = EMA(model, ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: _learning_rate_factor(epoch, epochs))
    loss_fn = SegmentWeightedMaskedLoss()
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    autocast_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16

    start_epoch, best, stale, history = 0, float("inf"), 0, []
    last_path = output_dir / "last_model.pt"
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("dataset_build_id") != build_id or checkpoint.get("model_config") != model_config:
            raise ValueError("Checkpoint dataset/model configuration does not match")
        model.load_state_dict(checkpoint["model_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_validation_loss"])
        stale = int(checkpoint.get("stale_epochs", 0))
        if (output_dir / "training_history.csv").is_file():
            history = pd.read_csv(output_dir / "training_history.csv").to_dict("records")

    epoch_range = range(start_epoch, epochs)
    if progress:
        from tqdm import tqdm
        epoch_range = tqdm(epoch_range, desc="学習epoch", unit="epoch")
    started = time.perf_counter()
    for epoch in epoch_range:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        weighted_sum, weight_sum, batches = 0.0, 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_cuda):
                per_record = process.per_record_loss(batch)
                loss = loss_fn.batch_loss(per_record, batch["segment_weight"]) / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            weights = batch["segment_weight"]
            weighted_sum += float((weights * per_record.detach().float()).sum())
            weight_sum += float(weights.sum())
            batches += 1
        train_loss = weighted_sum / max(weight_sum, 1e-12)

        validation_process = DiffusionProcess(ema.model, diffusion_steps, prediction_type=prediction_type).to(device)
        validation = validation_loss(validation_process, valid_loader, device, seed)
        improved = validation["loss"] < best - min_delta
        if improved:
            best, stale = validation["loss"], 0
        else:
            stale += 1
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation["loss"],
            "validation_record_loss": validation["record_loss"], "best_validation_loss": best,
            "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - started,
        })
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        checkpoint = {
            "epoch": epoch, "model_state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_loss": best, "stale_epochs": stale, "dataset_build_id": build_id,
            "model_config": model_config, "seed": seed,
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, output_dir / "best_model.pt")
        scheduler.step()
        if progress:
            epoch_range.set_postfix(train=f"{train_loss:.5f}", valid=f"{validation['loss']:.5f}", best=f"{best:.5f}")
        if stale >= patience:
            break

    resolved = {
        "dataset_dir": str(dataset_dir), "dataset_build_id": build_id, "device": str(device),
        "epochs_completed": len(history), "best_validation_loss": best, "model_config": model_config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "seed": int(seed),
        "training_elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved
