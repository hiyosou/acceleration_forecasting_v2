"""DDIMサンプリングによる12か月予測。1レコードにつき`num_samples`本の系列を生成し、
中央値・p10・p90(と標準偏差)を物理値で保存する。

`acceleration_forecasting_12m/.../inference/predict.py`を出発点に、target_mode=absolute
専用(guide_baselineの加算なし、SPEC.md 1.6)・CFGなしに単純化した。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from acceleration_forecasting_v2.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_v2.datasets.torch_dataset import ForecastDatasetV2
from acceleration_forecasting_v2.diffusion.process import DiffusionProcess
from acceleration_forecasting_v2.models.reference_modulated_unet import ReferenceModulatedUNetV2


def load_process(checkpoint_path, device, *, use_ema=True):
    """checkpointからモデルとDiffusionProcessを復元する(validationと同じEMA重みが既定)。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = ReferenceModulatedUNetV2(config["dropout"]).to(device)
    model.load_state_dict(checkpoint["ema_state_dict" if use_ema else "model_state_dict"])
    model.eval()
    process = DiffusionProcess(model, config["diffusion_steps"], prediction_type=config["prediction_type"]).to(device)
    return process, config


def _expand(batch, num_samples):
    return {key: value.repeat_interleave(num_samples, dim=0) if torch.is_tensor(value) else value
            for key, value in batch.items()}


@torch.inference_mode()
def predict(dataset_dir, checkpoint_path, output_dir, *, split="inference", device=None,
            num_samples=100, sampling_steps=50, eta=0.0, initial_noise_scale=1.0,
            batch_size=8, seed=42, use_ema=True, bounds=(PHYSICAL_MIN, PHYSICAL_MAX)):
    """指定splitの全レコードを予測し、predictions.csvとsamples.npzを書き出す。

    正解(target_values)は一切読み込まない(include_targets=False)ため、
    未来の実測値が予測に混入することはない。

    Returns:
        サマリdict(件数・設定)。
    """
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    process, config = load_process(checkpoint_path, device, use_ema=use_ema)

    dataset = ForecastDatasetV2(dataset_dir / split, dataset_dir, include_targets=False)
    metadata = pd.read_csv(dataset_dir / split / "metadata.csv", encoding="utf-8-sig")
    if len(metadata) != len(dataset):
        raise ValueError("metadata.csv と配列のレコード数が一致しません。")
    normalized_clip = tuple(float(value) for value in dataset.target_norm.normalize(np.asarray(bounds, np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    all_samples = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        size = batch["history_values"].shape[0]
        normalized = process.ddim(
            _expand(batch, num_samples), shape=(size * num_samples, 12), sampling_steps=sampling_steps,
            eta=eta, normalized_clip=normalized_clip, generator=generator, initial_noise_scale=initial_noise_scale,
        )
        physical = dataset.physical_prediction(normalized.cpu().numpy(), bounds=bounds)
        all_samples.append(physical.reshape(size, num_samples, 12))
    samples = np.concatenate(all_samples, axis=0)

    median = np.median(samples, axis=1)
    p10, p90 = np.percentile(samples, [10, 90], axis=1)
    std = samples.std(axis=1)
    rows = []
    for index in range(len(samples)):
        for month in range(12):
            rows.append({
                "trend_id": metadata.iloc[index]["trend_id"], "dataset_id": metadata.iloc[index].get("dataset_id", ""),
                "month_index": month + 1, "prediction_median": float(median[index, month]),
                "prediction_p10": float(p10[index, month]), "prediction_p90": float(p90[index, month]),
                "prediction_std": float(std[index, month]),
            })
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    # 文字列はobject型ではなく固定長のU型で保存する(allow_pickle=Falseで読めるように)。
    np.savez_compressed(output_dir / "samples.npz", trend_ids=metadata["trend_id"].astype(str).to_numpy().astype("U"),
                        samples=samples.astype(np.float32))
    summary = {
        "split": split, "record_count": int(len(samples)), "num_samples": int(num_samples),
        "sampling_steps": int(sampling_steps), "eta": float(eta), "seed": int(seed),
        "prediction_type": config["prediction_type"], "bounds": [float(bounds[0]), float(bounds[1])],
        "checkpoint": str(checkpoint_path), "use_ema": bool(use_ema),
    }
    (output_dir / "prediction_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
