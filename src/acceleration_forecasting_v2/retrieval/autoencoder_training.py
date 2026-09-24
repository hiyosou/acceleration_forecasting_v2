"""波形Autoencoderの学習。`acceleration_retrieval/training.py`を出発点に、
`model_split`列(model_train/model_validation/inference)を前提とした1系統のみに
単純化した(旧実装は日付単位split用の分岐も持っていたが、本リポジトリは
dataset_id単位分割のみを使うため不要)。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .constants import EMBEDDING_DIM, RANDOM_SEED, SAMPLES_PER_BIN
from .model import WaveformAutoencoder
from .waveforms import open_waveforms


class MemmapWaveformDataset(Dataset):
    def __init__(self, waveform_path, total_count, indices, mean, std):
        self.waveforms = open_waveforms(waveform_path, total_count)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        waveform = np.asarray(self.waveforms[int(self.indices[index])], dtype=np.float32).copy()
        waveform = (waveform - self.mean) / self.std
        return torch.from_numpy(waveform).unsqueeze(0)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_normalization(waveform_path, total_count, indices, chunk_size=4096):
    """`indices`(=model_trainのwaveform_index)のみから平均・標準偏差を計算する。

    model_validation/inferenceの波形はここに絶対に混ぜない
    (SPEC.md 2.3 チェック項目6: Autoencoderの学習データ範囲)。
    """
    waveforms = open_waveforms(waveform_path, total_count)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("正規化に使用できるmodel_train波形がありません。")
    total = 0.0
    square_total = 0.0
    value_count = 0
    for start in range(0, len(indices), chunk_size):
        chunk = np.asarray(waveforms[indices[start : start + chunk_size]], dtype=np.float64)
        total += float(chunk.sum())
        square_total += float(np.square(chunk).sum())
        value_count += int(chunk.size)
    mean = total / value_count
    variance = max(square_total / value_count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("波形の標準偏差が0または非有限です。")
    return float(mean), std


def _average_loss(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batch_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for waveform in loader:
            waveform = waveform.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(waveform)
            loss = criterion(reconstruction, waveform)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1
    return total_loss / max(batch_count, 1)


def train_autoencoder(
    manifest_path,
    waveform_path,
    artifact_dir,
    *,
    device=None,
    epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=10,
    seed=RANDOM_SEED,
    embedding_dim=EMBEDDING_DIM,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    total_count = len(manifest)
    train_rows = manifest.loc[manifest["model_split"] == "model_train"].copy()
    valid_rows = manifest.loc[manifest["model_split"] == "model_validation"].copy()
    if train_rows.empty or valid_rows.empty:
        raise ValueError("model_trainまたはmodel_validationが空です。")
    train_indices = train_rows["waveform_index"].astype(int)
    valid_indices = valid_rows["waveform_index"].astype(int)

    mean, std = calculate_normalization(waveform_path, total_count, train_indices.to_numpy())

    train_dataset = MemmapWaveformDataset(waveform_path, total_count, train_indices, mean, std)
    valid_dataset = MemmapWaveformDataset(waveform_path, total_count, valid_indices, mean, std)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, num_workers=0, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)

    set_random_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    embedding_dim = int(embedding_dim)
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    model = WaveformAutoencoder(embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    criterion = nn.SmoothL1Loss()
    checkpoint_path = artifact_dir / "autoencoder.pt"
    history = []
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        train_loss = _average_loss(model, train_loader, criterion, device, optimizer=optimizer)
        valid_loss = _average_loss(model, valid_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_loss - 1e-10:
            best_loss = valid_loss
            stale_epochs = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "embedding_dim": embedding_dim, "mean": mean, "std": std, "seed": int(seed)},
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    normalization = {
        "mean": mean,
        "std": std,
        "signal_column": "acc_z[m/s2]",
        "fitted_record_count": int(len(train_indices)),
        "samples_per_waveform": SAMPLES_PER_BIN,
        "split": "model_train",
        "embedding_dim": embedding_dim,
    }
    (artifact_dir / "normalization.json").write_text(json.dumps(normalization, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(artifact_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_valid_loss": best_loss,
        "epochs_completed": len(history),
        "device": str(device),
        "embedding_dim": embedding_dim,
    }


def load_trained_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = WaveformAutoencoder(int(checkpoint.get("embedding_dim", EMBEDDING_DIM))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"])
