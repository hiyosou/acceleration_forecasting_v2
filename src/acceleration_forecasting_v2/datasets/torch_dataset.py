"""学習用PyTorch Dataset。`acceleration_forecasting_12m/.../datasets/torch_dataset.py`の
6か月統一履歴・target_mode=absolute・segment_weight対応版。

旧実装との主な差分:
- "current"/"history"の分離を廃止し、"history_values"(6,)に統合(SPEC.md 1.1)。
- "segment_weight"をitemに追加(SPEC.md 1.1、損失関数で使用)。
- target_mode=absoluteのみを扱うため、residualモードの分岐(guide_baseline加算)は
  存在しない(SPEC.md 1.6)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .build import ARRAY_NAMES
from .cycle_features import CYCLE_FEATURE_NAME, NORMALIZATION_FILE, normalize_cycle_features
from .normalization import Normalization


class ForecastDatasetV2(Dataset):
    """`.npy`一式(build_raw_split相当の出力を保存したディレクトリ)を読み込むDataset。"""

    def __init__(self, split_dir, dataset_root, *, include_targets: bool = True):
        self.path, self.root = Path(split_dir), Path(dataset_root)
        input_arrays = [name for name in ARRAY_NAMES if name not in ("target_values", "target_masks")]
        self.arrays = {name: np.load(self.path / f"{name}.npy", mmap_mode="r") for name in input_arrays}
        self.include_targets = bool(include_targets)
        if include_targets:
            self.targets = np.load(self.path / "target_values.npy", mmap_mode="r")
            self.target_masks = np.load(self.path / "target_masks.npy", mmap_mode="r")
            # 合成値(datasets.extension)で埋めた月の印。古いデータセットには無い(=全て実測)。
            synthetic_path = self.path / "target_synthetic_masks.npy"
            self.target_synthetic_masks = np.load(synthetic_path, mmap_mode="r") if synthetic_path.is_file() else None
        self.condition_norm = Normalization.load(self.root / "condition_normalization.json")
        self.target_norm = Normalization.load(self.root / "target_normalization.json")
        # 忘却因子Dの要約(datasets.cycle_features、2026-10-27)。古いデータセットには無い。
        features_path = self.path / f"{CYCLE_FEATURE_NAME}.npy"
        self.cycle_features = None
        if features_path.is_file():
            self.cycle_features = np.load(features_path, mmap_mode="r")
            self.cycle_feature_norm = json.loads((self.root / NORMALIZATION_FILE).read_text(encoding="utf-8"))

    def __len__(self):
        return len(self.arrays["history_values"])

    @staticmethod
    def _finite_zero(values):
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __getitem__(self, index):
        history = self._finite_zero(self.condition_norm.normalize(self.arrays["history_values"][index]))
        guide_values = self._finite_zero(self.condition_norm.normalize(self.arrays["guide_values"][index]))
        guide_deltas = self._finite_zero(np.asarray(self.arrays["guide_deltas"][index]) / self.condition_norm.std)
        item = {
            "history_values": torch.from_numpy(history.copy()),
            "history_masks": torch.from_numpy(self.arrays["history_masks"][index].copy().astype(np.float32)),
            "guide_values": torch.from_numpy(guide_values.copy()),
            "guide_deltas": torch.from_numpy(guide_deltas.copy()),
            "guide_mask": torch.from_numpy(self.arrays["guide_masks"][index].copy().astype(np.float32)),
            "guide_similarities": torch.from_numpy(self.arrays["guide_similarities"][index].copy().astype(np.float32)),
            "retrieval_mask": torch.from_numpy(self.arrays["retrieval_masks"][index].copy().astype(np.float32)),
            "segment_weight": torch.tensor(float(self.arrays["segment_weights"][index]), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.cycle_features is not None:
            item["cycle_features"] = torch.from_numpy(self._finite_zero(
                normalize_cycle_features(self.cycle_features[index], self.cycle_feature_norm)).copy())
        if self.include_targets:
            target = self._finite_zero(self.target_norm.normalize(self.targets[index]))
            item["target"] = torch.from_numpy(target.copy())
            item["target_mask"] = torch.from_numpy(self.target_masks[index].copy().astype(np.float32))
            if self.target_synthetic_masks is not None:
                item["target_synthetic_mask"] = torch.from_numpy(
                    self.target_synthetic_masks[index].copy().astype(np.float32))
        return item

    def denormalize_target(self, values):
        return self.target_norm.denormalize(values)

    def physical_prediction(self, target_normalized, bounds=None):
        """target_mode=absoluteのため、guide_baselineの加算は行わない(SPEC.md 1.6)。"""
        values = self.denormalize_target(target_normalized)
        if bounds is not None:
            values = np.clip(values, float(bounds[0]), float(bounds[1]))
        return values

    def physical_target(self, index):
        if not self.include_targets:
            raise ValueError("Targets are not loaded")
        return np.asarray(self.targets[index], dtype=np.float32)
