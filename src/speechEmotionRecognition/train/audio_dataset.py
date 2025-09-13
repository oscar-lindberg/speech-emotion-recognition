#!/usr/bin/env python3
"""
Dataset over precomputed SER features.

For each row, loads <feature>_<base_id>.npy from `feature_dir` and returns:
  x: FloatTensor [N, F_max, T] stacked in the order of `features`
  y: int label
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_feature(path: Path) -> np.ndarray:
    """Load a single feature map saved as float32 [F, T]."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D feature at {path}, got shape {arr.shape}")
    return arr

def _stack_features(
    parts: List[np.ndarray],
    want_order: List[str],
    shapes: Dict[str, Tuple[int, int]],
) -> torch.Tensor:
    """Zero-pad each [F_i, T] to [F_max, T] and stack to [N, F_max, T]."""
    assert len(parts) == len(want_order)
    T = next(iter(shapes.values()))[1]
    F_max = max(F for F, _ in shapes.values())
    x = torch.zeros((len(parts), F_max, T), dtype=torch.float32)
    for i, name in enumerate(want_order):
        Fi, Ti = shapes[name]
        if parts[i].shape != (Fi, Ti):
            raise RuntimeError(f"{name}: expected {(Fi, Ti)}, got {parts[i].shape}")
        x[i, :Fi, :Ti] = torch.from_numpy(parts[i])
    return x

def _apply_specaugment_inplace(x: torch.Tensor, time_w: int, time_n: int, freq_w: int, freq_n: int) -> None:
    """SpecAugment (time/freq masks) on x ∈ ℝ^{F×T}; in-place zeroing."""
    F, T = x.shape
    rng = np.random.default_rng()
    for _ in range(int(freq_n)):
        w = int(min(max(0, freq_w), F))
        if w:
            f0 = int(rng.integers(0, F - w + 1))
            x[f0:f0 + w, :] = 0.0
    for _ in range(int(time_n)):
        w = int(min(max(0, time_w), T))
        if w:
            t0 = int(rng.integers(0, T - w + 1))
            x[:, t0:t0 + w] = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """
    Precomputed-feature dataset.
    """
    def __init__(
        self,
        csv_path: str,
        feature_dir: str,
        features: List[str],
        label_map: Dict[str, int],
        mode: str = "train",
        specaugment_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.feature_dir = Path(feature_dir)
        self.features = [str(f).lower() for f in features]
        self.label_map = {str(k).lower(): int(v) for k, v in label_map.items()}
        self.mode = str(mode).lower()
        self.spec_cfg = specaugment_cfg if str(mode).lower() == "train" else None

        if not self.csv_path.is_file():
            raise FileNotFoundError(str(self.csv_path))
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(str(self.feature_dir))

        df = pd.read_csv(self.csv_path)
        needed_cols = {"base_id", "dataset", "speaker_id", "augmented"}
        missing = needed_cols - set(df.columns)
        if missing:
            raise RuntimeError(f"{self.csv_path} missing columns: {sorted(missing)}")

        if "label" in df.columns:
            labels = df["label"].astype(int).to_numpy()
        elif "emotion" in df.columns:
            labels = df["emotion"].astype(str).str.lower().map(self.label_map).astype(int).to_numpy()
        else:
            raise RuntimeError(f"{self.csv_path} must contain 'label' or 'emotion' column.")

        self.rows = list(zip(df["base_id"].astype(str).to_numpy(), labels))
        if not self.rows:
            raise RuntimeError(f"No rows found in split: {self.csv_path}")

        # Infer per-branch shapes from the first row (and time axis T reference).
        base0 = self.rows[0][0]
        self._feature_shapes: Dict[str, Tuple[int, int]] = {}
        for name in self.features:
            arr = _load_feature(self.feature_dir / f"{name}_{base0}.npy")
            self._feature_shapes[name] = (int(arr.shape[0]), int(arr.shape[1]))
        self._T = next(iter(self._feature_shapes.values()))[1]  # common T across branches

        # Validate SpecAugment config shape (if enabled)
        if self.spec_cfg is not None:
            for top in ("time_mask", "freq_mask"):
                if top not in self.spec_cfg or not isinstance(self.spec_cfg[top], dict):
                    raise KeyError(f"specaugment missing '{top}' block")
                for k in ("param", "num"):
                    if k not in self.spec_cfg[top]:
                        raise KeyError(f"specaugment.{top} missing '{k}'")

    # Public API ---------------------------------------------------------------

    @property
    def feature_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Per-branch shapes {name: (F_i, T)} discovered from disk."""
        return dict(self._feature_shapes)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        base_id, y = self.rows[idx]

        parts: List[np.ndarray] = []
        for name in self.features:
            path = self.feature_dir / f"{name}_{base_id}.npy"
            arr = _load_feature(path)

            Fi, Ti = self._feature_shapes[name]
            Fi2, Ti2 = arr.shape
            if Ti2 != self._T:
                if Ti2 > self._T:
                    t0 = (Ti2 - self._T) // 2
                    arr = arr[:, t0:t0 + self._T]
                else:
                    pad = self._T - Ti2
                    l = pad // 2
                    arr = np.pad(arr, ((0, 0), (l, pad - l)), mode="constant")
                Fi2, Ti2 = arr.shape
            if (Fi2, Ti2) != (Fi, self._T):
                raise RuntimeError(f"Unexpected shape for {name}: got {(Fi2, Ti2)}, expected {(Fi, self._T)}")

            parts.append(arr.astype(np.float32, copy=False))

        # Stack to [N, F_max, T]
        x = _stack_features(parts, self.features, self._feature_shapes)

        # Optional SpecAugment (skip SSL branch)
        if self.spec_cfg is not None:
            tm_w = int(self.spec_cfg["time_mask"]["param"])
            tm_n = int(self.spec_cfg["time_mask"]["num"])
            fm_w = int(self.spec_cfg["freq_mask"]["param"])
            fm_n = int(self.spec_cfg["freq_mask"]["num"])
            for i, name in enumerate(self.features):
                if name == "ssl":
                    continue
                _apply_specaugment_inplace(x[i], time_w=tm_w, time_n=tm_n, freq_w=fm_w, freq_n=fm_n)

        return x, int(y)
