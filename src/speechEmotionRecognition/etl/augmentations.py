#!/usr/bin/env python3
"""
Waveform augmentations applied before feature extraction.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import librosa


# ──────────────────────────────────────────────────────────────────────────────
# Ops
# ──────────────────────────────────────────────────────────────────────────────

def _maybe(rng: np.random.Generator, p: float) -> bool:
    return rng.random() < float(p)


def _add_noise(y: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.standard_normal(size=y.shape).astype(np.float32)
    return (y + factor * n).astype(np.float32, copy=False)


def _time_shift(y: np.ndarray, max_frac: float, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    k = rng.integers(-int(max_frac * n), int(max_frac * n) + 1)
    if k == 0:
        return y
    return np.roll(y, k).astype(np.float32, copy=False)


def _pitch_shift(y: np.ndarray, sr: int, semitone_range: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    steps = float(rng.uniform(semitone_range[0], semitone_range[1]))
    out = librosa.effects.pitch_shift(y.astype(np.float32, copy=False), sr=sr, n_steps=steps)
    return out.astype(np.float32, copy=False)


def _time_stretch(y: np.ndarray, rate_range: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    rate = float(rng.uniform(rate_range[0], rate_range[1]))
    if abs(rate - 1.0) < 1e-6:
        return y
    out = librosa.effects.time_stretch(y.astype(np.float32, copy=False), rate=rate)
    return out.astype(np.float32, copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────

def apply_augmentations(y: np.ndarray, sr: int, cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    x = y.astype(np.float32, copy=False)

    # Time stretch
    tstr = cfg.get("time_stretch", {})
    if bool(tstr.get("enabled")) and _maybe(rng, float(tstr.get("prob", 0.0))):
        lo, hi = tstr.get("range", [1.0, 1.0])
        x = _time_stretch(x, (float(lo), float(hi)), rng)

    # Pitch shift
    psh = cfg.get("pitch_shift", {})
    if bool(psh.get("enabled")) and _maybe(rng, float(psh.get("prob", 0.0))):
        lo, hi = psh.get("range", [0.0, 0.0])
        x = _pitch_shift(x, sr, (float(lo), float(hi)), rng)

    # Time shift
    tsh = cfg.get("time_shift", {})
    if bool(tsh.get("enabled")) and _maybe(rng, float(tsh.get("prob", 0.0))):
        x = _time_shift(x, float(tsh.get("max", 0.0)), rng)

    # Noise
    noi = cfg.get("noise", {})
    if bool(noi.get("enabled")) and _maybe(rng, float(noi.get("prob", 0.0))):
        x = _add_noise(x, float(noi.get("factor", 0.0)), rng)

    x = np.nan_to_num(x, copy=False)
    return np.clip(x, -1.0, 1.0, out=x).astype(np.float32, copy=False)
