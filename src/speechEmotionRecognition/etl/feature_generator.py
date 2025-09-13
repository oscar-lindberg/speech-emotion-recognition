#!/usr/bin/env python3
"""
Extract SER features (mel/mfcc/chroma/ssl) and write a manifest.
Output CSV columns: base_id, emotion, dataset, speaker_id, augmented
"""
from __future__ import annotations
import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from speechEmotionRecognition.config_loader import load_config
from .augmentations import apply_augmentations


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _safe_mkdirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    try:
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)  # librosa>=0.10
    except TypeError:
        return librosa.resample(y, orig_sr, target_sr)  # librosa 0.9.x


def _fix_length_center(y: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop/pad waveform to exact target length (in samples)."""
    n = len(y)
    if n == target_len:
        return y
    if n > target_len:
        s = (n - target_len) // 2
        return y[s:s + target_len]
    pad = target_len - n
    l = pad // 2
    return np.pad(y, (l, pad - l))


def _power_to_db(S: np.ndarray) -> np.ndarray:
    return librosa.power_to_db(S, ref=np.max)


def _cmvn_rows(X: np.ndarray) -> np.ndarray:
    """Cepstral Mean/Variance Normalization row-wise (per frequency channel)."""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / np.where(s > 1e-5, s, 1.0)


def _resample_time(M: np.ndarray, T: int) -> np.ndarray:
    """Linear time-resampling to force exact T frames (non-SSL only)."""
    F, t = M.shape
    if t == T:
        return M
    x_old = np.linspace(0.0, 1.0, num=t, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=T, endpoint=True)
    out = np.empty((F, T), dtype=M.dtype)
    for i in range(F):
        out[i] = np.interp(x_new, x_old, M[i])
    return out


def _align_time(M: np.ndarray, T: int, is_ssl: bool) -> np.ndarray:
    if is_ssl:
        F, t = M.shape
        if t == T:
            return M
        if t > T:  # center-crop
            s = (t - T) // 2
            return M[:, s:s + T]
        # center-pad
        pad = T - t
        l = pad // 2
        return np.pad(M, ((0, 0), (l, pad - l)))
    # spectral: resample to T
    return _resample_time(M, T)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr.astype(np.float32, copy=False))
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# STFT params (from YAML)
# ──────────────────────────────────────────────────────────────────────────────

def _stft_params(cfg: Dict) -> Tuple[int, int, Optional[int], str]:
    stft = cfg["features"]["stft"]
    n_fft = int(stft["n_fft"])
    hop = int(stft["hop_length"])
    win = int(stft["win_length"]) if stft.get("win_length") is not None else None
    window = str(stft["window"])
    return n_fft, hop, win, window


# ──────────────────────────────────────────────────────────────────────────────
# SSL backend (lazy, AutoFeatureExtractor only)
# ──────────────────────────────────────────────────────────────────────────────

_SSL = {"model": None, "fe": None, "device": "cpu"}

def _resolve_device(pref: str) -> str:
    pref = pref.lower()
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _lazy_load_ssl(ssl_cfg: Dict, device_pref: str):
    if _SSL["model"] is not None:
        return _SSL["model"], _SSL["fe"], _SSL["device"]

    if str(ssl_cfg.get("backend", "")).lower() != "transformers":
        raise ValueError("Only 'transformers' backend is supported.")
    model_id = ssl_cfg["model_id"]
    device = _resolve_device(device_pref)

    import torch
    from transformers import AutoModel
    # Prefer AutoFeatureExtractor, fall back to AutoProcessor
    try:
        from transformers import AutoFeatureExtractor as _FE
    except Exception:
        try:
            from transformers import AutoProcessor as _FE  # type: ignore
        except Exception as e:
            raise ImportError(
                "Neither AutoFeatureExtractor nor AutoProcessor is available from transformers."
            ) from e

    logging.info("Loading SSL model '%s' on %s ...", model_id, device)
    model = AutoModel.from_pretrained(model_id)
    fe = _FE.from_pretrained(model_id)  # works for both extractor/processor

    model.to(torch.device(device))
    model.eval()

    _SSL.update(model=model, fe=fe, device=device)
    return model, fe, device


def _ssl_num_frames(sr: int, duration_s: float, ssl_cfg: Dict, device_pref: str) -> int:
    """Probe model with zeros to get its implied time length."""
    import torch
    model, fe, device = _lazy_load_ssl(ssl_cfg, device_pref)
    y = np.zeros(int(round(sr * duration_s)), dtype=np.float32)
    with torch.no_grad():
        inputs = fe(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: torch.as_tensor(v).to(device) for k, v in inputs.items()}
        out = model(**inputs)
        return int(out.last_hidden_state.shape[1])


def _ssl_features(y: np.ndarray, sr: int, T_target: int, ssl_cfg: Dict, device_pref: str) -> np.ndarray:
    import torch
    model, fe, device = _lazy_load_ssl(ssl_cfg, device_pref)
    with torch.no_grad():
        inputs = fe(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: torch.as_tensor(v).to(device) for k, v in inputs.items()}
        H = model(**inputs).last_hidden_state.squeeze(0).detach().cpu().numpy().T  # [D, T_ssl]
    return _align_time(H, T_target, is_ssl=True)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extractors
# ──────────────────────────────────────────────────────────────────────────────

def _mel(y, sr, n_fft, hop, win, window, n_mels) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win, window=window, power=2.0, n_mels=n_mels
    )
    return _power_to_db(S)


def _mfcc_from_logmel(logmel: np.ndarray, n_mfcc: int, lifter: Optional[int], dct_type: int) -> np.ndarray:
    # Robust to lifter=None in YAML (treated as 0)
    lifter_val = 0 if lifter is None else int(lifter)
    return librosa.feature.mfcc(S=logmel, n_mfcc=int(n_mfcc), lifter=lifter_val, dct_type=int(dct_type))


def _chroma(y, sr, n_fft, hop, win, window) -> np.ndarray:
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, win_length=win, window=window)) ** 2
    return librosa.feature.chroma_stft(S=S, sr=sr)


# ──────────────────────────────────────────────────────────────────────────────
# High-level helpers (readability)
# ──────────────────────────────────────────────────────────────────────────────

def load_audio_for_row(row: pd.Series, cfg: Dict) -> Tuple[str, np.ndarray, int]:
    """
    Load + trim + resample + center to duration + peak-normalize + clip.
    Returns (base_id, y_prepared, sr).
    """
    wav_path = row["filepath"]
    dataset = str(row["dataset"]).lower()
    stem = Path(wav_path).stem
    base_id = f"{dataset}_{stem}"

    sr = int(cfg["audio"]["sampling_rate"])
    dur = float(cfg["audio"]["duration"])
    tgt_len = int(round(dur * sr))

    y, file_sr = librosa.load(wav_path, sr=None, mono=True)
    if bool(cfg["audio"]["trim_silence"]["enabled"]):
        y, _ = librosa.effects.trim(y, top_db=int(cfg["audio"]["trim_silence"]["top_db"]))
    y = _resample(y, file_sr, sr)
    y = _fix_length_center(y, tgt_len).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak).astype(np.float32, copy=False)
    y = np.clip(y, -1.0, 1.0, out=y)  # keep safe range
    return base_id, y, sr


def target_frames(cfg: Dict, features: List[str], n_fft: int, hop: int, win: Optional[int], window: str) -> int:
    """
    Compute target frame count T:
      - For spectral features, use mel spectrogram of zeros (no dB) to get shape.
      - For SSL-only, probe the model on zeros.
    """
    wants_spectral = any(f in {"mel", "mfcc", "chroma"} for f in features)
    sr = int(cfg["audio"]["sampling_rate"])
    dur = float(cfg["audio"]["duration"])

    if wants_spectral:
        dummy = np.zeros(int(round(dur * sr)), dtype=np.float32)
        S = librosa.feature.melspectrogram(
            y=dummy, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win, window=window, power=2.0,
            n_mels=int(cfg["features"]["mel"]["n_mels"])
        )
        return int(S.shape[1])
    return _ssl_num_frames(sr, dur, cfg["features"]["ssl"], str(cfg.get("device", "auto")))


def extract_feature_mats(y: np.ndarray, sr: int, cfg: Dict, features: List[str], n_fft: int, hop: int, win: Optional[int],
                         window: str, T_target: int, device_pref: str, norm_cfg: Dict[str, bool]) -> List[Tuple[str, np.ndarray]]:
    """
    Compute requested feature matrices from prepared waveform y.
    Returns list of (feature_name, matrix[Freq, T_target]).
    """
    mats: List[Tuple[str, np.ndarray]] = []
    mel_cache: Optional[np.ndarray] = None

    for feat in features:
        if feat == "mel":
            mel = _mel(y, sr, n_fft, hop, win, window, int(cfg["features"]["mel"]["n_mels"]))
            if norm_cfg.get("mel", False):
                mel = _cmvn_rows(mel)
            mel_cache = mel
            mats.append((feat, _align_time(mel, T_target, is_ssl=False)))

        elif feat == "mfcc":
            if mel_cache is None:
                mel_cache = _mel(y, sr, n_fft, hop, win, window, int(cfg["features"]["mel"]["n_mels"]))
            mcfg = cfg["features"]["mfcc"]
            mfcc = _mfcc_from_logmel(mel_cache, int(mcfg["n_mfcc"]), mcfg.get("lifter", 0), int(mcfg["dct_type"]))
            if norm_cfg.get("mfcc", False):
                mfcc = _cmvn_rows(mfcc)
            mats.append((feat, _align_time(mfcc, T_target, is_ssl=False)))

        elif feat == "chroma":
            C = _chroma(y, sr, n_fft, hop, win, window)
            if norm_cfg.get("chroma", False):
                C = _cmvn_rows(C)
            mats.append((feat, _align_time(C, T_target, is_ssl=False)))

        elif feat == "ssl":
            H = _ssl_features(y, sr, T_target, cfg["features"]["ssl"], device_pref=device_pref)
            mats.append((feat, H))

        else:
            raise RuntimeError(f"Unknown feature: {feat}")

    return mats


def save_feature_mats(feature_dir: Path, base_id: str, mats: List[Tuple[str, np.ndarray]]) -> List[str]:
    paths: List[str] = []
    for feat, M in mats:
        out = feature_dir / f"{feat}_{base_id}.npy"
        _save_npy_atomic(out, M)
        paths.append(str(out))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Extract features and write features manifest.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO, -vv for DEBUG.")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config)

    feature_dir = Path(cfg["project"]["features_dir"])
    meta_csv = Path(cfg["project"]["metadata_csv"])
    out_manifest = Path(cfg["project"]["features_metadata_csv"])

    features: List[str] = list(cfg["features"]["types"])
    aug_features: List[str] = list(cfg["features"]["augmented_types"])
    norm_cfg: Dict[str, bool] = dict(cfg["features"]["normalize"])
    n_fft, hop, win, window = _stft_params(cfg)

    _safe_mkdirs(feature_dir)
    if not meta_csv.is_file():
        raise FileNotFoundError(str(meta_csv))

    # Determine target time length once for consistency
    T_target = target_frames(cfg, features, n_fft, hop, win, window)
    logging.info("Target frames T=%d", T_target)

    device_pref = str(cfg.get("device", "auto"))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    # Load metadata
    df = pd.read_csv(meta_csv)
    req_cols = {"filepath", "emotion", "dataset", "speaker_id"}
    miss = req_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"{meta_csv} missing columns: {sorted(miss)}")
    df = df.sort_values(["dataset", "speaker_id", "emotion", "filepath"]).reset_index(drop=True)

    # Quick helper for skipping existing bases
    def _all_exist(base_id: str, feats: List[str]) -> bool:
        return all((feature_dir / f"{f}_{base_id}.npy").is_file() for f in feats)

    entries: List[Dict] = []

    # Main loop
    for _, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True, leave=False, desc="Features"):
        base_id, y0, sr = load_audio_for_row(row, cfg)

        # Skip existing base features (note: aug also skipped by design)
        if bool(cfg["features"]["skip_existing"]) and _all_exist(base_id, features):
            entries.append({
                "base_id": base_id,
                "emotion": row["emotion"],
                "dataset": row["dataset"],
                "speaker_id": row["speaker_id"],
                "augmented": 0,
            })
            continue

        # Base features
        mats = extract_feature_mats(y0, sr, cfg, features, n_fft, hop, win, window, T_target, device_pref, norm_cfg)
        save_feature_mats(feature_dir, base_id, mats)
        entries.append({
            "base_id": base_id,
            "emotion": row["emotion"],
            "dataset": row["dataset"],
            "speaker_id": row["speaker_id"],
            "augmented": 0,
        })

        # Augmented copies (reuse y0; variability via RNG and ops)
        aug_cfg = cfg["augmentations"]
        if bool(aug_cfg["enabled"]) and int(aug_cfg["copies"]) > 0 and aug_features:
            dur = float(cfg["audio"]["duration"])
            tgt_len = int(round(dur * sr))  # sr already equals target sr
            for k in range(int(aug_cfg["copies"])):
                y_aug = apply_augmentations(y0, sr=sr, cfg=aug_cfg, rng=rng)
                y_aug = _fix_length_center(y_aug, tgt_len).astype(np.float32, copy=False)
                peak = float(np.max(np.abs(y_aug))) or 1.0
                y_aug = (y_aug / peak).astype(np.float32, copy=False)
                y_aug = np.clip(y_aug, -1.0, 1.0, out=y_aug)

                mats_aug = extract_feature_mats(y_aug, sr, cfg, aug_features, n_fft, hop, win, window,
                                                T_target, device_pref, norm_cfg)
                aug_id = f"{base_id}_aug{k+1}"
                save_feature_mats(feature_dir, aug_id, mats_aug)
                entries.append({
                    "base_id": aug_id,
                    "emotion": row["emotion"],
                    "dataset": row["dataset"],
                    "speaker_id": row["speaker_id"],
                    "augmented": 1,
                })

    # Write manifest
    _safe_mkdirs(out_manifest.parent)
    tmp = out_manifest.with_suffix(".tmp.csv")
    pd.DataFrame(entries).to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, out_manifest)

    logging.info("Wrote %d rows to %s", len(entries), out_manifest)
    logging.info("Feature dir: %s", feature_dir)


if __name__ == "__main__":
    main()
