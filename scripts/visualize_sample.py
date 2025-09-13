#!/usr/bin/env python3
"""
Visualize one sample through the SER pipeline.

Panels (saved separately):
- Waveforms: raw (orig SR) -> trimmed (orig SR) -> resampled (target SR) -> fixed (centered) -> prepared (fixed+peak-norm+clipped)
- Features from disk (post-norm): Mel / MFCC / Chroma / SSL
- Optional recomputed pre-norm spectra from the prepared waveform
- Optional N augmentation previews: waveform + pre-norm spectra (and optional SSL)

Examples:
python scripts/visualize_sample.py --config spectral_crnn.yaml --row-idx 4
python scripts/visualize_sample.py --config ssl_spectral_moe.yaml --split val \
       --base-id crema_1001_IOM_ANG_XX --show-ssl-dims 96 --save --no-show
python scripts/visualize_sample.py --config spectral_crnn.yaml --row-idx 0 \
       --preview-aug 2 --show-prenorm --save
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from speechEmotionRecognition.config_loader import load_config
from speechEmotionRecognition.etl.feature_generator import (
    _stft_params,
    _mel as mel_fn,
    _mfcc_from_logmel,
    _chroma as chroma_fn,
    _resample as fg_resample,
    _fix_length_center as fg_fix_length_center,
    _ssl_features as fg_ssl_features,
)
from speechEmotionRecognition.etl.augmentations import apply_augmentations


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_path(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

def _split_csv_path(cfg: Dict, split: str) -> Path:
    key = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}[split]
    return Path(cfg["project"][key])

def _feature_dir(cfg: Dict) -> Path:
    return Path(cfg["project"]["features_dir"])

def _load_disk_feature(feature_dir: Path, feat: str, base_id: str) -> Optional[np.ndarray]:
    p = feature_dir / f"{feat}_{base_id}.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D array in {p}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)

def _find_row(
    cfg: Dict,
    split: str,
    df_split: pd.DataFrame,
    df_meta: pd.DataFrame,
    row_idx: Optional[int],
    base_id: Optional[str],
    search_all_splits: bool = True,
) -> Tuple[pd.Series, str, str]:
    """
    Return (row_from_split, wav_path, split_used).
    If base_id isn't in the requested split and search_all_splits=True, search other splits.
    """
    if base_id is not None:
        m = df_split[df_split["base_id"] == base_id]
        if m.empty and search_all_splits:
            for other in ("train", "val", "test"):
                if other == split:
                    continue
                other_df = pd.read_csv(_safe_path(_split_csv_path(cfg, other)))
                mm = other_df[other_df["base_id"] == base_id]
                if not mm.empty:
                    print(f"[INFO] base_id '{base_id}' not in '{split}', using split '{other}'.")
                    df_split = mm
                    split = other
                    break
            else:
                raise ValueError(f"base_id '{base_id}' not found in any split.")
        row = (df_split[df_split["base_id"] == base_id].iloc[0]
               if not df_split[df_split["base_id"] == base_id].empty
               else df_split.iloc[0])
    else:
        row = df_split.iloc[int(row_idx or 0)]

    # Recover original wav path from metadata
    bid = str(row["base_id"])
    if "_" not in bid:
        raise ValueError(f"Unexpected base_id format: {bid}")
    ds, stem = bid.split("_", 1)
    ds = ds.lower()

    meta_ds = df_meta[df_meta["dataset"].astype(str).str.lower() == ds].copy()
    meta_ds["stem"] = meta_ds["filepath"].map(lambda s: Path(s).stem)
    m = meta_ds[meta_ds["stem"] == stem]
    if m.empty:
        raise ValueError(f"Could not locate original WAV for base_id '{bid}'.")
    wav_path = str(m["filepath"].iloc[0])
    return row, wav_path, split


# ──────────────────────────────────────────────────────────────────────────────
# Audio prep (mirror ETL order)
# ──────────────────────────────────────────────────────────────────────────────

def _waveform_stages(wav_path: str, cfg: Dict) -> Tuple[Dict[str, Tuple[np.ndarray, int]], int]:
    """
    Return ordered stages:
      raw       : (y_raw, file_sr)
      trimmed   : (y_trimmed, file_sr)
      resampled : (y_resampled, target_sr)
      fixed     : (y_centered, target_sr)
      prepared  : (fixed + peak-norm + clipped, target_sr)
    """
    target_sr = int(cfg["audio"]["sampling_rate"])
    dur = float(cfg["audio"]["duration"])
    tgt_len = int(round(target_sr * dur))

    y_raw, file_sr = librosa.load(wav_path, sr=None, mono=True)
    if bool(cfg["audio"]["trim_silence"]["enabled"]):
        y_trim, _ = librosa.effects.trim(y_raw, top_db=int(cfg["audio"]["trim_silence"]["top_db"]))
    else:
        y_trim = y_raw

    y_resamp = fg_resample(y_trim, file_sr, target_sr)
    y_fixed = fg_fix_length_center(y_resamp, tgt_len).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y_fixed))) or 1.0
    y_prep = (y_fixed / peak).astype(np.float32, copy=False)
    y_prep = np.clip(y_prep, -1.0, 1.0, out=y_prep)

    stages = {
        "raw": (y_raw.astype(np.float32, copy=False), file_sr),
        "trimmed": (y_trim.astype(np.float32, copy=False), file_sr),
        "resampled": (y_resamp.astype(np.float32, copy=False), target_sr),
        "fixed": (y_fixed, target_sr),
        "prepared": (y_prep, target_sr),
    }
    return stages, target_sr


# ──────────────────────────────────────────────────────────────────────────────
# Feature recompute (for pre-norm and aug previews)
# ──────────────────────────────────────────────────────────────────────────────

def _recompute_spectra(y: np.ndarray, sr: int, cfg: Dict) -> Dict[str, np.ndarray]:
    """Compute Mel/MFCC/Chroma (pre-norm) with ETL STFT params."""
    n_fft, hop, win, window = _stft_params(cfg)
    out: Dict[str, np.ndarray] = {}

    # Mel (dB)
    n_mels = int(cfg["features"]["mel"]["n_mels"])
    mel = mel_fn(y, sr, n_fft, hop, win, window, n_mels)
    out["mel"] = mel.astype(np.float32, copy=False)

    # MFCC (from log-Mel)
    mcfg = cfg["features"]["mfcc"]
    mfcc = _mfcc_from_logmel(mel, int(mcfg["n_mfcc"]), mcfg.get("lifter", 0), int(mcfg["dct_type"]))
    out["mfcc"] = mfcc.astype(np.float32, copy=False)

    # Chroma
    C = chroma_fn(y, sr, n_fft, hop, win, window)
    out["chroma"] = C.astype(np.float32, copy=False)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _plot_wave(ax, y: np.ndarray, sr: int, title: str):
    t = np.arange(len(y)) / float(sr)
    ax.plot(t, y, linewidth=0.8)
    ax.set_xlim(0, t[-1] if len(t) else 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amp")
    ax.set_title(title)

def _plot_spec(ax, M: np.ndarray, title: str):
    im = ax.imshow(M, aspect="auto", origin="lower")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Bins")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def _plot_ssl(ax, H: np.ndarray, show_dims: int, title: str):
    D, T = H.shape
    d = min(int(show_dims), D)
    im = ax.imshow(H[:d, :], aspect="auto", origin="lower")
    ax.set_xlabel("Frames")
    ax.set_ylabel(f"Dims (showing {d}/{D})")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Visualize a sample through the SER pipeline.")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--row-idx", type=int, default=None, help="Pick by row index (if --base-id not given).")
    ap.add_argument("--base-id", type=str, default=None, help="Pick by base_id (search other splits if needed).")
    ap.add_argument("--show-ssl-dims", type=int, default=96, help="How many SSL dims to display.")
    ap.add_argument("--preview-aug", type=int, default=0, help="# augmented previews to generate.")
    ap.add_argument("--show-prenorm", action="store_true", help="Also plot recomputed pre-norm spectra.")
    ap.add_argument("--aug-ssl", action="store_true", help="Also compute SSL for augmented previews.")
    ap.add_argument("--save", action="store_true", help="Save figures to reports/vis/<base_id>/ ...")
    ap.add_argument("--no-show", dest="no_show", action="store_true", help="Do not open the figures.")
    ap.add_argument("--outdir", type=str, default="reports/vis", help="Root directory for saved figures.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    feature_dir = _feature_dir(cfg)
    meta_csv = Path(cfg["project"]["metadata_csv"])
    df_meta = pd.read_csv(_safe_path(meta_csv))
    df_split = pd.read_csv(_safe_path(_split_csv_path(cfg, args.split)))

    # Pick row + find original waveform
    row, wav_path, used_split = _find_row(cfg, args.split, df_split, df_meta, args.row_idx, args.base_id)
    base_id = str(row["base_id"])
    emotion = str(row.get("emotion", ""))
    dataset = str(row.get("dataset", ""))
    print(f"[INFO] base_id={base_id} | emotion={emotion} | dataset={dataset}")
    print(f"[INFO] wav: {wav_path}")

    # Waveform stages
    stages, target_sr = _waveform_stages(wav_path, cfg)
    y_prep, _ = stages["prepared"]

    # Load disk features (post-norm)
    mel_disk = _load_disk_feature(feature_dir, "mel", base_id)
    mfcc_disk = _load_disk_feature(feature_dir, "mfcc", base_id)
    chroma_disk = _load_disk_feature(feature_dir, "chroma", base_id)
    ssl_disk = _load_disk_feature(feature_dir, "ssl", base_id)

    # Build output directory
    out_root = Path(args.outdir) / base_id
    if args.save:
        out_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Figure 1: Waveform stages
    # -------------------------------
    wave_panels: List[Tuple[str, Tuple[np.ndarray, int]]] = [
        ("Raw (orig SR)", stages["raw"]),
        ("Trimmed (orig SR)", stages["trimmed"]),
        ("Resampled (target SR)", stages["resampled"]),
        ("Fixed (centered)", stages["fixed"]),
        ("Prepared (fixed + peak-norm)", stages["prepared"]),
    ]
    n = len(wave_panels)
    fig_w, axes_w = plt.subplots(n, 1, figsize=(12, max(2.8 * n, 4)), constrained_layout=True)
    if n == 1: axes_w = [axes_w]
    for ax, (title, (yy, sr)) in zip(axes_w, wave_panels):
        _plot_wave(ax, yy, sr, title)
    fig_w.suptitle(f"{base_id}  ({emotion}, {dataset}) — Waveforms", y=1.01, fontsize=12)

    if args.save:
        out_w = out_root / f"{used_split}_{base_id}_waveforms.png"
        fig_w.savefig(out_w, dpi=150, bbox_inches="tight")
        print(f"[OK] figure saved → {out_w}")

    if not args.no_show:
        plt.show()
    plt.close(fig_w)

    # -------------------------------
    # Figure 2: Features from disk (post-norm)
    # -------------------------------
    feat_panels: List[Tuple[str, np.ndarray]] = []
    if mel_disk is not None:
        feat_panels.append(("Mel (disk, post-norm)", mel_disk))
    if mfcc_disk is not None:
        feat_panels.append(("MFCC (disk, post-norm)", mfcc_disk))
    if chroma_disk is not None:
        feat_panels.append(("Chroma (disk, post-norm)", chroma_disk))
    if ssl_disk is not None:
        feat_panels.append((f"SSL (disk) heatmap (first {args.show_ssl_dims} dims)", ssl_disk))

    if feat_panels:
        n = len(feat_panels)
        fig_f, axes_f = plt.subplots(n, 1, figsize=(12, max(3 * n, 4)), constrained_layout=True)
        if n == 1: axes_f = [axes_f]
        for ax, (title, M) in zip(axes_f, feat_panels):
            if "SSL" in title:
                _plot_ssl(ax, M, args.show_ssl_dims, title)
            else:
                _plot_spec(ax, M, title)
        fig_f.suptitle(f"{base_id}  ({emotion}, {dataset}) — Features (disk)", y=1.01, fontsize=12)
        if args.save:
            out_f = out_root / f"{used_split}_{base_id}_features_disk.png"
            fig_f.savefig(out_f, dpi=150, bbox_inches="tight")
            print(f"[OK] figure saved → {out_f}")
        if not args.no_show:
            plt.show()
        plt.close(fig_f)
    else:
        print("[WARN] No on-disk features found to plot.")

    # -------------------------------
    # Figure 3: Recomputed pre-norm spectra (optional)
    # -------------------------------
    if args.show_prenorm:
        pre = _recompute_spectra(y_prep, target_sr, cfg)
        pre_panels: List[Tuple[str, np.ndarray]] = []
        if "mel" in pre: pre_panels.append(("Mel (recomputed, pre-norm)", pre["mel"]))
        if "mfcc" in pre: pre_panels.append(("MFCC (recomputed, pre-norm)", pre["mfcc"]))
        if "chroma" in pre: pre_panels.append(("Chroma (recomputed, pre-norm)", pre["chroma"]))

        if pre_panels:
            n = len(pre_panels)
            fig_p, axes_p = plt.subplots(n, 1, figsize=(12, max(3 * n, 4)), constrained_layout=True)
            if n == 1: axes_p = [axes_p]
            for ax, (title, M) in zip(axes_p, pre_panels):
                _plot_spec(ax, M, title)
            fig_p.suptitle(f"{base_id}  ({emotion}, {dataset}) — Features (recomputed pre-norm)",
                           y=1.01, fontsize=12)
            if args.save:
                out_p = out_root / f"{used_split}_{base_id}_features_prenorm.png"
                fig_p.savefig(out_p, dpi=150, bbox_inches="tight")
                print(f"[OK] figure saved → {out_p}")
            if not args.no_show:
                plt.show()
            plt.close(fig_p)

    # -------------------------------
    # Augmentation previews
    # -------------------------------
    aug_cfg = cfg.get("augmentations", {})
    do_aug = bool(aug_cfg.get("enabled", False)) and int(args.preview_aug) > 0
    if do_aug:
        dur = float(cfg["audio"]["duration"])
        tgt_len = int(round(dur * target_sr))
        for k in range(1, int(args.preview_aug) + 1):
            # Augment prepared waveform (mirror ETL)
            y_aug = apply_augmentations(y_prep, sr=target_sr, cfg=aug_cfg, rng=np.random.default_rng(k))
            y_aug = fg_fix_length_center(y_aug, tgt_len).astype(np.float32, copy=False)
            peak = float(np.max(np.abs(y_aug))) or 1.0
            y_aug = (y_aug / peak).astype(np.float32, copy=False)
            y_aug = np.clip(y_aug, -1.0, 1.0, out=y_aug)

            # Waveform (aug)
            fig_aw, ax_aw = plt.subplots(1, 1, figsize=(12, 3.2), constrained_layout=True)
            _plot_wave(ax_aw, y_aug, target_sr, f"Waveform (aug #{k})")
            fig_aw.suptitle(f"{base_id}  ({emotion}, {dataset}) — Aug #{k} Waveform", y=1.01, fontsize=12)
            if args.save:
                out_aw = out_root / f"{used_split}_{base_id}_waveform_aug{k}.png"
                fig_aw.savefig(out_aw, dpi=150, bbox_inches="tight")
                print(f"[OK] figure saved → {out_aw}")
            if not args.no_show:
                plt.show()
            plt.close(fig_aw)

            # Features (aug, pre-norm)
            pre_aug = _recompute_spectra(y_aug, target_sr, cfg)
            aug_panels: List[Tuple[str, np.ndarray]] = []
            if "mel" in pre_aug: aug_panels.append((f"Mel (aug #{k}, pre-norm)", pre_aug["mel"]))
            if "mfcc" in pre_aug: aug_panels.append((f"MFCC (aug #{k}, pre-norm)", pre_aug["mfcc"]))
            if "chroma" in pre_aug: aug_panels.append((f"Chroma (aug #{k}, pre-norm)", pre_aug["chroma"]))

            # Optional SSL (aug): align to a reliable T_target
            if args.aug_ssl:
                if "mel" in pre_aug:
                    T_target = pre_aug["mel"].shape[1]
                elif "mfcc" in pre_aug:
                    T_target = pre_aug["mfcc"].shape[1]
                elif "chroma" in pre_aug:
                    T_target = pre_aug["chroma"].shape[1]
                elif mel_disk is not None:
                    T_target = mel_disk.shape[1]
                elif mfcc_disk is not None:
                    T_target = mfcc_disk.shape[1]
                elif chroma_disk is not None:
                    T_target = chroma_disk.shape[1]
                else:
                    T_target = None

                if T_target is not None:
                    try:
                        H_aug = fg_ssl_features(
                            y_aug, target_sr, T_target, cfg["features"]["ssl"],
                            device_pref=str(cfg.get("device", "auto"))
                        )
                        aug_panels.append((f"SSL (aug #{k}) heatmap (first {args.show_ssl_dims} dims)", H_aug))
                    except Exception as e:
                        print(f"[WARN] SSL for aug #{k} failed: {e}")
                else:
                    print(f"[WARN] Could not determine T_target for SSL (aug #{k}); skipping SSL panel.")

            if aug_panels:
                n = len(aug_panels)
                fig_af, axes_af = plt.subplots(n, 1, figsize=(12, max(3 * n, 4)), constrained_layout=True)
                if n == 1: axes_af = [axes_af]
                for ax, (title, M) in zip(axes_af, aug_panels):
                    if "SSL" in title:
                        _plot_ssl(ax, M, args.show_ssl_dims, title)
                    else:
                        _plot_spec(ax, M, title)
                fig_af.suptitle(f"{base_id}  ({emotion}, {dataset}) — Aug #{k} Features",
                                y=1.01, fontsize=12)
                if args.save:
                    out_af = out_root / f"{used_split}_{base_id}_features_aug{k}.png"
                    fig_af.savefig(out_af, dpi=150, bbox_inches="tight")
                    print(f"[OK] figure saved → {out_af}")
                if not args.no_show:
                    plt.show()
                plt.close(fig_af)


if __name__ == "__main__":
    main()
