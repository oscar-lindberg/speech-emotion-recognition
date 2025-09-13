#!/usr/bin/env python3
"""
Extract raw audio metadata into a unified CSV.
Output CSV columns: filepath, emotion, dataset, speaker_id
"""
from __future__ import annotations
import os
import csv
import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Dict, Set, Callable
import pandas as pd
from speechEmotionRecognition.config_loader import load_config


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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_wav(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in {".wav", ".wave"}


def _safe_scandir(path: Path) -> Iterable[os.DirEntry]:
    try:
        return os.scandir(path)
    except FileNotFoundError:
        logging.warning("Directory not found: %s", path)
        return ()


def _normalize_map(m: Dict[str, str]) -> Dict[str, str]:
    """Lowercase keys+values so filename tokens can be matched case-insensitively."""
    return {str(k).lower(): str(v).lower() for k, v in m.items()}


def _resolve_datasets(cfg: Dict) -> List[str]:
    """Use the order given in the config; warn and drop unknowns."""
    requested = [str(d).lower() for d in cfg["extract"]["datasets"]]
    known = set(PARSERS.keys())
    unknown = [d for d in requested if d not in known]
    if unknown:
        logging.warning("Ignoring unknown dataset(s): %s (known: %s)", unknown, sorted(known))
    return [d for d in requested if d in known]


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────

def parse_crema(root: Path, keep: Set[str], emap: Dict[str, str]) -> pd.DataFrame:
    """CREMA-D: <ID>_<...>_<EMOTION>_<...>.wav (emotion is 3rd token)."""
    rows = []
    for de in _safe_scandir(root):
        if not de.is_file() or not _is_wav(de.name):
            continue
        parts = de.name.split("_")
        if len(parts) < 3:
            continue
        code = parts[2].lower()
        emotion = emap.get(code)
        if emotion in keep:
            speaker_id = parts[0]
            rows.append({"filepath": str(Path(de.path)), "emotion": emotion, "dataset": "crema", "speaker_id": speaker_id})
    return pd.DataFrame(rows)


def parse_ravdess(root: Path, keep: Set[str], emap: Dict[str, str]) -> pd.DataFrame:
    """RAVDESS: 03-01-05-01-02-01-12.wav (emotion is 3rd token)."""
    rows = []
    for actor in _safe_scandir(root):
        if not actor.is_dir():
            continue
        speaker_id = actor.name  # e.g., 'Actor_12'
        for de in _safe_scandir(Path(actor.path)):
            if not de.is_file() or not _is_wav(de.name):
                continue
            parts = de.name.split("-")
            if len(parts) < 3:
                continue
            code = parts[2]
            emotion = emap.get(code)
            if emotion in keep:
                rows.append({"filepath": str(Path(de.path)), "emotion": emotion, "dataset": "ravdess", "speaker_id": speaker_id})
    return pd.DataFrame(rows)


def parse_savee(root: Path, keep: Set[str], emap: Dict[str, str]) -> pd.DataFrame:
    """SAVEE: <SPK>_<code><num>.wav; code ∈ {a,d,f,h,sa,su}."""
    rows = []
    for de in _safe_scandir(root):
        if not de.is_file() or not _is_wav(de.name):
            continue
        base = os.path.splitext(de.name)[0]
        parts = base.split("_")
        if len(parts) < 2:
            continue
        code_raw = parts[1].lower()
        code = code_raw[:2] if code_raw.startswith(("sa", "su")) else code_raw[0]
        emotion = emap.get(code)
        if emotion in keep:
            speaker_id = parts[0]
            rows.append({"filepath": str(Path(de.path)), "emotion": emotion, "dataset": "savee", "speaker_id": speaker_id})
    return pd.DataFrame(rows)


def parse_tess(root: Path, keep: Set[str], emap: Dict[str, str]) -> pd.DataFrame:
    """TESS: folders per speaker; filename like OAF_back_angry.wav (emotion is last token)."""
    rows = []
    for folder in _safe_scandir(root):
        if not folder.is_dir():
            continue
        speaker_id = folder.name.split("_")[0]  # OAF / YAF
        for de in _safe_scandir(Path(folder.path)):
            if not de.is_file() or not _is_wav(de.name):
                continue
            token = os.path.splitext(de.name)[0].split("_")[-1].lower()
            emotion = emap.get(token)
            if emotion in keep:
                rows.append({"filepath": str(Path(de.path)), "emotion": emotion, "dataset": "tess", "speaker_id": speaker_id})
    return pd.DataFrame(rows)


# Register dataset parsers (signature: (root, keep, emap) -> DataFrame)
PARSERS: Dict[str, Callable[[Path, Set[str], Dict[str, str]], pd.DataFrame]] = {
    "crema": parse_crema,
    "ravdess": parse_ravdess,
    "savee": parse_savee,
    "tess": parse_tess,
}


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_metadata(audio_root: Path, metadata_csv: Path, keep: Set[str], datasets: List[str], maps: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    logging.info("Raw data root: %s", audio_root)
    audio_root = Path(audio_root)
    rows = []

    for ds in datasets:
        ds_root = audio_root / ds
        parser = PARSERS[ds]
        emap = _normalize_map(maps.get(ds, {}))
        if not emap:
            logging.warning("No emotion map found for dataset '%s' in config; skipping.", ds)
            continue

        logging.info("Scanning %-7s @ %s", ds, ds_root)
        df = parser(ds_root, keep, emap)
        if df.empty:
            logging.warning("No files kept for %s (missing dir or no matching labels).", ds)
        else:
            logging.info("  -> %5d files kept (%s)", len(df), ", ".join(sorted(df["emotion"].unique())))
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["filepath", "emotion", "dataset", "speaker_id"])
    if combined.empty:
        logging.error("No audio files found across requested datasets.")
        return combined

    # De-duplicate by filepath
    before = len(combined)
    combined.drop_duplicates(subset=["filepath"], inplace=True)
    removed = before - len(combined)
    if removed:
        logging.info("Removed %d duplicate file entries.", removed)

    # Write CSV
    metadata_csv = Path(metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_csv.with_suffix(".tmp.csv")
    combined.to_csv(tmp_path, index=False, quoting=csv.QUOTE_MINIMAL)
    Path(tmp_path).replace(metadata_csv)

    # Summary
    by_ds = combined.groupby("dataset").size().to_dict()
    by_label = combined.groupby("emotion").size().to_dict()
    logging.info("Wrote %d rows -> %s", len(combined), metadata_csv)
    logging.info("Per-dataset counts: %s", by_ds)
    logging.info("Per-label counts  : %s", by_label)
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a unified metadata CSV for SER datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path or name of YAML config.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO, -vv for DEBUG.")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config)

    # Keep only canonical labels present in emotion_labels
    keep: Set[str] = {str(k).lower() for k in cfg["emotion_labels"].keys()}

    # Datasets to scan (in declared order)
    datasets = _resolve_datasets(cfg)

    # Paths (relative paths are resolved against current working directory = repo root)
    audio_root = Path(cfg["project"]["raw_data_dir"])
    metadata_csv = Path(cfg["project"]["metadata_csv"])

    # Dataset-specific code -> label maps
    maps: Dict[str, Dict[str, str]] = {k.lower(): v for k, v in cfg.get("emotion_maps", {}).items()}

    build_metadata(audio_root, metadata_csv, keep, datasets, maps)


if __name__ == "__main__":
    main()
