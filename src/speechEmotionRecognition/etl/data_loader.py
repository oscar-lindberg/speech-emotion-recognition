#!/usr/bin/env python3
"""
Prepare SER train/val/test splits from features manifest.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
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

def _safe_mkdirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _preindex_feature_files(feature_dir: Path) -> set[str]:
    """Index filenames in feature_dir for fast existence checks."""
    return {p.name for p in feature_dir.glob("*.npy")}


def _required_feature_files_exist_fast(base_id: str, req_feats: List[str], have: set[str]) -> bool:
    return all(f"{f}_{base_id}.npy" in have for f in req_feats)


def _label_map_from_cfg(cfg: Dict) -> Dict[str, int]:
    return {str(k).lower(): int(v) for k, v in cfg["emotion_labels"].items()}


def _apply_label_map(df: pd.DataFrame, label_map: Dict[str, int]) -> pd.Series:
    return df["emotion"].astype(str).str.lower().map(label_map)


_AUG_RE = re.compile(r"_aug\d+$")

def _group_key(df: pd.DataFrame, mode: str) -> pd.Series:
    """
    Grouping options:
    - speaker: all rows with same speaker_id
    - speaker+dataset: speaker_id within dataset
    - sample: base row + its augmented copies
    - none: each row is its own group (leakage risk)
    """
    mode = str(mode).lower()
    if mode == "speaker":
        return df["speaker_id"].astype(str)
    if mode == "speaker+dataset":
        return df["speaker_id"].astype(str) + "@" + df["dataset"].astype(str)
    if mode == "sample":
        roots = df["base_id"].astype(str).str.replace(_AUG_RE, "", regex=True)
        return roots
    if mode == "none":
        return df.index.astype(str)
    raise ValueError(f"Unknown group_by mode: {mode}")


def _summarize_split(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        logging.info("%s: 0 rows", name)
        return
    n = len(df)
    aug = int(df.get("augmented", pd.Series([0] * n)).sum())
    lbl_counts = df["emotion"].value_counts().to_dict() if "emotion" in df.columns else {}
    logging.info("%s: %d rows | aug=%d | labels=%s", name, n, aug, lbl_counts)


def _warn_if_missing_classes(name: str, df: pd.DataFrame, label_map: Dict[str, int]) -> None:
    have = set(df["label"].unique().tolist())
    want = set(label_map.values())
    miss = sorted(want - have)
    if miss:
        logging.warning("%s: missing classes %s in split.", name, miss)


# ──────────────────────────────────────────────────────────────────────────────
# Grouped stratified splitting
# ──────────────────────────────────────────────────────────────────────────────

def _group_label_matrix(df: pd.DataFrame, group_col: str, num_classes: int, label_col: str = "label") -> Tuple[List[str], np.ndarray]:
    groups = df[group_col].unique().tolist()
    idx = {g: i for i, g in enumerate(groups)}
    M = np.zeros((len(groups), num_classes), dtype=np.int64)
    for g, y in zip(df[group_col], df[label_col]):
        M[idx[g], int(y)] += 1
    return groups, M


def _greedy_fill(groups: List[str], M: np.ndarray, target_counts: np.ndarray) -> List[str]:
    """Pick groups to approach target_counts (L2 distance)."""
    chosen = []
    cur = np.zeros_like(target_counts)
    remaining = set(range(len(groups)))
    peaky = np.argsort(-np.max(M, axis=1) / (M.sum(axis=1) + 1e-9))  # rare-class heavy first
    for gi in peaky:
        if gi not in remaining:
            continue
        new = cur + M[gi]
        if np.linalg.norm(new - target_counts) <= np.linalg.norm(cur - target_counts):
            chosen.append(groups[gi])
            cur = new
            remaining.remove(gi)
    if (cur < target_counts).any():
        for gi in [i for i in range(len(groups)) if i in remaining]:
            new = cur + M[gi]
            if np.linalg.norm(new - target_counts) < np.linalg.norm(cur - target_counts):
                chosen.append(groups[gi])
                cur = new
    return chosen


def _split_grouped_stratified( df: pd.DataFrame, group_col: str, num_classes: int, val_size: float, test_size: float,
                               seed: int, label_col: str = "label") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Greedy, label-aware split by groups. Falls back to random if needed."""
    rng = np.random.default_rng(seed)
    groups, M = _group_label_matrix(df, group_col, num_classes, label_col)
    rng.shuffle(groups)

    totals = np.array([np.sum(M[:, c]) for c in range(num_classes)], dtype=np.float64)
    totals[totals == 0] = 1.0
    tgt_test = np.round(totals * test_size)
    tgt_val = np.round(totals * val_size)

    test_groups = set(_greedy_fill(groups, M, tgt_test))
    rem_mask = np.array([g not in test_groups for g in groups])
    M_val_candidates = M[rem_mask]
    groups_val_candidates = [g for g in groups if g not in test_groups]
    val_groups = set(_greedy_fill(groups_val_candidates, M_val_candidates, tgt_val))

    is_test = df[group_col].isin(test_groups)
    is_val = (~is_test) & df[group_col].isin(val_groups)
    train = df[~(is_test | is_val)].copy()
    val = df[is_val].copy()
    test = df[is_test].copy()

    if train.empty or val.empty or test.empty:
        logging.warning("Stratified grouped split failed; falling back to random grouped split.")
        all_groups = df[group_col].drop_duplicates().tolist()
        rng.shuffle(all_groups)
        n = len(all_groups)
        n_test = int(round(test_size * n))
        n_val = int(round(val_size * n))
        test_g = set(all_groups[:n_test])
        val_g = set(all_groups[n_test:n_test + n_val])
        is_test = df[group_col].isin(test_g)
        is_val = (~is_test) & df[group_col].isin(val_g)
        train = df[~(is_test | is_val)].copy()
        val = df[is_val].copy()
        test = df[is_test].copy()

    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def _expand_with_aug(df_all: pd.DataFrame, base_groups: pd.Series, group_col: str, keep_aug: bool) -> pd.DataFrame:
    """Expand chosen base groups to full df and optionally include augmented rows."""
    gset = set(base_groups.unique().tolist())
    out = df_all[df_all[group_col].isin(gset)].copy()
    if not keep_aug:
        out = out[out["augmented"] == 0].copy()
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Prepare SER train/val/test splits from feature manifest.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO, -vv for DEBUG.")
    args = p.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config)

    # Paths
    features_meta = Path(cfg["project"]["features_metadata_csv"])
    feature_dir = Path(cfg["project"]["features_dir"])
    split_dir = Path(cfg["project"]["splits_dir"])
    train_csv = Path(cfg["project"]["train_csv"])
    val_csv = Path(cfg["project"]["val_csv"])
    test_csv = Path(cfg["project"]["test_csv"])
    _safe_mkdirs(split_dir)

    # Required features, labels, grouping
    req_feats: List[str] = list(cfg["model"]["features"])
    label_map = _label_map_from_cfg(cfg)
    group_mode = str(cfg["split"].get("group_by", "speaker")).lower()
    group_col_name = "group_key"

    # Split config
    cross = bool(cfg["split"].get("cross_dataset", False))
    included = [d.lower() for d in (cfg["split"].get("included_datasets") or [])]
    train_ds = [d.lower() for d in (cfg["split"].get("train_datasets") or [])]
    test_ds = [d.lower() for d in (cfg["split"].get("test_datasets") or [])]
    val_size = float(cfg["split"]["val_size"])
    test_size = float(cfg["split"]["test_size"])
    seed = int(cfg.get("seed", 0))
    strict = bool(cfg["split"].get("strict_required_features", True))
    use_aug_train = bool(cfg["split"].get("use_augmented_train",
                         cfg["split"].get("use_augmented_for_train", True)))
    use_aug_val = bool(cfg["split"].get("use_augmented_val",
                       cfg["split"].get("use_augmented_for_val", False)))

    # Load manifest
    if not features_meta.is_file():
        raise FileNotFoundError(f"features_metadata.csv not found: {features_meta}")
    df = pd.read_csv(features_meta)

    # Standardize
    needed_cols = {"base_id", "emotion", "dataset", "speaker_id", "augmented"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"{features_meta} missing columns: {sorted(missing)}")
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["emotion"] = df["emotion"].astype(str).str.lower()
    df = df[df["emotion"].isin(label_map.keys())].copy()
    df["label"] = _apply_label_map(df, label_map)
    df[group_col_name] = _group_key(df, group_mode)

    # Enforce required features availability (fast index)
    if strict:
        have = _preindex_feature_files(feature_dir)
        mask = df["base_id"].apply(lambda bid: _required_feature_files_exist_fast(bid, req_feats, have))
        before, after = len(df), int(mask.sum())
        if after < before:
            logging.info("Dropping %d rows without all required features.", before - after)
        df = df[mask].reset_index(drop=True)
    else:
        logging.info("strict_required_features=False (keeping rows even if some features are missing).")

    # Dataset filtering
    if cross:
        df_train_pool = df[df["dataset"].isin(train_ds)].copy()
        df_test = df[df["dataset"].isin(test_ds)].copy()
    else:
        df_train_pool = df[df["dataset"].isin(included)].copy()
        df_test = pd.DataFrame(columns=df.columns)

    # Never augment test
    if not df_test.empty:
        df_test = df_test[df_test["augmented"] == 0].copy()

    num_classes = len(label_map)

    if not cross:
        # Choose groups using BASE rows only to avoid augmentation bias
        base_pool = df_train_pool[df_train_pool["augmented"] == 0].copy()
        train_b, val_b, test_b = _split_grouped_stratified(
            base_pool, group_col=group_col_name, num_classes=num_classes,
            val_size=val_size, test_size=test_size, seed=seed, label_col="label"
        )
        # Expand selected groups to full pool; include augs per flags
        train_df = _expand_with_aug(df_train_pool, train_b[group_col_name], group_col_name, keep_aug=use_aug_train)
        val_df   = _expand_with_aug(df_train_pool,   val_b[group_col_name], group_col_name, keep_aug=use_aug_val)
        test_df  = _expand_with_aug(df_train_pool, test_b[group_col_name], group_col_name, keep_aug=False)
    else:
        # Cross-dataset: test fixed; split val inside train pool using BASE rows only
        base_pool = df_train_pool[df_train_pool["augmented"] == 0].copy()
        _, val_b, _ = _split_grouped_stratified(
            base_pool, group_col=group_col_name, num_classes=num_classes,
            val_size=val_size, test_size=0.0, seed=seed, label_col="label"
        )
        val_df   = _expand_with_aug(df_train_pool, val_b[group_col_name], group_col_name, keep_aug=use_aug_val)
        val_groups = set(val_b[group_col_name].unique().tolist())
        train_df = df_train_pool[~df_train_pool[group_col_name].isin(val_groups)].copy()
        if not use_aug_train:
            train_df = train_df[train_df["augmented"] == 0].copy()
        test_df  = df_test.copy()  # already base-only

    # Cleanup
    for d in (train_df, val_df, test_df):
        d.drop(columns=[group_col_name], inplace=True, errors="ignore")
        d["label"] = d["label"].astype(int)

    # Summaries and sanity
    _summarize_split("TRAIN", train_df)
    _summarize_split("VAL",   val_df)
    _summarize_split("TEST",  test_df)
    _warn_if_missing_classes("TRAIN", train_df, label_map)
    _warn_if_missing_classes("VAL",   val_df,   label_map)
    _warn_if_missing_classes("TEST",  test_df,  label_map)

    # Save splits
    _safe_mkdirs(train_csv.parent)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    logging.info("Wrote splits → %s, %s, %s", train_csv, val_csv, test_csv)

    # Snapshot meta
    meta = {
        "required_features": req_feats,
        "group_by": group_mode,
        "cross_dataset": cross,
        "included_datasets": included,
        "train_datasets": train_ds,
        "test_datasets": test_ds,
        "val_size": val_size,
        "test_size": test_size,
        "seed": seed,
        "counts": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
    }
    with open(split_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote split meta → %s", split_dir / "meta.json")


if __name__ == "__main__":
    main()
