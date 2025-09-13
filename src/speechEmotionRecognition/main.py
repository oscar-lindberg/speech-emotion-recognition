#!/usr/bin/env python3
"""
SER pipeline CLI.

Usage:
ser --config <yaml|name> [--steps step,step,...] [--epochs N] [--run-name STR] [--dry-run] [-v|-vv]

Steps:
doctor | extract | transform | load | train
(default: extract,transform,load,train)

Examples
# Full pipeline: ser --config spectral_crnn.yaml --steps extract,transform,load,train
# Load + train only: ser --config spectral_crnn.yaml --steps load,train
# Quick smoke test: ser --config spectral_crnn.yaml --steps train --epochs 5 --run-name quicktest
# Environment check: ser --config spectral_crnn.yaml --steps doctor

Note: Run from the repo root (relative paths resolve against it).
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from speechEmotionRecognition.config_loader import load_config


# ──────────────────────────────────────────────────────────────────────────────
# Constants and step registry
# ──────────────────────────────────────────────────────────────────────────────
VALID_STEPS: Tuple[str, ...] = ("doctor", "extract", "transform", "load", "train")
DEFAULT_STEPS = "extract,transform,load,train"

SCRIPTS: Dict[str, str] = {
    "extract": "speechEmotionRecognition.etl.data_extractor",
    "transform": "speechEmotionRecognition.etl.feature_generator",
    "load": "speechEmotionRecognition.etl.data_loader",
    "train": "speechEmotionRecognition.train.train",
}


# ──────────────────────────────────────────────────────────────────────────────
# Printing utilities
# ──────────────────────────────────────────────────────────────────────────────

def _rule(title: str = "") -> None:
    line = "─" * 78
    print(f"{line}\n{title}\n{line}" if title else line)


def _kv(label: str, value) -> None:
    print(f"  {label:<20}: {value}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _err(msg: str) -> None:
    print(f"[ERR] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────
def pkg_root() -> Path:
    """src/speechEmotionRecognition"""
    return Path(__file__).resolve().parent

def repo_root() -> Path:
    """Project root (contains pyproject.toml, configs/, data/, ...)."""
    start = pkg_root()
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            return d
    # Fallback for the standard src layout: src/speechEmotionRecognition -> repo/
    return start.parents[1]


# ──────────────────────────────────────────────────────────────────────────────
# Step parsing and config summary
# ──────────────────────────────────────────────────────────────────────────────

def parse_steps(steps_csv: str) -> List[str]:
    """Parse and validate --steps."""
    steps = [s.strip().lower() for s in steps_csv.split(",") if s.strip()]
    unknown = [s for s in steps if s not in VALID_STEPS]
    if unknown:
        raise ValueError(f"Unknown step(s): {unknown}. Valid: {VALID_STEPS}")
    return steps


def print_cfg_summary(cfg: dict) -> None:
    """Print a compact summary of key config values."""
    _rule("CONFIG SUMMARY")
    model = cfg.get("model", {})
    features = cfg.get("features", {})
    project = cfg.get("project", {})
    _kv("model.name", model.get("name", "-"))
    _kv("model.features", model.get("features", "-"))
    _kv("features.types", features.get("types", "-"))
    _kv("device", cfg.get("device", "auto"))
    _kv("raw_data_dir", project.get("raw_data_dir", "-"))
    _rule()


# ──────────────────────────────────────────────────────────────────────────────
# Step implementations
# ──────────────────────────────────────────────────────────────────────────────
def run_subprocess(step: str, cfg_path: Optional[str], extra_args: Optional[List[str]] = None) -> None:
    """
    Execute a pipeline script as a subprocess (module form), forwarding --config + extras.
    Stops the pipeline on first failure.
    """
    if step not in SCRIPTS:
        raise ValueError(f"Step '{step}' is not directly runnable.")

    mod = SCRIPTS[step]
    cmd = [sys.executable, "-m", mod]
    if cfg_path:
        cmd += ["--config", cfg_path]
    if extra_args:
        cmd += extra_args

    _rule(f"RUN {step.upper()}")
    _kv("cmd", " ".join(cmd))
    try:
        # Run from repo root so relative paths (configs/, data/, runs/) are stable
        subprocess.run(cmd, cwd=str(repo_root()), check=True)
        _ok(f"{step} ✓")
    except subprocess.CalledProcessError as e:
        _err(f"{step} failed with code {e.returncode}")
        raise

def step_doctor(cfg: dict) -> None:
    """
    Print environment and config sanity checks.
    """
    _rule("DOCTOR")
    try:
        import torch  # noqa: F401
        import torch.backends  # noqa: F401
        _kv("torch.version", torch.__version__)
        _kv("cuda_available", getattr(torch.cuda, "is_available", lambda: False)())
        _kv("mps_available", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        _kv("cfg.device", cfg.get("device", "auto"))
    except Exception as e:
        _warn(f"torch check failed: {e}")

    feats = set(cfg["features"]["types"])
    model_feats = set(cfg["model"]["features"])
    if feats.isdisjoint(model_feats):
        _warn(f"model.features {model_feats} do not overlap with features.types {feats}")

    if ("ssl" in feats) or ("ssl" in model_feats):
        try:
            import transformers  # noqa: F401
            _kv("transformers", "available")
        except Exception as e:
            _warn(f"transformers not available: {e}")
        ssl = cfg.get("ssl", {})
        _kv("ssl.backend", ssl.get("backend", "-"))
        _kv("ssl.model_id", ssl.get("model_id", "-"))

    data = cfg["project"]
    raw_data_dir = Path(data["raw_data_dir"])
    _kv("raw_data_dir.exists", raw_data_dir.exists())
    _rule()


# ──────────────────────────────────────────────────────────────────────────────
# CLI and orchestration
# ──────────────────────────────────────────────────────────────────────────────

def build_plan(steps: List[str], epochs: Optional[int], run_name: Optional[str], verbose: int = 0) -> List[Tuple[str, List[str]]]:
    """Translate steps into a run plan with per-step extra CLI args."""
    # Scripts that accept -v / -vv
    VERBOSE_FRIENDLY = {"extract", "transform", "load"}

    extra_common = ["-v"] * max(0, int(verbose))
    extra_train: List[str] = []
    if epochs is not None:
        extra_train += ["--epochs", str(epochs)]
    if run_name:
        extra_train += ["--run-name", run_name]

    plan: List[Tuple[str, List[str]]] = []
    for s in steps:
        if s == "doctor":
            continue
        extras: List[str] = []
        if s in VERBOSE_FRIENDLY and verbose:
            extras += extra_common
        if s == "train":
            extras += extra_train
        plan.append((s, extras))
    return plan


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Readable orchestrator for the SER pipeline (extract/transform/load/train).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument(
        "--steps",
        type=str,
        default=DEFAULT_STEPS,
        help=f"Comma-separated steps from {VALID_STEPS}. Include 'doctor' for env checks.",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override epochs for the train step.")
    p.add_argument("--run-name", type=str, default=None, help="Optional suffix to tag the training run.")
    p.add_argument("--dry-run", action="store_true", help="Show the plan without executing.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO, -vv for DEBUG")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    print_cfg_summary(cfg)

    steps = parse_steps(args.steps)

    # Optional environment checks first
    if "doctor" in steps:
        step_doctor(cfg)

    # Build and (optionally) preview the plan
    plan = build_plan(steps, epochs=args.epochs, run_name=args.run_name, verbose=args.verbose)
    if args.dry_run:
        _rule("DRY RUN — PLAN")
        for s, extra in plan:
            _kv(s, f"script={SCRIPTS[s]} extras={extra or '[]'}")
        _rule()
        return

    # Execute steps
    for s, extra in plan:
        run_subprocess(s, args.config, extra_args=extra)

    _ok("Pipeline finished.")


if __name__ == "__main__":
    main()
