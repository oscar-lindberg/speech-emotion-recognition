#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any
import yaml


class ConfigError(RuntimeError):
    """Config is missing or malformed."""


def _find_yaml(path_str: str) -> Path:
    p = Path(os.path.expanduser(os.path.expandvars(path_str)))

    # 1. Absolute or relative to CWD
    cand = p if p.is_absolute() else Path.cwd() / p
    if cand.is_file():
        return cand

    # 2. CWD/configs/<name>
    if not p.parent or str(p.parent) == ".":
        cand = Path.cwd() / "configs" / p.name
        if cand.is_file():
            return cand

    # 3. Repo root/configs/<name> (repo root = two levels up from this file)
    repo_root = Path(__file__).resolve().parents[2]
    cand = repo_root / "configs" / p.name
    if cand.is_file():
        return cand

    raise FileNotFoundError(f"Config YAML not found: {path_str}")


def load_config(yaml_path: str) -> Dict[str, Any]:
    path = _find_yaml(yaml_path)
    with open(path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("YAML root must be a mapping.")
    return cfg
