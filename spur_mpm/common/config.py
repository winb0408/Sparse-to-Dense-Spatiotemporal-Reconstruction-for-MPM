"""YAML configs: load, deep-merge, override and hash.

A run's config must be sufficient to reproduce it without reading the code, so
this module keeps configs as plain nested dicts (easy to serialise, easy to
hash) and refuses silent key creation: overriding a typo'd key raises instead
of inventing a new branch of the tree that nothing reads.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

__all__ = ["load_config", "deep_merge", "apply_overrides", "save_config",
           "config_hash", "split_csv"]


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, apply ``key=value`` overrides (dotted keys allowed)."""
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge ``patch`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f"override path {key!r}: {part!r} does not exist "
                               "in the config (silent key creation is disabled)")
            if not isinstance(node[part], dict):
                raise KeyError(f"override path {key!r}: {part!r} is not a mapping")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"override key {parts[-1]!r} does not exist under "
                           f"{'.'.join(parts[:-1]) or '<root>'}")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def config_hash(cfg: dict) -> str:
    """Stable hash of a config, used to detect unintentional protocol drift."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def split_csv(value: str | None) -> list[str]:
    """``"a,b, c"`` -> ``["a", "b", "c"]``; empty string -> []."""
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]
