"""Reproducibility metadata recorded with every run.

Plan §10 requires each formal experiment to carry its code revision, config,
dataset/split hashes, seed, checkpoint identity and hardware. Everything here is
best-effort: a missing git binary or an uncommitted tree is recorded as such
rather than silently omitted, because "the numbers cannot be traced to a commit"
is exactly the failure this project is trying to avoid.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

__all__ = ["git_state", "file_hash", "hardware_summary", "environment_summary",
           "collect_provenance"]


def _run(cmd: list[str], cwd: str | Path | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def git_state(repo_root: str | Path | None = None) -> dict:
    cwd = str(repo_root or Path.cwd())
    commit = _run(["git", "rev-parse", "HEAD"], cwd)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    dirty = _run(["git", "status", "--porcelain"], cwd)
    return {"commit": commit or "unknown", "branch": branch or "unknown",
            "dirty": bool(dirty), "n_dirty_files": len(dirty.splitlines())}


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hardware_summary() -> dict:
    import torch
    out = {"cpu_count": os.cpu_count(), "platform": platform.platform()}
    if torch.cuda.is_available():
        out["gpus"] = [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
            for i in range(torch.cuda.device_count())
        ]
    else:
        out["gpus"] = []
    return out


def environment_summary() -> dict:
    import numpy
    import torch
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": numpy.__version__, "cuda": torch.version.cuda}
    try:
        import skimage
        out["scikit_image"] = skimage.__version__
    except Exception:
        out["scikit_image"] = "missing"
    return out


def collect_provenance(repo_root: str | Path | None = None,
                       config: dict | None = None,
                       artifacts: dict[str, str] | None = None) -> dict:
    """Full provenance record; ``artifacts`` maps label -> file path to hash."""
    from spur_mpm.common.config import config_hash

    hashes = {label: file_hash(p) for label, p in (artifacts or {}).items()}
    return {
        "git": git_state(repo_root),
        "hardware": hardware_summary(),
        "environment": environment_summary(),
        "config_hash": config_hash(config) if config else None,
        "artifact_hashes": hashes,
    }


def dumps_sorted(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
