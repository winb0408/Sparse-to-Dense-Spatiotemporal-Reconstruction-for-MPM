"""Run directory layout and logging.

Every run writes, under ``runs/<experiment>/<arm>__<tag>/``::

    config.yaml     the fully-resolved config (no defaults left implicit)
    provenance.json git commit, hardware, hashes of config/split/manifest
    log.csv         per-step training log
    metrics.jsonl   one JSON object per validation event
    samples.csv     per-sample evaluation rows (evaluate.py)
    best.pt/last.pt weights

An empty or missing file is a bug, not "no data": :meth:`RunDir.validate`
checks the expected files exist at the end of a run.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

__all__ = ["RunDir"]


class RunDir:
    def __init__(self, root: str | Path, name: str, resume: bool = False):
        self.path = Path(root) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.started = time.time()
        self._log_writer = None
        self._log_fh = None
        self._log_header: list[str] | None = None
        if not resume:
            (self.path / "metrics.jsonl").write_text("")
            # samples.csv is created on first write: an empty file at train time
            # would mean "evaluation produced no rows", which is indistinguishable
            # from a bug when you inspect the run later.

    # -- config & provenance ------------------------------------------------
    def write_config(self, cfg: dict) -> None:
        import yaml
        (self.path / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    def write_provenance(self, payload: dict) -> None:
        (self.path / "provenance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str))

    # -- step log -----------------------------------------------------------
    def log_step(self, row: dict) -> None:
        if self._log_writer is None:
            self._log_header = list(row)
            self._log_fh = open(self.path / "log.csv", "a", newline="")
            self._log_writer = csv.DictWriter(self._log_fh, fieldnames=self._log_header)
            if (self.path / "log.csv").stat().st_size == 0:
                self._log_writer.writeheader()
        self._log_writer.writerow({k: _fmt(row.get(k, "")) for k in self._log_header})
        self._log_fh.flush()

    # -- metrics ------------------------------------------------------------
    def log_metrics(self, payload: dict) -> None:
        with open(self.path / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

    def append_sample_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        path = self.path / "samples.csv"
        write_header = path.stat().st_size == 0
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow({k: _fmt(r.get(k, "")) for k in rows[0]})

    # -- bookkeeping --------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def validate(self, expect_weights: bool = True) -> list[str]:
        """Return a list of problems with the finished run directory."""
        problems = []
        for name in ("config.yaml", "provenance.json", "log.csv", "metrics.jsonl"):
            p = self.path / name
            if not p.exists() or p.stat().st_size == 0:
                problems.append(f"missing or empty: {name}")
        if expect_weights and not (self.path / "best.pt").exists():
            problems.append("missing: best.pt")
        return problems

    def close(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
            self._log_writer = None


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return v
