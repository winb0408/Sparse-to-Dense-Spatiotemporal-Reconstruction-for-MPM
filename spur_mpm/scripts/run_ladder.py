"""Run the supervised ladder on the fixed cohort: arms x sparsity levels.

One command, one GPU queue, one progress file. The point is that every cell of
the results table is produced by *the same* invocation shape -- the only things
that vary between two runs are ``model.arm`` and the sampling spec -- so a
difference in the table cannot come from a difference in how the runs were
launched.

    # what would run, without running it
    python -m spur_mpm.scripts.run_ladder --stage ablation --dry-run

    # the innovation ablation at the headline sparsity, on three free GPUs
    python -m spur_mpm.scripts.run_ladder --stage ablation --devices 0,1,2

Stages (each is a deliberate question, not a slice of a grid):

``ablation``   A0 / A1 / S0 / S1 / S2 / S1g at ``y_k2``. The three innovations
               are S0->S1 (physics warm start), S1->S2 (uncertainty) and
               S1->S1g (gated fusion); A0/A1 are the no-learning floors those
               deltas are measured above.
``sweep``      the best supervised arm down the Y ladder (k2 -> k8) and across
               the Z and joint families. This is the "how sparse can it get"
               curve, and it is what a acquisition-protocol recommendation
               rests on.
``matched``    the same 25% budget reached three different ways (``y_k4`` /
               ``z_k4`` / ``yz_k2x2``), so the families are compared at equal
               cost rather than at equal ``k``.
``all``        the three stages in the order above.

Cost and honesty notes:

* A0 and A1 need no training, so they are evaluated directly. Their rows are
  cheap; they are included because a delta with no floor under it is not a
  result.
* ``--jobs`` is bounded by ``--devices``: a card carrying two runs at once would
  make the wall-clock numbers meaningless, and the GPU we do not own is never
  used (see ``scripts/benchmark.py`` for the probe).
* Progress is rewritten after every finished cell and the script exits non-zero
  if any cell failed, so a resumed run cannot silently lose a row.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = ["STAGES", "plan", "main"]

#: The cohort this ladder is defined on. Fixed by design: one split, one test
#: animal (E020). A second animal would be a different experiment, not a rerun.
COHORT_INDEX = "data_local/dryad_cache/index_cohort.json"
COHORT_SPLIT = "spur_mpm/experiments/E020_cohort/split.json"

FREE_ARMS = ("A0", "A1")
TRAINED_ARMS = ("S0", "S1", "S2", "S1g")

STAGES: dict[str, list[tuple[str, str]]] = {
    "ablation": [(arm, "y_k2") for arm in FREE_ARMS + TRAINED_ARMS],
    "sweep": [("S1", level) for level in
              ("y_k3", "y_k4", "y_k8", "z_k2", "z_k4", "yz_k2x2", "yz_k2x4")],
    "matched": [(arm, level) for level in ("y_k4", "z_k4", "yz_k2x2")
                for arm in ("A0", "A1", "S1", "S2")],
}


def plan(stages: list[str]) -> list[tuple[str, str]]:
    """Expand stages into ``(arm, level)`` pairs, first occurrence wins."""
    seen: dict[tuple[str, str], None] = {}
    for stage in stages:
        for pair in STAGES[stage]:
            seen.setdefault(pair, None)
    return list(seen)


def _spec(level: str) -> dict:
    from spur_mpm.data.sparse import LEVELS_3D
    if level not in LEVELS_3D:
        raise SystemExit(f"unknown level {level!r}; have {sorted(LEVELS_3D)}")
    return LEVELS_3D[level].to_dict()


def _overrides(level: str) -> list[str]:
    spec = _spec(level)
    return [
        f"data.index={COHORT_INDEX}",
        f"data.split={COHORT_SPLIT}",
        "data.fold=0",
        f"sampling.mode={spec['mode']}",
        f"sampling.ky={spec['ky']}",
        f"sampling.kz={spec['kz']}",
        f"sampling.pattern={spec.get('pattern', 'grid')}",
        f"sampling.seed={spec.get('seed', 0)}",
    ]


def _free_gpus() -> list[str]:
    """Indices of GPUs with no compute process, least-used memory first."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    used: dict[str, int] = {}
    for line in out.strip().splitlines():
        idx, mem = (part.strip() for part in line.split(","))
        used[idx] = int(mem)
    busy = set()
    try:
        procs = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                "--format=csv,noheader"],
                               capture_output=True, text=True, check=True).stdout
        busy = {u.strip() for u in procs.split() if u.strip()}
        if busy:
            uuid_out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid",
                 "--format=csv,noheader"], capture_output=True, text=True,
                check=True).stdout
            by_uuid = {u.strip(): i.strip() for i, u in
                       (line.split(",") for line in uuid_out.strip().splitlines())}
            busy = {by_uuid.get(u, u) for u in busy}
    except (OSError, subprocess.CalledProcessError):
        pass
    free = [i for i in used if i not in busy]
    return sorted(free, key=lambda i: used[i])


class State:
    """Append-only progress file, rewritten atomically after every cell."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.rows: list[dict] = []
        if path.exists():
            self.rows = json.loads(path.read_text())["cells"]

    def add(self, row: dict) -> None:
        with self._lock:
            self.rows = [r for r in self.rows
                         if (r["arm"], r["level"]) != (row["arm"], row["level"])]
            self.rows.append(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"cells": self.rows}, indent=2))
            tmp.replace(self.path)

    def done(self, arm: str, level: str) -> dict | None:
        for r in self.rows:
            if (r["arm"], r["level"]) == (arm, level) and r["status"] == "ok":
                return r
        return None


def _run(cmd: list[str], env: dict, log: Path) -> tuple[int, float]:
    started = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return proc.returncode, time.time() - started


def _cell(arm: str, level: str, *, out_root: Path, steps: int | None,
          device: str, state: State, dry: bool,
          eval_only: bool = False) -> dict:
    name = f"E021_{arm}_{level}"
    run_dir = out_root / name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device,
               PYTHONUNBUFFERED="1")
    started = time.time()
    row = {"arm": arm, "level": level, "name": name, "device": device,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    ov = _overrides(level)
    if dry:
        print(f"[{device}] {name}: "
              + " ".join(f"--set {o}" for o in ov))
        return {**row, "status": "dry"}

    rc = 0
    if eval_only and arm in TRAINED_ARMS and not (run_dir / "best.pt").exists():
        return {**row, "status": "no_checkpoint"}
    if arm in TRAINED_ARMS and not eval_only:
        cmd = [sys.executable, "-m", "spur_mpm.train",
               "--arm", arm, "--name", name, "--out-root", str(out_root),
               "--set", "experiment=" + name]
        for o in ov:
            cmd += ["--set", o]
        if steps is not None:
            cmd += ["--set", f"train.steps={steps}"]
        rc, secs = _run(cmd, env, out_root / "_logs" / f"{name}.train.log")
        row["train_seconds"] = round(secs, 1)
        if rc != 0:
            return {**row, "status": "train_failed", "rc": rc}

    ckpt = run_dir / "best.pt"
    ck = ["--ckpt", str(ckpt)] if ckpt.exists() else []
    cmd = [sys.executable, "-m", "spur_mpm.evaluate",
           "--arm", arm, "--split", "test", "--out", str(run_dir / "eval_test")]
    cmd += ck
    for o in ov:
        cmd += ["--set", o]
    rc, secs = _run(cmd, env, out_root / "_logs" / f"{name}.eval.log")
    row["eval_seconds"] = round(secs, 1)
    summary = run_dir / "eval_test" / "summary.json"
    if rc != 0:
        return {**row, "status": "eval_failed", "rc": rc}
    row["status"] = "ok"
    row["wall_seconds"] = round(time.time() - started, 1)
    if summary.exists():
        m = json.loads(summary.read_text())["metrics"]
        # Keep the scalar means only: the per-sample rows stay in samples.csv.
        row["metrics"] = {k: v.get("mean_of_group_means", v.get("mean"))
                          for k, v in m.items() if isinstance(v, dict)}
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", action="append", default=None,
                   choices=sorted(STAGES), help="repeatable; default: ablation")
    p.add_argument("--out-root", type=Path, default=Path("runs/E021_ladder"))
    p.add_argument("--devices", default=None,
                   help="comma separated GPU indices; default: probe for free ones")
    p.add_argument("--jobs", type=int, default=None, help="parallel runs")
    p.add_argument("--steps", type=int, default=None,
                   help="override train.steps (smoke runs)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="redo finished cells")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training and re-score the existing best.pt files "
                        "(use after a metric or segmentation change: the "
                        "checkpoints do not depend on either)")
    args = p.parse_args(argv)

    stages = args.stage or ["ablation"]
    pairs = plan(stages)
    devices = ([d.strip() for d in args.devices.split(",")] if args.devices
               else _free_gpus())
    if not devices:
        devices = ["0"]
        print("warning: no free GPU found by probe; defaulting to device 0",
              file=sys.stderr)
    jobs = args.jobs or len(devices)

    state = State(args.out_root / "ladder.json")
    todo = [pr for pr in pairs if args.force or state.done(*pr) is None]
    print(f"stages={stages} cells={len(pairs)} todo={len(todo)} "
          f"devices={devices} jobs={jobs}")
    if not todo:
        print("nothing to do")
        return 0

    failures: list[str] = []
    started = time.time()

    def work(item):
        i, (arm, level) = item
        device = devices[i % len(devices)]
        row = _cell(arm, level, out_root=args.out_root, steps=args.steps,
                    device=device, state=state, dry=args.dry_run,
                    eval_only=args.eval_only)
        if not args.dry_run:
            state.add(row)
        tag = row["status"]
        msg = f"[{device}] {arm:4s} {level:8s} {tag}"
        if "metrics" in row:
            msg += (f"  psnr={row['metrics'].get('psnr', float('nan')):.3f}"
                    f"  dice={row['metrics'].get('dice', float('nan')):.4f}"
                    f"  cl_dice={row['metrics'].get('cl_dice', float('nan')):.4f}")
        print(msg, flush=True)
        if tag not in ("ok", "dry"):
            failures.append(f"{arm}/{level}")
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(work, enumerate(todo)))

    elapsed = time.time() - started
    print(f"\nfinished {len(todo)} cells in {elapsed / 3600:.2f} h; "
          f"state -> {args.out_root / 'ladder.json'}")
    if failures:
        print("FAILED: " + ", ".join(failures) + f"  (logs in {args.out_root}/_logs)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
