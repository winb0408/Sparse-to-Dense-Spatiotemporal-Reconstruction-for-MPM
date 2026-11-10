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
import re
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
TRAINED_ARMS = ("S0", "S1", "S2", "S1g", "S2g")

STAGES: dict[str, list[tuple[str, str]]] = {
    "ablation": [(arm, "y_k2") for arm in FREE_ARMS + TRAINED_ARMS],
    "sweep": [("S1", level) for level in
              ("y_k3", "y_k4", "y_k8", "z_k2", "z_k3", "z_k4", "yz_k2x2",
               "yz_k2x4")],
    "matched": [(arm, level) for level in ("y_k4", "z_k4", "yz_k2x2")
                for arm in ("A0", "A1", "S0", "S1", "S2", "S1g", "S2g")],
    # A floor under *every* rung, not only under y_k2: a budget curve where
    # most levels have no parameter-free number cannot say whether a drop came
    # from the method or from the budget. A0 and A1 coincide under the identity
    # PSF (E001), so only A0 is run and the degeneracy is reported instead of
    # paying for it nine more times.
    # Plan §3.3 control: the same three 25 % budgets at one fixed context, so
    # the "axis does not matter" finding cannot be explained by the Z context
    # each protocol happens to need. Run it with --context-slices 7.
    "control": [("S1", level) for level in ("y_k4", "z_k4", "yz_k2x2")],
    "floors": [("A0", level) for level in
               ("y_k2", "y_k3", "y_k4", "y_k8", "z_k2", "z_k3", "z_k4",
                "yz_k2x2", "yz_k2x4")],
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


def level_context(level: str) -> int:
    """Z context a level needs, or 1 when it skips no planes.

    Plan §3.3 forbids letting a ``kz`` setting quietly see more planes than its
    neighbour, so this is the *minimum* a protocol needs (``2kz-1``), not a
    tuned value, and the cell name does not hide it: the number goes into
    ``data.context_slices`` and the run's ``config.yaml``.
    """
    from spur_mpm.data.datasets import required_context_depth
    return required_context_depth(_spec(level)["kz"])


def _overrides(level: str, context: int | None = None) -> list[str]:
    spec = _spec(level)
    ov = [
        f"data.index={COHORT_INDEX}",
        f"data.split={COHORT_SPLIT}",
        "data.fold=0",
        f"sampling.mode={spec['mode']}",
        f"sampling.ky={spec['ky']}",
        f"sampling.kz={spec['kz']}",
        f"sampling.pattern={spec.get('pattern', 'grid')}",
        f"sampling.seed={spec.get('seed', 0)}",
    ]
    ctx = level_context(level) if context is None else int(context)
    if ctx > 1:
        ov.append(f"data.context_slices={ctx}")
    return ov


def cell_name(arm: str, level: str, context: int | None = None) -> str:
    """``E021_<arm>_<level>``, plus ``_c<N>`` when the context is forced.

    A forced context is a *different experiment* -- it fixes the usable depth so
    that two protocols can be compared at equal context instead of at equal
    ``kz`` -- so it must not overwrite the cell that trained at the level's own
    minimum. The suffix keeps both cells, and the report can pair them.
    """
    name = f"E021_{arm}_{level}"
    if context is None or int(context) == level_context(level):
        return name          # the level already runs at that context
    return f"{name}_c{int(context)}"


def _free_gpus(run=subprocess.run) -> list[str]:
    """Indices of GPUs with no compute process, least-used memory first.

    "Free" means *no compute process of anyone's*, not "enough memory left".
    Free memory on a card another user is training on is not ours to take: the
    job that owns it may ask for more later and be killed by our allocation, and
    any timing we measure beside it is meaningless (see ``scripts/benchmark.py``).

    ``run`` is injected so the parsing can be tested without ``nvidia-smi``.
    Two failure modes return ``[]`` on purpose -- "I cannot tell" must read as
    "nothing is free", because the caller turns ``[]`` into a refusal:

    * ``nvidia-smi`` missing or failing;
    * a compute process whose GPU uuid is not in the index/uuid map. The old
      code kept the uuid as if it were an index, so an unmappable process
      excluded nothing and the card was handed out as free.
    """
    try:
        out = run(["nvidia-smi", "--query-gpu=index,memory.used",
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
        procs = run(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, check=True).stdout
        busy = [u.strip() for u in procs.split() if u.strip()]
        if busy:
            uuid_out = run(["nvidia-smi", "--query-gpu=index,uuid",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout
            by_uuid = {u.strip(): i.strip() for i, u in
                       (line.split(",") for line in uuid_out.strip().splitlines())}
            if any(u not in by_uuid for u in busy):
                return []           # a running job we cannot place: assume all busy
            busy = {by_uuid[u] for u in busy}
    except (OSError, subprocess.CalledProcessError):
        return []                   # cannot enumerate processes: assume all busy
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

    @staticmethod
    def _key(row: dict) -> tuple:
        # Keyed on the directory name: a forced-context rerun of the same level
        # is a different cell. Keying on (arm, level) used to make the _c7 row
        # *replace* the minimum-context row, so the ladder forgot that the cell
        # it already had was finished (and a later run would retrain it).
        name = row.get("name")
        return ("name", name) if name else ("arm", row["arm"], row["level"])

    def add(self, row: dict) -> None:
        with self._lock:
            key = self._key(row)
            self.rows = [r for r in self.rows if self._key(r) != key]
            self.rows.append(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"cells": self.rows}, indent=2))
            tmp.replace(self.path)

    def done(self, arm: str, level: str, name: str | None = None) -> dict | None:
        """Finished row for this exact cell.

        Keyed on the directory name rather than on ``(arm, level)`` so that a
        forced-context rerun of the same level is a separate cell and is not
        mistaken for its minimum-context sibling.
        """
        want = name or f"E021_{arm}_{level}"
        for r in self.rows:
            if r.get("name", want) == want and r["status"] == "ok":
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
          eval_only: bool = False, context: int | None = None,
          skeleton: bool = False) -> dict:
    name = cell_name(arm, level, context)
    run_dir = out_root / name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device,
               PYTHONUNBUFFERED="1")
    started = time.time()
    row = {"arm": arm, "level": level, "name": name, "device": device,
           "context": level_context(level) if context is None else int(context),
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    ov = _overrides(level, context)
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
    if skeleton:
        cmd.append("--skeleton")
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


def _arm_level_from_name(name: str) -> tuple[str, str] | None:
    """Split "E021_<arm>_<level>[_c<N>]" back into its arm and level."""
    from spur_mpm.models.registry import ARMS

    if not name.startswith("E021_"):
        return None
    rest = name[len("E021_"):]
    for arm in sorted(ARMS, key=len, reverse=True):
        if rest.startswith(arm + "_"):
            # the _c<N> suffix is the context override, not part of the level
            return arm, re.sub(r"_c\d+$", "", rest[len(arm) + 1:])
    return None


def rebuild_state(out_root: Path) -> list[dict]:
    """Reconstruct ladder.json from the run directories.

    The state file is a convenience, not the source of truth: every finished
    cell already carries ``config.yaml`` and ``eval_test/summary.json``. This
    makes the ladder recoverable after a state bug, without retraining, and is
    what ``--rebuild-state`` writes back.
    """
    rows = []
    for run_dir in sorted(out_root.glob("E021_*")):
        summary = run_dir / "eval_test" / "summary.json"
        parsed = _arm_level_from_name(run_dir.name)
        if not summary.exists() or parsed is None:
            continue
        arm, level = parsed
        m = json.loads(summary.read_text()).get("metrics", {})
        rows.append({"arm": arm, "level": level, "name": run_dir.name,
                     "status": "ok",
                     "context": _context_of(run_dir),
                     "metrics": {k: v.get("mean_of_group_means", v.get("mean"))
                                 for k, v in m.items() if isinstance(v, dict)}})
    return rows


def _context_of(run_dir: Path) -> int | None:
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return None
    import yaml

    return yaml.safe_load(cfg.read_text()).get("data", {}).get("context_slices")


def select(pairs, only, context=None):
    """Keep the cells whose *final* name contains one of ``only``.

    Matching on the final name (not on arm or level) is what makes this safe:
    "y_k4" would otherwise also select the "y_k4_c7" cell, which is a different
    experiment -- same rows, seven planes of context instead of one.
    """
    return [pr for pr in pairs
            if any(sub in cell_name(pr[0], pr[1], context) for sub in only)]


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
    p.add_argument("--rebuild-state", action="store_true",
                   help="rebuild ladder.json from the run directories before "
                        "planning (the directories, not the state file, are the "
                        "source of truth)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="redo finished cells")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training and re-score the existing best.pt files "
                        "(use after a metric or segmentation change: the "
                        "checkpoints do not depend on either)")
    p.add_argument("--context-slices", type=int, default=None,
                   help="force one Z context for every cell (plan §3.3 control: "
                        "protocols compared at equal context, not equal kz). "
                        "Cells are suffixed _c<N> so they never overwrite the "
                        "cells trained at their own minimum context.")
    p.add_argument("--only", action="append", default=None, metavar="SUBSTR",
                   help="restrict to cells whose name contains SUBSTR "
                        "(repeatable). Used to re-score a handful of cells "
                        "after a metric change without redoing the ladder. "
                        "Substring, so do not pass --context-slices while "
                        "selecting a cell that ran at its own context: the "
                        "name it would match is the _c<N> one.")
    p.add_argument("--skeleton", action="store_true",
                   help="also score the skeleton per window (slower; adds "
                        "skelF1 and the skeleton counts to summary.json)")
    args = p.parse_args(argv)

    stages = args.stage or ["ablation"]
    pairs = plan(stages)
    if args.only:
        before = len(pairs)
        pairs = select(pairs, args.only, args.context_slices)
        print(f"--only kept {len(pairs)} of {before} cells")
    devices = ([d.strip() for d in args.devices.split(",")] if args.devices
               else _free_gpus())
    if not devices:
        if args.dry_run:
            devices = ["dry-run"]
        else:
            raise SystemExit("No free GPU detected. Refusing to silently use GPU 0 "
                             "on a shared server. Supply --devices N after checking "
                             "the selected device is available.")
    jobs = args.jobs or len(devices)

    state = State(args.out_root / "ladder.json")
    if args.rebuild_state:
        n = len(state.rows)
        for row in rebuild_state(args.out_root):
            state.add(row)
        print(f"--rebuild-state: {n} rows -> {len(state.rows)}")
    todo = [pr for pr in pairs
            if args.force or state.done(pr[0], pr[1],
                                        cell_name(pr[0], pr[1], args.context_slices)) is None]
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
                    eval_only=args.eval_only,
                    context=args.context_slices, skeleton=args.skeleton)
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
