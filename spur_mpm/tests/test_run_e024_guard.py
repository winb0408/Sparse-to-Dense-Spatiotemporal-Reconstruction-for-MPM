"""The rules `run_e024.sh` enforces before it touches a GPU.

A refusal is only worth having if it is tested, and a test that reads the real
`nvidia-smi` is not a test -- it passes or fails depending on what the machine
happens to be doing. So the occupancy is stubbed through `NVIDIA_SMI`, and what
is pinned is the *rule*.

1. **GPU 2 is on probation.** Filling it has hung and restarted the shared
   server, taking other users' jobs down with ours. It is usable only while
   under half full, measured at launch. Unreadable occupancy is a refusal:
   "I could not tell" is not "it is free".
2. **No GPU list, no run.** A silent fallback to card 0 turns "which card was
   this measured on" into an unanswerable question.
3. **`SKIP_DONE=1` means fill in what is missing**, not redo the ladder: a run
   counts as done only when its checkpoint *and* its test summary both exist.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "spur_mpm/experiments/run_e024.sh"


def _stub_smi(tmp_path: Path, answer: str) -> Path:
    """A fake nvidia-smi that always answers with `answer`."""
    path = tmp_path / "fake-nvidia-smi"
    path.write_text(f"#!/bin/sh\necho '{answer}'\n")
    path.chmod(0o755)
    return path


def _run(env: dict[str, str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items()
            if k not in {"GPUS", "DRY", "STAGE", "SKIP_DONE", "ALLOW_GPU2",
                         "FOLDS", "SEEDS", "OUT", "NVIDIA_SMI",
                         "GPU2_MAX_FRACTION"}}
    base.update(env)
    return subprocess.run(["bash", str(SCRIPT)], env=base, cwd=str(cwd or REPO),
                          capture_output=True, text=True, timeout=300)


# --- 1. GPU 2: usable while under half full, refused at or above ------------

def test_a_half_full_or_busier_gpu_two_is_refused(tmp_path: Path):
    smi = _stub_smi(tmp_path, "26274, 49140")          # 53.5 %, the real state
    p = _run({"GPUS": "0,2", "DRY": "0", "STAGE": "dual", "FOLDS": "0",
              "NVIDIA_SMI": str(smi)})
    assert p.returncode == 2, (p.stdout, p.stderr)
    assert "refusing to use GPU 2" in p.stderr, p.stderr
    assert "53.5%" in p.stderr, p.stderr        # the number it decided on
    assert "under half full" in p.stderr, p.stderr


def test_the_ceiling_is_inclusive_so_exactly_half_counts_as_full(tmp_path: Path):
    smi = _stub_smi(tmp_path, "24570, 49140")          # exactly 50 %
    p = _run({"GPUS": "2", "DRY": "1", "STAGE": "dual", "FOLDS": "0",
              "NVIDIA_SMI": str(smi)})
    assert p.returncode == 2, (p.stdout, p.stderr)
    assert "50.0%" in p.stderr, p.stderr


def test_a_quiet_gpu_two_may_be_used(tmp_path: Path):
    smi = _stub_smi(tmp_path, "18000, 49140")          # 36.6 %
    p = _run({"GPUS": "2", "DRY": "1", "STAGE": "dual", "FOLDS": "0",
              "NVIDIA_SMI": str(smi)})
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert "CUDA_VISIBLE_DEVICES=2" in p.stdout, p.stdout
    assert "under the ceiling" in p.stderr, p.stderr


def test_unreadable_occupancy_is_refused_rather_than_assumed_free(tmp_path: Path):
    smi = _stub_smi(tmp_path, "")
    p = _run({"GPUS": "2", "DRY": "1", "STAGE": "dual", "FOLDS": "0",
              "NVIDIA_SMI": str(smi)})
    assert p.returncode == 2, (p.stdout, p.stderr)
    assert "could not read its occupancy" in p.stderr, p.stderr
    assert "CUDA_VISIBLE_DEVICES=2" not in p.stdout, p.stdout


def test_the_override_is_explicit_and_named(tmp_path: Path):
    smi = _stub_smi(tmp_path, "47000, 49140")          # nearly full
    p = _run({"GPUS": "2", "DRY": "1", "STAGE": "dual", "FOLDS": "0",
              "NVIDIA_SMI": str(smi), "ALLOW_GPU2": "1"})
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert "CUDA_VISIBLE_DEVICES=2" in p.stdout, p.stdout


def test_the_threshold_is_configurable_but_has_a_default(tmp_path: Path):
    smi = _stub_smi(tmp_path, "18000, 49140")          # 36.6 %
    p = _run({"GPUS": "2", "DRY": "1", "STAGE": "dual", "FOLDS": "0",
              "NVIDIA_SMI": str(smi), "GPU2_MAX_FRACTION": "0.3"})
    assert p.returncode == 2, (p.stdout, p.stderr)


def test_a_gpu_list_without_card_two_never_consults_the_occupancy(tmp_path: Path):
    """The probe must not run for cards the rule does not govern -- otherwise a
    broken nvidia-smi blocks work on the cards that are fine."""
    smi = _stub_smi(tmp_path, "")
    p = _run({"GPUS": "0", "DRY": "1", "STAGE": "dual", "FOLDS": "0",
              "NVIDIA_SMI": str(smi)})
    assert p.returncode == 0, (p.stdout, p.stderr)


# --- 2. no silent default card ---------------------------------------------

def test_running_without_a_gpu_list_is_refused_rather_than_defaulted():
    p = _run({"GPUS": "", "DRY": "0", "STAGE": "dual", "FOLDS": "0"})
    assert p.returncode == 2, (p.returncode, p.stderr)
    assert "refusing to run" in p.stderr, p.stderr


# --- 3. SKIP_DONE -----------------------------------------------------------

def test_skip_done_leaves_a_finished_stage_alone(tmp_path: Path):
    """A finished run needs a summary too; a bare checkpoint is not evidence."""
    out = tmp_path / "ladder"
    for name in ("E024_P2_Direct_y_k2_fold0_s0", "E024_P2_ResidualI_y_k2_fold0_s0"):
        (out / name).mkdir(parents=True)
        (out / name / "best.pt").write_bytes(b"ckpt")
        (out / f"{name}_test").mkdir()
        (out / f"{name}_test" / "summary.json").write_text("{}")
    p = _run({"GPUS": "0", "DRY": "0", "STAGE": "residual", "FOLDS": "0",
              "SKIP_DONE": "1", "OUT": str(out)})
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert p.stdout.count("skip E024_P2_") == 2, p.stdout
    assert "CUDA_VISIBLE_DEVICES" not in p.stdout, p.stdout


def test_skip_done_still_launches_when_the_summary_is_missing(tmp_path: Path):
    """The half-run case: a checkpoint without a test summary is not done.

    Asserted through the refusal path rather than by launching: with an
    occupied card 2 in the list the script must get as far as refusing, which
    it only reaches after SKIP_DONE has said "not done".
    """
    smi = _stub_smi(tmp_path, "30000, 49140")
    out = tmp_path / "ladder"
    (out / "E024_P2_Direct_y_k2_fold0_s0").mkdir(parents=True)
    (out / "E024_P2_Direct_y_k2_fold0_s0" / "best.pt").write_bytes(b"ckpt")
    p = _run({"GPUS": "0,2", "DRY": "0", "STAGE": "residual", "FOLDS": "0",
              "SKIP_DONE": "1", "OUT": str(out), "NVIDIA_SMI": str(smi)})
    assert p.returncode == 2, (p.stdout, p.stderr)
    assert "refusing to run" not in p.stderr, p.stderr
    assert "refusing to use GPU 2" in p.stderr, p.stderr


# --- 4. one job per card ---------------------------------------------------
#
# Round-robin alone does not cap concurrency: hand in one card and a two-arm
# stage launches both halves on it, because wrapping around the list is not the
# same as there being room. On a card already carrying someone else's job that
# is an OOM that the scheduler caused. The fake `python` below records when each
# job starts and ends, so the number this pins is real overlap, not intent.

_FAKE_PY = """#!/bin/sh
echo "S $$ $(date +%s%N)" >> "$FAKE_PY_LOG"
sleep 0.6
echo "E $$ $(date +%s%N)" >> "$FAKE_PY_LOG"
exit 0
"""


def _peak_concurrency(path: Path) -> int:
    events = []
    for line in path.read_text().splitlines():
        kind, pid, ns = line.split()
        events.append((kind, pid, int(ns)))
    events.sort(key=lambda e: e[2])
    cur = peak = 0
    for kind, _pid, _ns in events:
        cur += 1 if kind == "S" else -1
        peak = max(peak, cur)
    return peak


def _run_with_fake_python(tmp_path: Path, gpus: str) -> tuple[int, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(_FAKE_PY)
    (bin_dir / "python").chmod(0o755)
    log = tmp_path / "jobs.log"
    log.write_text("")
    out = tmp_path / "ladder"
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_PY_LOG": str(log),
           "GPUS": gpus, "DRY": "0", "STAGE": "dual", "FOLDS": "0",
           "SEEDS": "0", "OUT": str(out)}
    p = _run(env)
    return p.returncode, log


def test_one_card_means_the_arms_queue_instead_of_sharing_it(tmp_path: Path):
    rc, log = _run_with_fake_python(tmp_path, "0")
    assert rc == 0, rc
    # two arms x (train + eval) = four python processes, each start + end
    assert log.read_text().count("\n") == 8, log.read_text()
    assert _peak_concurrency(log) == 1, log.read_text()


def test_two_cards_mean_the_arms_run_together(tmp_path: Path):
    rc, log = _run_with_fake_python(tmp_path, "0,1")
    assert rc == 0, rc
    assert _peak_concurrency(log) == 2, log.read_text()
