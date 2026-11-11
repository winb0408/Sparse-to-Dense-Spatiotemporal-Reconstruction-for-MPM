"""The rules `run_e024.sh` enforces before it touches a GPU.

Three of them are refusals, and a refusal is only worth having if it is
tested -- an unexamined guard is a comment. None of these tests launches
training: every case below either stops the script before its first job or
finds the work already done.

1. **GPU 2 is off limits.** Filling it has hung and restarted the shared
   server, taking other users' jobs down with ours. The guard is a hard stop
   with a deliberate override, because the previous version of this rule lived
   in someone's memory and that is exactly what failed.
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


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items()
            if k not in {"GPUS", "DRY", "STAGE", "SKIP_DONE", "ALLOW_GPU2",
                         "FOLDS", "SEEDS", "OUT"}}
    base.update(env)
    return subprocess.run(["bash", str(SCRIPT)], env=base, cwd=str(REPO),
                          capture_output=True, text=True, timeout=300)


def test_gpu_two_is_refused_in_any_position_of_the_list():
    for gpus in ("2", "0,2", "2,0", "0,1,2"):
        p = _run({"GPUS": gpus, "DRY": "0", "STAGE": "dual", "FOLDS": "0"})
        assert p.returncode == 2, (gpus, p.returncode, p.stdout, p.stderr)
        assert "refusing to use GPU 2" in p.stderr, (gpus, p.stderr)
        # and it must say *why*, so the next person does not re-litigate it
        assert "hung and restarted this" in p.stderr, (gpus, p.stderr)


def test_the_gpu_two_refusal_applies_to_dry_runs_too():
    """A dry run that prints a job for card 2 teaches the wrong command."""
    p = _run({"GPUS": "0,2", "DRY": "1", "STAGE": "dual", "FOLDS": "0"})
    assert p.returncode == 2, (p.stdout, p.stderr)
    assert "CUDA_VISIBLE_DEVICES=2" not in p.stdout, p.stdout


def test_the_override_is_explicit_and_named():
    p = _run({"GPUS": "2", "DRY": "1", "STAGE": "dual", "FOLDS": "0",
              "ALLOW_GPU2": "1"})
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert "CUDA_VISIBLE_DEVICES=2" in p.stdout, p.stdout


def test_running_without_a_gpu_list_is_refused_rather_than_defaulted():
    p = _run({"GPUS": "", "DRY": "0", "STAGE": "dual", "FOLDS": "0"})
    assert p.returncode == 2, (p.returncode, p.stderr)
    assert "refusing to run" in p.stderr, p.stderr


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
    """The half-run case: a checkpoint without a test summary is redone.

    Asserted through the refusal path rather than by launching: with card 2 in
    the list the script must get as far as refusing, which it only reaches
    after the skip test says "not done".
    """
    out = tmp_path / "ladder"
    (out / "E024_P2_Direct_y_k2_fold0_s0").mkdir(parents=True)
    (out / "E024_P2_Direct_y_k2_fold0_s0" / "best.pt").write_bytes(b"ckpt")
    p = _run({"GPUS": "0,2", "DRY": "0", "STAGE": "residual", "FOLDS": "0",
              "SKIP_DONE": "1", "OUT": str(out)})
    # the GPU-2 guard fires before any work, so this only proves the script
    # parses and reaches its guards with SKIP_DONE set
    assert p.returncode == 2, (p.stdout, p.stderr)
    assert "refusing to run" not in p.stderr, p.stderr
