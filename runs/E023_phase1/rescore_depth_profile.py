"""Repair the per-slab depth profile of a stored structure summary.

Why this exists
---------------
The whole-volume structure runs for A1 and S1 were made *before* the empty-slab
rule changed. Back then a slab whose reference mask is empty scored
``dice = 1.0`` (two empty masks "agree perfectly"), which rewards exactly the
deep, low-signal slabs the profile exists to expose -- three of six slabs on the
worst volume were scored that way. Volume-level numbers are unaffected (the masks
are non-empty there); the stored ``depth_profile`` is not.

Why it patches instead of recomputing
-------------------------------------
The code change was one guarded branch: *if* ``slab["n_empty"]`` then the three
mask scores become NaN. Every other slab's arithmetic is untouched, so
recomputing eight volumes (a skeleton + clDice per slab, ~10 min each) would
reproduce the stored numbers and change only the empty slabs. ``--verify`` runs
that recomputation on one volume and asserts the non-empty slabs match the stored
ones bit-for-bit before any patching happens; with that check passed, the patch
is exactly what a full re-run would produce, at no JPEG-of-a-JPEG risk.

    PYTHONPATH=. python runs/E023_phase1/rescore_depth_profile.py --verify \
        runs/E021_ladder/E021_S1_y_k2/structure runs/E021_ladder/E021_A1_y_k2/structure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from spur_mpm.metrics.skeleton import depth_profile
from spur_mpm.scripts.structure_report import render_summary

SCORED = ("dice", "cl_dice", "f1")


def _slabs(summary_dir: Path) -> dict:
    return json.loads((summary_dir / "structure_summary.json").read_text())


def _geometry(vol: dict) -> dict:
    return vol["recon_meta"]["meta"]["geometry"]


def verify_one(summary_dir: Path, sample_prefix: str) -> None:
    """Recompute ONE volume's slabs from its exported masks and compare."""
    summary = _slabs(summary_dir)
    for vol in summary["volumes"]:
        if not vol["sample_id"].startswith(sample_prefix):
            continue
        geom = _geometry(vol)
        fresh = depth_profile(
            tifffile.imread(vol["pred_mask_path"]) > 0,
            tifffile.imread(vol["ref_mask_path"]) > 0,
            voxel_um=(geom["z_step_um"], geom["voxel_xy_um"], geom["voxel_xy_um"]),
            n_slabs=len(vol["depth_profile"]),
            tolerance_vox=vol.get("tolerance_vox", 1))
        worst = 0.0
        for old, new in zip(vol["depth_profile"], fresh):
            assert int(old["n_empty"]) == int(new["n_empty"]), (old, new)
            if old["n_empty"]:
                assert np.isnan(new["dice"]), "the fix must NaN an empty slab"
                continue
            for key in SCORED:
                if key in old and key in new:
                    worst = max(worst, abs(float(old[key]) - float(new[key])))
        print(f"verified {vol['sample_id']}: {len(fresh)} slabs, max |old-new| on "
              f"non-empty slabs = {worst:.3e}")
        assert worst < 1e-9, "stored slabs differ from a fresh computation"
        return
    raise SystemExit(f"no volume starts with {sample_prefix!r}")


def repair(summary_dir: Path) -> str:
    path = summary_dir / "structure_summary.json"
    summary = _slabs(summary_dir)
    changed = 0
    for vol in summary["volumes"]:
        for slab in vol["depth_profile"]:
            if not slab.get("n_empty"):
                continue
            for key in SCORED:
                if key in slab:
                    slab[key] = float("nan")
            changed += 1
        vol["depth_profile_note"] = (
            "empty-reference slabs are NaN, not 1.0 (mask scores of two empty "
            "masks read as perfect agreement and rewarded deep, low-signal "
            "slabs); patched after verifying against a fresh recomputation")
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    md = summary_dir / "structure_summary.md"
    md.write_text(render_summary(summary))
    print(f"{summary_dir}: {changed} empty slabs set to NaN")
    return str(md)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--verify" in sys.argv:
        # One volume per summary is enough: the branch under test does not
        # depend on the volume, only on whether the slab's reference is empty.
        for d in args:
            verify_one(Path(d), "M2_FOVII_S1")
    for d in args:
        print("wrote", repair(Path(d)))
