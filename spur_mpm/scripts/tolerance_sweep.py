"""How much does the skeleton score depend on the match tolerance?

``structure_report`` scores skeleton overlap with ``tolerance_vox=1``. That one
voxel is not one distance: the stacks are 0.98 um in XY and 5.0 um along Z, so
"1 voxel" forgives ten times more physical error in depth than across the
plane, and a reconstruction that misses a vessel axially is charged the same as
one that misses it laterally. The reviewer's objection (runbook §4.3) is that
"1 voxel" must not be read as "1 um", so this script re-scores the *same saved
masks* at a list of physical tolerances and reports how much the headline
number moves.

It reads the exported ``*_pred_mask.tif`` / ``*_ref_mask.tif`` rather than
reconstructing: the masks are the ones the published numbers came from, so the
sweep cannot drift away from the table it is qualifying, and it needs no GPU.

    python -m spur_mpm.scripts.tolerance_sweep \
        --structure runs/E021_ladder/E021_S1_y_k2/structure \
        --structure runs/E021_ladder/E021_A1_y_k2/structure \
        --out runs/E021_ladder/tolerance_sweep

Tolerances are physical (um) unless given as ``--voxel``, in which case they are
Euclidean distances in the sampled voxel grid, which is what the existing table
means.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from spur_mpm.metrics.skeleton import skeleton

# The distances a reader is likely to compare against: 0 (strict), one voxel
# across the plane, one micrometre, one axial step, and a step and a bit.
DEFAULT_TOLERANCES_UM = (0.0, 0.98, 1.0, 5.0, 10.0)


def matched_fraction(a: np.ndarray, b: np.ndarray, tolerance: float,
                     spacing=None) -> float:
    """Share of ``a``'s skeleton voxels within ``tolerance`` of ``b``'s.

    ``spacing`` makes the distance physical: with ``(z, y, x)`` micrometres the
    tolerance is in micrometres and the axial pitch is honoured. Passing
    ``None`` measures in voxels, which is what ``skeleton_overlap`` does.
    """
    if not a.any():
        return 1.0 if not b.any() else 0.0
    if not b.any():
        return 0.0
    distance = ndi.distance_transform_edt(~b, sampling=spacing)
    return float((distance[a] <= float(tolerance) + 1e-9).mean())


def overlap_at(pred_skel: np.ndarray, ref_skel: np.ndarray,
               tolerance: float, spacing=None) -> dict:
    precision = matched_fraction(pred_skel, ref_skel, tolerance, spacing)
    recall = matched_fraction(ref_skel, pred_skel, tolerance, spacing)
    total = precision + recall
    return {"precision": precision, "recall": recall,
            "f1": float(2 * precision * recall / total) if total > 0 else 0.0}


def _sampling_for(ref_tif: Path, recon_json: Path | None) -> tuple[float, ...] | None:
    """``(z, y, x)`` spacing in micrometres, or ``None`` if it is unknown.

    Unknown spacing is a hard failure rather than a silent fallback to voxels:
    the whole point of the sweep is the physical distance.
    """
    if recon_json is None or not recon_json.exists():
        return None
    geom = json.loads(recon_json.read_text()).get("geometry") or {}
    z, xy = geom.get("z_step_um"), geom.get("voxel_xy_um")
    if z is None or xy is None:
        return None
    return (float(z), float(xy), float(xy))


def sweep_volume(pred_tif: Path, ref_tif: Path, tolerances_um, tolerances_vox,
                 spacing) -> dict:
    pred_mask = tifffile.imread(pred_tif) > 0
    ref_mask = tifffile.imread(ref_tif) > 0
    if pred_mask.shape != ref_mask.shape:
        raise RuntimeError(f"{pred_tif.name}: prediction {pred_mask.shape} and "
                           f"reference {ref_mask.shape} are not the same volume")
    pred_skel, ref_skel = skeleton(pred_mask), skeleton(ref_mask)
    out = {"shape": list(pred_mask.shape),
           "pred_skeleton_voxels": int(pred_skel.sum()),
           "ref_skeleton_voxels": int(ref_skel.sum()),
           "tolerance_um": {}, "tolerance_vox": {}}
    for t in tolerances_um:
        out["tolerance_um"][f"{t:g}"] = overlap_at(pred_skel, ref_skel, t, spacing)
    for t in tolerances_vox:
        out["tolerance_vox"][f"{t:g}"] = overlap_at(pred_skel, ref_skel, t)
    return out


def _pairs(structure_dir: Path) -> list[tuple[str, Path, Path, Path | None]]:
    """Every ``(volume, pred tif, ref tif, recon json)`` in a structure directory."""
    found = []
    for pred in sorted(structure_dir.glob("*_pred_mask.tif")):
        stem = pred.name[: -len("_pred_mask.tif")]
        ref = structure_dir / f"{stem}_ref_mask.tif"
        if not ref.exists():
            continue
        recon = structure_dir / f"{stem}_recon.json"
        m = re.match(r"^(?P<arm>[A-Za-z0-9]+)_(?P<vol>.+)$", stem)
        found.append((m.group("vol") if m else stem, pred, ref, recon))
    if not found:
        raise RuntimeError(f"{structure_dir}: no *_pred_mask.tif / *_ref_mask.tif "
                           f"pairs -- run structure_report first")
    return found


def render(rows: list[dict], tolerances_um, tolerances_vox, spacing_note) -> str:
    md = ["# 骨架匹配容差敏感性", "",
          "同一对已导出的掩膜（`structure_report` 算指标用的那一对），只改匹配容差重算。"
          "物理容差用各向异性 EDT 计算，因此 Z 方向 5 µm 与 XY 方向 0.98 µm 在"
          "微米尺度上等价；`tolerance_vox` 那几列才是现行表里的“体素”口径。",
          "",
          f"体素间距：{spacing_note}。", ""]
    head = ("| arm | volume | 骨架体素 ref→pred | "
            + " | ".join(f"{t:g} µm" for t in tolerances_um)
            + " | " + " | ".join(f"{t:g} vox" for t in tolerances_vox) + " |")
    md += [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        cells = [f"{r['tolerance_um'][f'{t:g}']['f1']:.4f}" if f"{t:g}" in r["tolerance_um"]
                 and np.isfinite(r["tolerance_um"][f"{t:g}"]["f1"]) else "—"
                 for t in tolerances_um]
        vcells = [f"{r['tolerance_vox'][f'{t:g}']['f1']:.4f}" if f"{t:g}" in r["tolerance_vox"]
                  and np.isfinite(r["tolerance_vox"][f"{t:g}"]["f1"]) else "—"
                  for t in tolerances_vox]
        md.append(f"| {r['arm']} | {r['volume']} | "
                  f"{r['ref_skeleton_voxels']} → {r['pred_skeleton_voxels']} | "
                  + " | ".join(cells) + " | " + " | ".join(vcells) + " |")
    md.append("")
    return "\n".join(md)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", action="append", required=True,
                   help="structure output directory (repeatable)")
    p.add_argument("--tolerances-um", type=float, nargs="+",
                   default=list(DEFAULT_TOLERANCES_UM))
    p.add_argument("--tolerances-vox", type=float, nargs="+", default=[1.0])
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, spacings = [], set()
    for sdir in map(Path, args.structure):
        arm = sdir.parent.name.split("_")[1] if "_" in sdir.parent.name else sdir.parent.name
        for volume, pred, ref, recon in _pairs(sdir):
            spacing = _sampling_for(ref, recon)
            if spacing is None:
                raise SystemExit(f"{sdir}: {volume} has no voxel spacing in "
                                 f"{recon.name}; refusing to report a physical "
                                 f"tolerance without it")
            spacings.add(spacing)
            rec = sweep_volume(pred, ref, args.tolerances_um,
                               args.tolerances_vox, spacing)
            rec.update({"arm": arm, "volume": volume, "spacing_um": spacing,
                        "structure_dir": str(sdir)})
            rows.append(rec)
            print(f"{arm} {volume}: "
                  + " ".join(f"{t:g}um={rec['tolerance_um'][f'{t:g}']['f1']:.3f}"
                             for t in args.tolerances_um), flush=True)

    spacing_note = "; ".join(f"z={z:g} µm, xy={xy:g} µm" for z, xy, _ in sorted(spacings))
    (out_dir / "tolerance_sweep.json").write_text(json.dumps(
        {"tolerances_um": list(args.tolerances_um),
         "tolerances_vox": list(args.tolerances_vox),
         "volumes": rows}, indent=2))
    text = render(rows, args.tolerances_um, args.tolerances_vox, spacing_note)
    (out_dir / "tolerance_sweep.md").write_text(text + "\n")
    print(text)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
