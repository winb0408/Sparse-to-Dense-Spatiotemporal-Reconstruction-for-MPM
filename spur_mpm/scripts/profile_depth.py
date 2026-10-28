"""Per-plane intensity profile of every included volume.

Plan §2.2 asks the audit to record, per stack, the signal level by depth,
background, saturation and drift -- not just the shapes. Those numbers decide
how the data may legitimately be used:

* **saturation** at the top of the stack would make the calibration
  ``offset + scale`` collapse a range of real values onto 1.0, and every metric
  computed there would be measuring the clip, not the method;
* **depth attenuation** is why PSNR and vessel Dice move in opposite directions
  with depth (SPUR-E001) and why any depth comparison must be done in physical
  micrometres;
* **drift** between the first and last planes tells us whether a longitudinal
  session comparison is even meaningful.

Writes ``depth_profile.csv`` (one row per plane, all volumes) plus a markdown
summary, and a matplotlib figure when it is available.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from spur_mpm.data.datasets import VolumeCache

__all__ = ["profile_volume", "summarise"]


def profile_volume(cache: VolumeCache, sid: str, saturated_at: float | None = None,
                   chunk: int = 16) -> list[dict]:
    """One row per Z plane. Reads in chunks so a 200-plane stack stays in RAM."""
    arr = cache.array(sid)
    geom = cache.geometry(sid)
    step = float(geom.z_step_um or 1.0)
    origin = float(geom.depth_origin_um or 0.0)
    full = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) \
        else 1.0
    sat = full if saturated_at is None else float(saturated_at)

    rows: list[dict] = []
    for z0 in range(0, arr.shape[0], chunk):
        block = np.asarray(arr[z0:z0 + chunk], dtype=np.float32)
        for i in range(block.shape[0]):
            plane = block[i]
            rows.append({
                "sample_id": sid,
                "mouse_id": geom.mouse_id,
                "session_id": geom.session_id,
                "fov_id": geom.fov_id,
                "plane": z0 + i,
                "depth_um": (z0 + i) * step + origin,
                "mean": float(plane.mean()),
                "median": float(np.median(plane)),
                "p99": float(np.percentile(plane, 99)),
                "std": float(plane.std()),
                "zero_frac": float((plane == 0).mean()),
                "saturated_frac": float((plane >= sat).mean()),
            })
    return rows


def summarise(rows: list[dict], bands: int = 3) -> list[dict]:
    """Per-volume depth-band summary: the shape of the stack, in numbers."""
    by_volume: dict[str, list[dict]] = {}
    for row in rows:
        by_volume.setdefault(row["sample_id"], []).append(row)

    out = []
    for sid, planes in sorted(by_volume.items()):
        planes.sort(key=lambda r: r["plane"])
        n = len(planes)
        idx = [int(i * (n - 1) / max(1, bands - 1)) for i in range(bands)]
        bands_out = [{"depth_um": planes[i]["depth_um"],
                      "mean": planes[i]["mean"],
                      "p99": planes[i]["p99"]} for i in idx]
        top_mean = planes[0]["mean"]
        bottom_mean = planes[-1]["mean"]
        out.append({
            "sample_id": sid,
            "n_planes": n,
            "depth_span_um": planes[-1]["depth_um"] - planes[0]["depth_um"],
            "max_saturated_frac": max(p["saturated_frac"] for p in planes),
            "max_zero_frac": max(p["zero_frac"] for p in planes),
            "descending_ratio": (bottom_mean / top_mean) if top_mean else 0.0,
            "noisiest_plane": min(planes, key=lambda p: p["p99"])["plane"],
            "bands": bands_out,
            "problems": _problems(planes),
            "notes": _notes(planes),
        })
    return out


def _problems(planes: list[dict]) -> list[str]:
    """Conditions that would silently invalidate later numbers."""
    problems = []
    worst = max(p["saturated_frac"] for p in planes)
    if worst > 0.01:
        where = [p["plane"] for p in planes if p["saturated_frac"] > 0.01]
        problems.append(
            f"{worst:.1%} of a plane sits at full scale (planes {where[:5]}"
            f"{'...' if len(where) > 5 else ''}); the calibration clips there "
            "and any metric over those pixels measures the clip")
    if max(p["zero_frac"] for p in planes) > 0.5:
        problems.append("at least one plane is more than half exact zeros: "
                        "check for missing/blank pages rather than assuming "
                        "low signal")
    return problems


def _notes(planes: list[dict]) -> list[str]:
    """Properties that are not defects, but constrain how the data may be read."""
    notes = []
    p99 = [p["p99"] for p in planes]
    if p99 and max(p99) > 0:
        ratio = min(p99) / max(p99)
        if ratio < 0.85:
            notes.append(
                f"p99 falls to {ratio:.2f} of its maximum across the stack "
                f"(depth {planes[0]['depth_um']:.0f}->{planes[-1]['depth_um']:.0f} "
                "µm): the stack mixes an SNR regime, so depth-pooled metrics "
                "average over it and bands must be reported separately")
    return notes


def write_markdown(summaries: list[dict], out_path: Path) -> None:
    lines = ["# 深度剖面（每卷）", "",
             "由 `scripts/profile_depth.py` 生成；数值来自原始 16-bit 数据，未经归一化。", "",
             "| sample | 层数 | 深度跨度 (µm) | 首/末层均值比 | 最大饱和比例 | 最深 p99/最浅 p99 |",
             "|---|---|---|---|---|---|"]
    for s in summaries:
        bands = s["bands"]
        ratio = (bands[-1]["p99"] / bands[0]["p99"]) if bands[0]["p99"] else 0.0
        lines.append(f"| {s['sample_id']} | {s['n_planes']} | "
                     f"{s['depth_span_um']:.0f} | {s['descending_ratio']:.2f} | "
                     f"{s['max_saturated_frac']:.2%} | {ratio:.2f} |")
    flagged = [s for s in summaries if s["problems"]]
    lines += ["", "## 数据缺陷", ""]
    if not flagged:
        lines.append("（无：无饱和、无空白层）")
    for s in flagged:
        lines.append(f"- **{s['sample_id']}**")
        for prob in s["problems"]:
            lines.append(f"  - {prob}")

    noted = [s for s in summaries if s.get("notes")]
    lines += ["", "## 分析约束（不是缺陷，但必须在分析中体现）", ""]
    if not noted:
        lines.append("（无）")
    for s in noted:
        lines.append(f"- **{s['sample_id']}**")
        for note in s["notes"]:
            lines.append(f"  - {note}")
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--chunk", type=int, default=16)
    args = p.parse_args(argv)

    cache = VolumeCache.load(args.index, fold=args.fold)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sid in cache.volumes:
        rows += profile_volume(cache, sid, chunk=args.chunk)

    csv_path = args.out_dir / "depth_profile.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = summarise(rows)
    (args.out_dir / "depth_profile.json").write_text(
        json.dumps(summaries, indent=1, ensure_ascii=False))
    write_markdown(summaries, args.out_dir / "depth_profile.md")

    for s in summaries:
        flag = "  <-- 数据缺陷" if s["problems"] else ""
        print(f"{s['sample_id']:26s} planes={s['n_planes']:4d} "
              f"span={s['depth_span_um']:6.0f} µm "
              f"sat={s['max_saturated_frac']:.2%}{flag}")
    print(f"wrote {csv_path} and depth_profile.{{md,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
