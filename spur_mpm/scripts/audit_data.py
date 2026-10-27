"""SPUR-E000: audit a raw TIFF directory into a traceable manifest.

    python -m spur_mpm.scripts.audit_data \
        --spec spur_mpm/configs/data/dryad_3p_vessels.yaml \
        --out "spur_mpm/experiments/E000_data_audit"

Fails (exit code 1) when any volume still has unresolved issues, because
training on a volume whose axes or animal identity are unknown is precisely the
mistake this step exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from spur_mpm.data.manifest import build_manifest, load_spec, write_manifest

__all__ = ["run_audit", "write_report"]


def run_audit(spec_path: str | Path, out_dir: str | Path,
              with_sha256: bool = False, limit: int | None = None) -> dict:
    spec = load_spec(spec_path)
    rows = build_manifest(spec, with_sha256=with_sha256, limit=limit)
    if not rows:
        raise SystemExit(f"no TIFF files matched the spec under {spec.root}")

    csv_path, json_path = write_manifest(rows, out_dir, spec_path=spec_path,
                                         root=spec.root)
    report = write_report(rows, out_dir, spec)
    return {"rows": rows, "csv": csv_path, "json": json_path, "report": report}


def write_report(rows: list[dict], out_dir: str | Path, spec) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_bad = sum(1 for r in rows if r.get("issues"))
    shapes = Counter(str(r.get("shape")) for r in rows)
    dtypes = Counter(str(r.get("dtype")) for r in rows)
    zspans = sorted({round(float(r.get("z_span_um") or 0), 1) for r in rows})
    mice = sorted({str(r.get("mouse_id", "")) for r in rows})
    fovs = sorted({str(r.get("fov_id", "")) for r in rows})

    lines = [
        "# SPUR-E000 数据审计报告",
        "",
        f"- 数据集目录: `{spec.root}`",
        f"- 文件数: **{len(rows)}**（有 issue 的 {n_bad} 个）",
        f"- 动物 (mouse_id): {mice}",
        f"- FOV: {fovs}",
        f"- Z 覆盖 (µm): {zspans}",
        "",
        "## 形状与类型",
        "",
        "| shape | 文件数 |",
        "|---|---|",
    ]
    lines += [f"| {s} | {n} |" for s, n in shapes.most_common()]
    lines += ["", "| dtype | 文件数 |", "|---|---|"]
    lines += [f"| {d} | {n} |" for d, n in dtypes.most_common()]

    lines += ["", "## 逐文件清单", "",
              "| sample_id | mouse | session | fov | axes | shape | z_span_um | issues |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('sample_id')} | {r.get('mouse_id')} | {r.get('session_id')} | "
            f"{r.get('fov_id')} | {r.get('axes')} | {r.get('shape')} | "
            f"{r.get('z_span_um')} | {r.get('issues') or '—'} |")

    if n_bad:
        lines += ["", "## 未解决项（必须先解决才能训练）", ""]
        for r in rows:
            if r.get("issues"):
                lines.append(f"- `{r.get('sample_id')}`: {r['issues']}")
    warnings = [(r.get("sample_id"), r["warnings"]) for r in rows if r.get("warnings")]
    if warnings:
        lines += ["", "## 警告（不阻塞，但需人工确认）", ""]
        for sid, w in warnings:
            lines.append(f"- `{sid}`: {w}")

    path = out / "audit_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sha256", action="store_true",
                   help="hash every file (slow, needed for the formal audit)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-issues", action="store_true",
                   help="exit 0 even when volumes have unresolved issues")
    args = p.parse_args(argv)

    result = run_audit(args.spec, args.out, with_sha256=args.sha256,
                       limit=args.limit)
    rows = result["rows"]
    n_bad = sum(1 for r in rows if r.get("issues"))
    summary = {
        "n_files": len(rows),
        "n_with_issues": n_bad,
        "manifest_csv": str(result["csv"]),
        "manifest_json": str(result["json"]),
        "report": str(result["report"]),
    }
    print(json.dumps(summary, indent=2))
    if n_bad and not args.allow_issues:
        print(f"\n{n_bad} volume(s) still have unresolved issues; fix the dataset "
              "spec (axes / mouse_id / z_step) before training.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
