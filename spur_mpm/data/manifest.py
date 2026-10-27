"""Data audit: turn a directory of TIFF stacks into a machine-readable manifest.

This is the implementation of step SPUR-E000 (plan §2.2). It never guesses
axis order from array shape: metadata comes from the dataset spec, from the
filename, or from the TIFF's own series description -- and anything still
unknown is written to an ``issues`` column so the audit can fail loudly.

Outputs (under the audit directory)::

    manifest.csv    one row per volume, human-inspectable
    manifest.json   same rows + issues + spec hash, for code
    audit_report.md short summary written by scripts/audit_data.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from spur_mpm.data.geometry import (
    AUDIT_FIELDS,
    GeometryError,
    VolumeGeometry,
    check_axis_consistency,
)

__all__ = ["DatasetSpec", "load_spec", "probe_tiff", "build_manifest",
           "write_manifest", "read_manifest", "sha256_of"]

_KNOWN_SERIES_AXES = set("TZCYXS")
#: tifffile uses these for "an unnamed page sequence".
_UNKNOWN_SERIES_AXES = set("QI")


@dataclass
class FileRule:
    """Glob pattern plus the metadata every match inherits."""

    glob: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DatasetSpec:
    name: str
    root: Path
    defaults: dict = field(default_factory=dict)
    rules: list[FileRule] = field(default_factory=list)
    filename_regex: str | None = None
    per_file: dict = field(default_factory=dict)

    def match(self, path: Path) -> dict:
        """Metadata for one file: defaults <- first matching rule <- per-file."""
        meta = dict(self.defaults)
        rel = path.as_posix()
        for rule in self.rules:
            if path.match(rule.glob) or Path(rel).match(rule.glob):
                meta.update(rule.metadata)
                break
        for key, override in self.per_file.items():
            # per_file keys are exact basenames or glob fragments
            if path.name == key or path.match(key):
                meta.update(override)
        if self.filename_regex:
            m = re.search(self.filename_regex, path.name)
            if m:
                meta.update({k: v for k, v in m.groupdict().items() if v is not None})
        return meta


def load_spec(path: str | Path, repo_root: Path | None = None) -> DatasetSpec:
    """Load a dataset YAML into a :class:`DatasetSpec`.

    ``root`` inside the YAML may be relative; it is resolved against
    ``repo_root`` (default: the repository root inferred from this file).
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    root = Path(raw.get("root", "."))
    if not root.is_absolute():
        base = repo_root or path.resolve().parents[2]
        root = (base / root).resolve()
    rules = [FileRule(glob=r["glob"], metadata=r.get("metadata", {}))
             for r in raw.get("files", [])]
    return DatasetSpec(
        name=raw.get("name", path.stem),
        root=root,
        defaults=raw.get("defaults", {}) or {},
        rules=rules,
        filename_regex=raw.get("filename_regex"),
        per_file=raw.get("per_file", {}) or {},
    )


def sha256_of(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe_tiff(path: str | Path) -> dict:
    """Read shape/dtype/axis hints from a TIFF without loading all pixels."""
    import tifffile

    info: dict = {"path": str(path), "bytes": Path(path).stat().st_size}
    with tifffile.TiffFile(path) as tf:
        info["pages"] = len(tf.pages)
        try:
            series = tf.series[0]
            info["series_shape"] = list(series.shape)
            info["series_axes"] = str(series.axes)
        except Exception:                      # pragma: no cover - exotic TIFFs
            info["series_shape"] = []
            info["series_axes"] = ""
        page = tf.pages[0]
        info["dtype"] = str(np.dtype(page.dtype))
        info["page_shape"] = list(page.shape)
        desc = (page.description or "").strip().replace("\n", " ")
        info["description"] = desc[:300]
        info["imagej"] = bool(tf.is_imagej)
        if info["imagej"] and tf.imagej_metadata:
            meta = tf.imagej_metadata
            info["imagej_spacing"] = meta.get("spacing")
            info["imagej_unit"] = meta.get("unit")
    return info


def build_manifest(spec: DatasetSpec, with_sha256: bool = False,
                   limit: int | None = None) -> list[dict]:
    """Scan every TIFF under ``spec.root`` and resolve its metadata."""
    if not spec.root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {spec.root}")

    patterns = [r.glob for r in spec.rules] or ["**/*.tif", "**/*.tiff"]
    seen: set[Path] = set()
    paths: list[Path] = []
    for pat in patterns:
        for p in sorted(spec.root.glob(pat)):
            if p.suffix.lower() in (".tif", ".tiff") and p not in seen:
                seen.add(p)
                paths.append(p)
    if limit:
        paths = paths[:limit]

    rows: list[dict] = []
    for path in paths:
        rows.append(_manifest_row(spec, path, with_sha256))
    return rows


def _manifest_row(spec: DatasetSpec, path: Path, with_sha256: bool) -> dict:
    meta = spec.match(path)
    issues: list[str] = []

    try:
        probe = probe_tiff(path)
    except Exception as exc:                   # pragma: no cover - corrupt files
        probe = {"path": str(path), "dtype": "", "pages": None,
                 "series_shape": [], "series_axes": ""}
        issues.append(f"unreadable: {exc}")

    declared_axes = str(meta.get("axes", "")).upper()
    series_axes = str(probe.get("series_axes", "")).upper()

    if not declared_axes:
        issues.append("axes not declared in spec")
        declared_axes = ""
    # Cross-check against what tifffile believes, only when it knows.
    if (series_axes and declared_axes and series_axes in _KNOWN_SERIES_AXES
            and not set(series_axes) <= _UNKNOWN_SERIES_AXES
            and series_axes != declared_axes):
        issues.append(f"declared axes {declared_axes} != TIFF series axes {series_axes}")

    shape = tuple(probe.get("series_shape") or [])
    geom = VolumeGeometry(
        sample_id=str(meta.get("sample_id") or path.stem),
        mouse_id=str(meta.get("mouse_id", "")),
        session_id=str(meta.get("session_id", "")),
        fov_id=str(meta.get("fov_id", "")),
        channel=str(meta.get("channel", "")),
        axes=declared_axes or "?",
        shape=shape,
        dtype=str(probe.get("dtype", "")),
        voxel_xy_um=_as_float(meta.get("voxel_xy_um", 1.0)),
        z_step_um=_as_float(meta.get("z_step_um", 1.0)),
        excitation_nm=_as_int(meta.get("excitation_nm")),
        power_mw=_as_float(meta.get("power_mw")),
        dwell_time_us=_as_float(meta.get("dwell_time_us")),
        depth_origin_um=_as_float(meta.get("depth_origin_um")),
        source=str(path),
    )

    warnings: list[str] = []
    if declared_axes and shape:
        try:
            warnings = geom.validate(actual_shape=shape,
                                     declared_planes=probe.get("pages"))
        except GeometryError as exc:
            issues.append(str(exc))
        else:
            warnings += check_axis_consistency(shape, declared_axes)

    for key in ("mouse_id", "session_id", "fov_id"):
        if not getattr(geom, key):
            issues.append(f"{key} unresolved")

    # ImageJ pixel spacing is only a hint: report disagreement, never apply it.
    ij_spacing = probe.get("imagej_spacing")
    if ij_spacing and abs(float(ij_spacing) - geom.z_step_um) > 1e-6:
        warnings.append(f"ImageJ spacing {ij_spacing} vs spec z_step "
                        f"{geom.z_step_um} -- spec wins, reconcile manually")

    row = {k: getattr(geom, k) for k in AUDIT_FIELDS}
    row["shape"] = list(shape)
    row["path"] = str(path.relative_to(spec.root)) if path.is_relative_to(spec.root) else str(path)
    row["bytes"] = probe.get("bytes", 0)
    row["pages"] = probe.get("pages")
    row["tiff_series_axes"] = series_axes
    row["z_span_um"] = round(geom.depth_span_um, 3) if geom.depth else 0.0
    row["sha256"] = sha256_of(path) if with_sha256 else ""
    row["issues"] = "; ".join(issues)
    row["warnings"] = "; ".join(warnings)
    return row


def _as_float(v):
    return None if v is None else float(v)


def _as_int(v):
    return None if v is None else int(v)


def write_manifest(rows: list[dict], out_dir: str | Path,
                   spec_path: str | Path | None = None,
                   root: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``manifest.csv`` + ``manifest.json``; return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    columns = list(AUDIT_FIELDS) + ["path", "bytes", "pages", "tiff_series_axes",
                                    "z_span_um", "sha256", "issues", "warnings"]
    csv_path = out / "manifest.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _flat(row.get(k, "")) for k in columns})

    payload = {
        "n_files": len(rows),
        "dataset_root": str(Path(root).resolve()) if root else None,
        "spec": str(spec_path) if spec_path else None,
        "spec_sha256": sha256_of(spec_path) if spec_path and Path(spec_path).exists() else None,
        "rows": rows,
        "n_files_with_issues": sum(1 for r in rows if r.get("issues")),
    }
    json_path = out / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2))
    return csv_path, json_path


def read_manifest(path: str | Path) -> list[dict]:
    """Load manifest rows from either the CSV or the JSON form."""
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())["rows"]
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def read_manifest_meta(path: str | Path) -> dict:
    """Manifest payload without the rows (root, spec hash, counts).

    Returns ``{}`` for the CSV form, which carries no header information; callers
    must then resolve paths themselves and say so.
    """
    path = Path(path)
    if path.suffix != ".json":
        return {}
    payload = json.loads(path.read_text())
    payload.pop("rows", None)
    return payload


def resolve_row_path(row: dict, manifest_path: str | Path,
                     root: str | Path | None = None) -> Path:
    """Absolute path of a manifest row's TIFF.

    Manifest rows store paths *relative to the dataset root*, not to the
    manifest file, because the manifest is written into a separate audit
    directory. Resolution therefore tries, in order: the recorded dataset root,
    the manifest file's own directory (CSV-only manifests), then the bare path.
    """
    raw = Path(str(row["path"]))
    if raw.is_absolute():
        return raw
    candidates = []
    root = root or read_manifest_meta(manifest_path).get("dataset_root")
    if root:
        candidates.append(Path(root) / raw)
    candidates.append(Path(manifest_path).resolve().parent / raw)
    candidates.append(raw)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _flat(v):
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v)
    if v is None:
        return ""
    return v
