"""Fig. 2: the method (a) and the ablation matrix (b) -- generated, not drawn.

**Why the matrix is computed from the registry.** The ablation matrix is the
paper's claim about *what differs between arms*. Typed by hand it would be a
second copy of ``spur_mpm/models/registry.py`` that drifts silently: a later arm
edit (say, giving S1 an uncertainty channel) would leave the figure asserting a
comparison nobody ran, and nothing in the pipeline would notice. So every tick
here is derived from the ``Arm`` record itself -- ``features``, ``fusion``,
``needs_training``, ``warm_start``, ``supervision`` -- and each band of the
matrix is derived from ``supervision`` / ``needs_training`` rather than from a
list of names. The provenance JSON written beside the PNGs records, per cell and
per number, the field it came from, so the figure can be re-derived without
being trusted.

**The pipeline panel is held to the same rule from the other side.** Block
names, warm-start mode strings, lattice modes and channel counts are imported
or counted from the code (``physics/wiener.WARM_START_MODES``,
``physics/sampling.MODES``, ``len(arm.features)``, ``build_model``'s
``in_ch = len(features) * t``) instead of being written in prose that could go
stale. Which optional block an arm switches on is printed from the registry,
not from memory.

**Every number on the figure comes from a file.** Parameter counts, MACs and
latency are read from ``runs/E021_ladder/benchmark_supervised.json`` with
``runs/E021_ladder/complexity.json`` as fallback. That order is deliberate:
``benchmark_supervised.json`` is the machine-readable twin of
``runs/E021_ladder/complexity.md`` (its latency column is the one the .md
quotes, digit for digit), it covers all seven Phase-1 arms including S2g, and it
carries a measured warm-start latency for A0/A1 instead of ``null``. Where the
two files disagree the figure prints the first and the provenance JSON records
the other, so a disagreement is visible rather than averaged away. A cell that
is null everywhere prints ``n/a``; it is never turned into 0.

**Phase-1 scope is drawn, not implied.** The self-supervised arms were frozen on
2026-10-30 (``archive/selfsup_v1/``). They keep their design ticks -- so the
matrix still says what they were -- but their number cells read ``frozen``
instead of being blank, because an empty cell reads as a failure when the truth
is "not measured in this phase".

Outputs (``--out``, default ``spur_mpm/figures/framework``)::

    fig2_pipeline.png          (a) sparse observation -> X0/prior fields -> shared trunk
    fig2_ablation_matrix.png   (b) which component each arm turns on, plus its cost
    fig2_provenance.json       per-tick / per-number source, machine readable

    python -m spur_mpm.scripts.framework_figure
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from spur_mpm.models.registry import ARMS, Arm
from spur_mpm.physics.sampling import MODES
from spur_mpm.physics.wiener import WARM_START_MODES

__all__ = [
    "NA", "FROZEN", "NOT_RUN", "REGISTRY", "ARCHIVE", "NUM_SOURCES",
    "Column", "Number", "Cell", "Row", "Block", "COLUMNS", "GROUPS",
    "load_numbers", "format_number", "group_of", "cell_for", "matrix_row",
    "matrix_rows", "matrix_blocks", "pipeline_facts", "provenance",
    "render_pipeline", "render_matrix", "render", "main",
]

#: Repo root, so the documented relative defaults work from any cwd.
_REPO = Path(__file__).resolve().parents[2]

REGISTRY = "spur_mpm/models/registry.py"
ARCHIVE = "archive/selfsup_v1/README.md"
BENCHMARK = Path("runs/E021_ladder/benchmark_supervised.json")
COMPLEXITY = Path("runs/E021_ladder/complexity.json")
NUM_SOURCES: tuple[Path, ...] = (BENCHMARK, COMPLEXITY)

NA = "n/a"
FROZEN = "frozen"
#: A PR2 arm that exists in code and has no real-data number yet. Saying
#: "NOT_RUN" is the only honest cell: "n/a" would read as "not applicable".
NOT_RUN = "NOT_RUN"

INK = "#1f3864"          # measured / present
INK_2 = "#2f5d8c"
OPT = "#a15c00"          # optional block
GREY = "#9d9d9d"         # frozen arm
INK_3 = "#7f9ac4"        # implemented, not yet run
RULE = "#d9dde3"

_TICKED = ("y", "mask", "x0", "uncertainty")
_TRACKED = ("params_m", "macs_per_slice", "latency_ms_per_slice")
#: Display scale per source field (unit conversion only, no re-measurement).
_SCALE = {"params_m": 1.0, "macs_per_slice": 1e-9, "latency_ms_per_slice": 1.0}


def _resolve(path: Path | str) -> Path:
    """Inputs are documented relative to the repo root, not to the cwd."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    alt = _REPO / p
    return alt if alt.exists() else p


# ---------------------------------------------------------------------------
# numbers: read from file, keep nulls as nulls
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Number:
    """One measured quantity, with the file and field it was read from."""

    value: float | None
    source: str
    field: str
    note: str = ""
    fallback: bool = False           # first source had no value; a later one did
    alternates: tuple[str, ...] = ()  # other files that hold a different value


def load_numbers(sources: Sequence[Path | str] = NUM_SOURCES
                 ) -> dict[str, dict[str, Number]]:
    """Per ``(arm, field)`` the first non-null value, in source order.

    ``None`` is kept rather than dropped: the caller has to be able to print
    ``n/a``. Files that disagree are recorded in ``alternates`` instead of being
    silently overridden -- latency was measured twice (10 and 50 timing
    iterations) and the two runs differ by up to 1.2 ms/slice, which a reader is
    entitled to see.
    """
    seen: dict[str, dict[str, list[tuple[str, float | None, str]]]] = {}
    for src in dict.fromkeys(str(s) for s in sources):      # keep order, no repeats
        doc = json.loads(_resolve(src).read_text(encoding="utf-8"))
        for entry in doc.get("arms", []):
            cells = seen.setdefault(entry["arm"], {})
            for field in _TRACKED:
                raw = entry.get(field)
                cells.setdefault(field, []).append(
                    (src, None if raw is None else float(raw), str(entry.get("note", ""))))

    out: dict[str, dict[str, Number]] = {}
    for arm, fields in seen.items():
        out[arm] = {}
        for field, tries in fields.items():
            chosen = next(((p, v, n) for p, v, n in tries if v is not None), None)
            if chosen is None:
                out[arm][field] = Number(
                    None, "; ".join(dict.fromkeys(p for p, _, _ in tries)),
                    f"arms[{arm}].{field} (null in every source)",
                    next((n for _, _, n in reversed(tries) if n), ""))
                continue
            path, value, note = chosen
            alts = tuple(f"{p}:{v!r}" for p, v, _ in tries
                         if v is not None and v != value)
            out[arm][field] = Number(
                value, path, f"arms[{arm}].{field}", note,
                fallback=any(v is None for _, v, _ in tries),
                alternates=alts)
    return out


def format_number(number: Number | None, field: str) -> str:
    """Display text for one cell. A null source value stays ``n/a``.

    Units are the only thing changed here (``macs_per_slice`` is shown in
    GMAC/slice); nothing is re-measured or rounded up.
    """
    if number is None or number.value is None:
        return NA
    v = number.value * _SCALE[field]
    if field == "params_m":
        return f"{v:.3f}"
    if field == "macs_per_slice":
        return f"{v:.2f}"
    return f"{v:.3f}" if v < 1.0 else f"{v:.2f}"


# ---------------------------------------------------------------------------
# the matrix: one column per thing an arm can turn on
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Column:
    key: str
    label: str
    kind: str      # tick | text | number
    field: str     # Arm attribute, or the number-source field
    rule: str      # the exact rule, copied into the provenance JSON


COLUMNS: tuple[Column, ...] = (
    Column("y", "Y", "tick", "features", "'y' in arm.features"),
    Column("mask", "M", "tick", "features", "'mask' in arm.features"),
    Column("x0", "X0\nin", "tick", "features", "'x0' in arm.features"),
    Column("uncertainty", "U", "tick", "features", "'uncertainty' in arm.features"),
    Column("fusion", "Gate", "tick", "fusion", "arm.fusion"),
    Column("trunk", "Swin\nUNet", "tick", "needs_training", "arm.needs_training"),
    Column("warm_start", "warm start", "text", "warm_start", "arm.warm_start"),
    Column("supervision", "super-\nvision", "text", "supervision", "arm.supervision"),
    Column("params", "params\n(M)", "number", "params_m", "source field params_m"),
    Column("macs", "MACs/slice\n(G)", "number", "macs_per_slice",
           "source field macs_per_slice / 1e9"),
    Column("latency", "ms/slice", "number", "latency_ms_per_slice",
           "source field latency_ms_per_slice"),
)

COLUMN_BY_KEY = {c.key: c for c in COLUMNS}

GROUPS: tuple[tuple[str, str], ...] = (
    ("floors", "no-learning floors  (needs_training = False) - the reference a "
               "network has to beat"),
    ("supervised", "supervised - measured in Phase 1"),
    ("planned", "PR2 dual-prior - implemented in code, real-data run NOT_RUN "
                "(no free GPU; see docs/PHASE1_PR2_ACCEPTANCE.md)"),
    ("frozen", "self-supervised - frozen 2026-10-30, NOT measured in Phase 1 "
               "(archive/selfsup_v1)"),
)


@dataclass(frozen=True)
class Cell:
    text: str
    tick: bool | None
    source: str
    field: str
    rule: str
    note: str = ""
    alternates: tuple[str, ...] = ()


@dataclass(frozen=True)
class Row:
    arm: Arm
    group: str
    cells: dict[str, Cell]
    frozen: bool
    planned: bool = False

    def cell(self, key: str) -> Cell:
        return self.cells[key]


@dataclass(frozen=True)
class Block:
    key: str
    title: str
    rows: tuple[Row, ...]


def group_of(arm: Arm) -> str:
    """Which band of the matrix an arm belongs to -- derived, never listed.

    A self-supervised arm is frozen whatever else is true of it; otherwise the
    presence of a network decides floor vs supervised arm. Listing the band
    members by hand is exactly the drift this script exists to prevent.
    """
    if arm.supervision == "self":
        return "frozen"
    return "supervised" if arm.needs_training else "floors"


def cell_for(arm: Arm, column: Column,
             numbers: dict[str, dict[str, Number]] | None = None,
             frozen: bool = False, planned: bool = False) -> Cell:
    """One matrix cell, derived from ``arm`` and the loaded numbers."""
    reg = f"ARMS['{arm.name}'].{column.field}"
    if column.kind == "tick":
        if column.field == "features":
            on = column.key in arm.features
        else:
            on = bool(getattr(arm, column.field))
        return Cell("", on, REGISTRY, reg, column.rule)
    if column.kind == "text":
        return Cell(str(getattr(arm, column.field)), None, REGISTRY, reg, column.rule)

    if frozen:
        # Not "n/a": the arm was not measured this phase, which is a scope
        # statement, not a missing value.
        return Cell(FROZEN, None, ARCHIVE, "-", "self-supervised arm is frozen")
    if planned:
        # Also not "n/a": the arm exists in code and has simply not been run on
        # real data. Drawing a number here would be the one thing this figure
        # is built to prevent.
        return Cell(NOT_RUN, None, REGISTRY, "-",
                    "PR2 arm implemented; real-data experiment NOT_RUN")
    num = (numbers or {}).get(arm.name, {}).get(column.field)
    return Cell(format_number(num, column.field), None,
                num.source if num else "no source",
                num.field if num else column.field, column.rule,
                note=num.note if num else "", alternates=num.alternates if num else ())


def matrix_row(arm: Arm, numbers: dict[str, dict[str, Number]],
               frozen: bool | None = None, planned: bool = False) -> Row:
    """One arm's row. ``frozen`` defaults to what the registry says it is."""
    if frozen is None:
        frozen = group_of(arm) == "frozen"
    cells = {c.key: cell_for(arm, c, numbers, frozen=frozen, planned=planned)
             for c in COLUMNS}
    return Row(arm=arm, group="planned" if planned else group_of(arm),
               cells=cells, frozen=frozen, planned=planned)


def matrix_rows(arms: dict[str, Arm] | None = None,
                numbers: dict[str, dict[str, Number]] | None = None
                ) -> list[Row]:
    """Every arm in the registry, in band then registry order.

    Refuses to draw in either direction of inconsistency: a supervised arm with
    no measured numbers (the figure would claim a comparison it cannot source),
    or a self-supervised arm that *does* have numbers (the figure would call a
    measured arm frozen).
    """
    arms = ARMS if arms is None else arms
    numbers = load_numbers() if numbers is None else numbers
    rows: list[Row] = []
    for arm in arms.values():
        if arm.family == "pr2" and arm.name not in numbers:
            # Implemented, unmeasured: its own band with NOT_RUN cells. The
            # moment a real number exists this branch stops firing and the arm
            # joins the measured band -- no list to maintain by hand.
            rows.append(matrix_row(arm, numbers, planned=True))
            continue
        frozen = group_of(arm) == "frozen"
        if frozen and arm.name in numbers:
            raise RuntimeError(
                f"{arm.name}: the registry says self-supervised (frozen) but "
                f"{NUM_SOURCES[0]} carries numbers for it; measure it or drop "
                "it from the source, do not draw it as frozen")
        if not frozen and arm.name not in numbers:
            raise KeyError(
                f"{arm.name}: no measured numbers in "
                f"{[str(s) for s in NUM_SOURCES]}; add the measurement or the "
                "figure cannot honestly show this arm")
        rows.append(matrix_row(arm, numbers, frozen=frozen))
    order = {key: i for i, (key, _) in enumerate(GROUPS)}
    return sorted(rows, key=lambda r: (order[r.group], list(arms).index(r.arm.name)))


def matrix_blocks(rows: Iterable[Row]) -> list[Block]:
    """Group rows under the band headings, keeping every row exactly once."""
    rows = list(rows)
    blocks: list[Block] = []
    for key, title in GROUPS:
        members = tuple(r for r in rows if r.group == key)
        if members:
            blocks.append(Block(key, title, members))
    placed = sum(len(b.rows) for b in blocks)
    if placed != len(rows):          # a row in an unknown band would vanish silently
        raise RuntimeError(f"{len(rows) - placed} matrix rows belong to no band")
    return blocks


# ---------------------------------------------------------------------------
# pipeline facts: also derived, never written in prose
# ---------------------------------------------------------------------------

def pipeline_facts(arms: dict[str, Arm] | None = None) -> dict:
    """Everything the pipeline panel asserts about the code, as data."""
    arms = ARMS if arms is None else arms

    def names(pred) -> list[str]:
        return [a.name for a in arms.values() if pred(a)]

    channels = {a.name: len(a.features) for a in arms.values() if a.needs_training}
    return {
        "registry": REGISTRY,
        "warm_start_modes": list(WARM_START_MODES),
        "warm_start_source": "spur_mpm/physics/wiener.py::WARM_START_MODES",
        "sampling_modes": list(MODES),
        "sampling_modes_source": "spur_mpm/physics/sampling.py::MODES",
        "arms_by_warm_start": {m: names(lambda a, m=m: a.warm_start == m)
                               for m in WARM_START_MODES},
        "x0_input_arms": names(lambda a: "x0" in a.features),
        "uncertainty_arms": names(lambda a: "uncertainty" in a.features),
        "fusion_arms": names(lambda a: a.fusion),
        "trunk_arms": names(lambda a: a.needs_training),
        "no_network_arms": names(lambda a: not a.needs_training),
        "frozen_arms": names(lambda a: a.supervision == "self"),
        "features_per_arm": channels,
        "channels_source": f"{REGISTRY}::Arm.features (len), "
                           "SwinUNet2D(in_ch = len(features) * context_depth)",
    }


def _channels_text(facts: dict) -> str:
    """``2 (S0), 3 (S1, S1g), 4 (S2, S2g)`` -- measured trained arms only."""
    buckets: dict[int, list[str]] = {}
    for name, n in facts["features_per_arm"].items():
        if name in facts["frozen_arms"]:
            continue
        buckets.setdefault(n, []).append(name)
    return ", ".join(f"{n} ({', '.join(v)})" for n, v in sorted(buckets.items()))


def _lattice_text(facts: dict) -> str:
    """``lattices: y | z | yz | xy | full`` -- quoted from ``sampling.MODES``."""
    return "lattices: " + " | ".join(facts["sampling_modes"])


def _mode_text(facts: dict) -> str:
    parts = []
    for mode, members in facts["arms_by_warm_start"].items():
        parts.append(f"{mode} -> {', '.join(members) if members else 'no arm'}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def provenance(rows: Sequence[Row], facts: dict,
               sources: Sequence[Path | str] = NUM_SOURCES) -> dict:
    """Machine-readable record of where each tick and each number came from."""
    return {
        "what": "Fig. 2 provenance: every tick and every number with the field "
                "it was derived from",
        "why": "the matrix is generated from ARMS so that editing the registry "
               "changes the figure; a hand-typed row could claim an experiment "
               "nobody ran",
        "registry": REGISTRY,
        "numbers_sources": [str(s) for s in sources],
        "numbers_precedence": "first non-null value per field, in source order; "
                              "disagreeing files are listed under 'alternates'",
        "columns": [{"key": c.key, "label": c.label.replace("\n", " "),
                     "kind": c.kind, "field": c.field, "rule": c.rule}
                    for c in COLUMNS],
        "groups": {key: title for key, title in GROUPS},
        "arms": {
            row.arm.name: {
                "group": row.group,
                "frozen": row.frozen,
                "cells": {
                    key: {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in {
                              "tick": c.tick, "text": c.text, "source": c.source,
                              "field": c.field, "rule": c.rule,
                              **({"note": c.note} if c.note else {}),
                              **({"alternates": c.alternates} if c.alternates else {}),
                          }.items()}
                    for key, c in row.cells.items()
                },
            }
            for row in rows
        },
        "pipeline": facts,
    }


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def _box(ax, x0: float, y0: float, w: float, h: float, title: str, sub: str = "",
         optional: bool = False) -> None:
    edge = OPT if optional else INK
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=0.9, linestyle="--" if optional else "-",
        edgecolor=edge, facecolor="#f2f6fb" if not optional else "#fdf9f2"))
    ax.text(x0 + w / 2, y0 + h - 1.1, title, fontsize=7.8, ha="center", va="top",
            color=edge, weight="bold")
    if sub:
        ax.text(x0 + w / 2, y0 + h - 4.0, sub, fontsize=5.9, ha="center", va="top",
                color="#2b2b2b", linespacing=1.4)
    if optional:
        ax.text(x0 + 1.0, y0 + 0.9, "optional", fontsize=5.4, ha="left", va="bottom",
                color=edge, style="italic")


def _arrow(ax, p0, p1, label: str = "", dashed: bool = False, color: str = INK,
           lw: float = 0.9, rad: float = 0.0, size: float = 7.0) -> None:
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=size, linewidth=lw, color=color,
        shrinkA=0, shrinkB=0, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.7, label, fontsize=5.6,
                ha="center", va="bottom", color=color)


# ---------------------------------------------------------------------------
# panel (a): pipeline
# ---------------------------------------------------------------------------

def render_pipeline(facts: dict, path: Path, dpi: int = 200) -> Path:
    """The forward path, with U and the gate drawn as the optional blocks they are."""
    fig = plt.figure(figsize=(14.0, 7.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 140)
    ax.set_ylim(12, 86)     # 10 data units per inch on both axes: isotropic
    ax.axis("off")

    ax.text(1.5, 85.0, "Fig. 2a  Method: sparse observation -> physics warm start "
            "-> shared Swin-UNet", fontsize=10.5, ha="left", va="top", color="black")
    ax.text(1.5, 82.8, "Blocks, mode strings and channel counts are read from the "
            "code (paths in each box); ticks and numbers on panel (b) come from "
            f"{facts['registry']} and the complexity benchmark.",
            fontsize=6.2, ha="left", va="top", color="#333333")

    # ---- the forward chain ----
    top, h, w, gap = 78.0, 14.0, 21.0, 2.0
    y0 = top - h
    xs = [1.5 + i * (w + gap) for i in range(6)]
    chain = [
        ("D  dense reference", "[Z, H, W]\nloss target only --\nnever an input feature"),
        ("Virtual acquisition", "observe():  Y = M * (H * D) + n\n"
                               "H = identity in image domain A\nphysics/acquisition.py"),
        ("Sparse observation",
         "Y zero-filled,  M in {0, 1}\n" + _lattice_text(facts)
         + "\nphysics/sampling.py"),
        ("Feature stack", "concat(Y, M [, X0] [, U])\n"
                          "[F * t, H, W], feature-major\n"
                          "F = len(arm.features), t = context_depth\n"
                          f"F = {_channels_text(facts)}"),
        ("Shared Swin-UNet-2D", "stem Conv2d -> 3 stages -> head\n"
                                "window 8, base 48, depths 2/2/2\nheads 4/8/16\n"
                                "one carrier for every trained arm"),
        ("x_hat", "[t, H, W] -> centre plane\n"
                  "loss: dense target (supervised)\n"
                  "held-out measurements (self-sup.)"),
    ]
    for i, (title, sub) in enumerate(chain):
        _box(ax, xs[i], y0, w, h, title, sub)
    # No labels between chain boxes: the 2-unit gaps cannot hold text at this
    # font size, and the subtitles already name the hand-off.
    for i in range(5):
        _arrow(ax, (xs[i] + w, y0 + h / 2), (xs[i + 1], y0 + h / 2))

    # ---- feeders: the prior and the two optional blocks ----
    fy, fh = 42.0, 16.0
    x0_box = (20.0, 62.0)
    u_box = (66.0, 102.0)
    gate_box = (106.0, 138.5)
    # One mode per line: the joined "mode -> arms" string does not fit a box.
    mode_lines = "\n".join(f"{m} -> {', '.join(a) if a else 'no arm'}"
                           for m, a in facts["arms_by_warm_start"].items())
    _box(ax, x0_box[0], fy, x0_box[1] - x0_box[0], fh,
         "Physics warm start  X0 = W(Y, M; H)",
         "physics/wiener.py::warm_start\n"
         "modes: " + " | ".join(facts["warm_start_modes"]) + "\n"
         + mode_lines + "\n"
         "identity PSF here -> the Wiener branches reduce to gap filling "
         "(A1 == A0)")
    _box(ax, u_box[0], fy, u_box[1] - u_box[0], fh,
         "Sampling-conditioned fields  U, C_s",
         "models/uncertainty.py\n"
         "C_s = G_sigma * M   (support, confidence)\n"
         "U_geo: distance to the nearest sample\n"
         "U_phys = 1 - C_p from the OTF\n"
         "none of them is a calibrated error",
         optional=True)
    _box(ax, gate_box[0], fy, gate_box[1] - gate_box[0], fh,
         "PriorFusionStem  (gate)",
         "models/prior_fusion.py\n"
         "F_in = F_Y + G * F_W\n"
         "G = sigmoid(Conv([F_S, F_W]))\n"
         "replaces the trunk's first conv, so the\n"
         "backbone width stays at base",
         optional=True)

    # Y,M feeds both priors; X0 and U feed the stack; the gate feeds the trunk.
    # The vertical space between the feeder band and the chain (64 -> 58) is the
    # only place a label fits, so the labels stay short and sit in that gap.
    _arrow(ax, (55.0, y0), (44.0, fy + fh), rad=-0.10, label="Y, M")
    _arrow(ax, (63.0, y0), (72.0, fy + fh), rad=0.0, dashed=True, color=OPT,
           label="M")
    _arrow(ax, (56.0, fy + fh), (78.0, y0), rad=0.10, label="'x0'")
    _arrow(ax, (88.0, fy + fh), (87.0, y0), label="'uncertainty'", dashed=True,
           color=OPT)
    _arrow(ax, (112.0, y0), (122.0, fy + fh), rad=-0.05, dashed=True, color=OPT,
           label="F_Y, F_W")
    _arrow(ax, (130.0, fy + fh), (116.0, y0), rad=-0.05, label="arm.fusion",
           dashed=True, color=OPT)

    # ---- registry-derived bookkeeping strip ----
    lines = [
        ("optional blocks and the arms that switch them on  (derived from "
         f"{facts['registry']})", True),
        ("X0 as an input feature:  " + " ".join(facts["x0_input_arms"]), False),
        ("U (uncertainty) feature:  " + " ".join(facts["uncertainty_arms"]), False),
        ("gated prior fusion:  " + " ".join(facts["fusion_arms"]), False),
        ("no network (no trunk, no gate, no U):  " +
         " ".join(facts["no_network_arms"]) +
         "  -- their prediction is X0 itself", False),
    ]
    ty = 39.0
    for text, head in lines:
        ax.text(1.5, ty, text, fontsize=6.4 if head else 6.2, ha="left", va="top",
                color="black" if head else "#2b2b2b", weight="bold" if head else "normal")
        ty -= 2.4
    ax.text(1.5, ty - 0.4, "frozen 2026-10-30 = self-supervised arm "
            f"({ARCHIVE}); its ticks are drawn in the matrix in grey, and it is "
            "not measured in Phase 1",
            fontsize=6.2, ha="left", va="top", color=GREY)

    ax.text(1.5, 22.0,
            "Schematic. No quantity on this panel is measured: it is the code's "
            "data path, and every name on it is imported or counted, not typed.",
            fontsize=6.2, ha="left", va="top", color="#555555")
    ax.text(1.5, 19.0,
            "Two facts a reader should not have to look up: with the identity PSF "
            "the warm start is interpolation, so A1 reproduces A0 exactly "
            "(physics/wiener.py); and the gated stem keeps the input width at "
            "base, so 'fusion helps' cannot be confused with 'more parameters help' "
            "(models/prior_fusion.py).",
            fontsize=6.2, ha="left", va="top", color="#555555")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# panel (b): ablation matrix
# ---------------------------------------------------------------------------

_WIDTH = {"arm": 1.55, "y": 0.55, "mask": 0.55, "x0": 0.62, "uncertainty": 0.55,
          "fusion": 0.62, "trunk": 0.78, "warm_start": 1.5, "supervision": 0.9,
          "params": 0.8, "macs": 0.95, "latency": 0.8}

_NOTES = (
    "Every tick is computed from Arm.features / .fusion / .needs_training / "
    ".warm_start / .supervision in " + REGISTRY + ": a registry edit changes this "
    "matrix. Band membership is derived the same way (Arm.supervision, "
    "Arm.needs_training); the bands themselves are a schematic grouping.",
    "Numbers are read from runs/E021_ladder/benchmark_supervised.json (fallback "
    "runs/E021_ladder/complexity.json), patch 256, n_slices 200, context_depth 1, "
    "cuda:2. n/a = null in the source, not zero. Where the two files disagree the "
    "provenance JSON lists both values.",
    "A0/A1 have no network: params 0 by construction (source note) and MACs are "
    "not applicable; their ms/slice is the measured warm-start operator on the "
    "same [200, 256, 256] volume, not a forward pass.",
    "A1's own feature list is (y, mask), so its X0 tick is off: X0 is its output, "
    "not an input channel. Parameter-free arms are scored on BASELINE_FEATURES = "
    "(y, mask, x0) (registry.py).",
    "Self-supervised arms keep their design ticks but print 'frozen' where a "
    "measurement would be: they were archived on 2026-10-30 (" + ARCHIVE + ") and "
    "are out of Phase-1 scope, which is not the same as a failed result.",
)


def render_matrix(rows: Sequence[Row], path: Path, dpi: int = 200) -> Path:
    """Rows = arms, columns = the components and the cost of turning them on."""
    blocks = matrix_blocks(rows)
    margin = 0.22
    total_w = sum(_WIDTH.values()) + 2 * margin
    row_h, band_h, head_h = 0.30, 0.32, 0.72
    foot_h = 0.235 * len(_NOTES) + 0.12
    axe_h = 0.30 * sum(len(b.rows) + (1 if b.key != "frozen" else 1.2) for b in blocks)
    total_h = 0.62 + head_h + axe_h * 0.9 + foot_h + 0.18

    fig = plt.figure(figsize=(total_w, total_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)          # 1 data unit = 1 inch on both axes
    ax.axis("off")

    ax.text(margin, total_h - 0.14,
            "Fig. 2b  Ablation matrix: what each arm turns on, and what it costs",
            fontsize=10.0, ha="left", va="top", color="black")
    ax.text(margin, total_h - 0.42,
            "tick = the arm's declared input or module;  -  = not used;  grey = "
            "frozen self-supervised arm (design shown, not a Phase-1 measurement)",
            fontsize=6.2, ha="left", va="top", color="#333333")

    # column x positions
    xs: dict[str, float] = {}
    x = margin
    for key in _WIDTH:
        xs[key] = x
        x += _WIDTH[key]

    def centre(key: str) -> float:
        return xs[key] + _WIDTH[key] / 2

    y = total_h - 0.62
    for col in COLUMNS:
        ax.text(centre(col.key), y - 0.04, col.label, fontsize=6.3, ha="center",
                va="top", color="black", weight="bold", linespacing=1.3)
    ax.text(xs["arm"] + 0.05, y - 0.04, "arm", fontsize=6.3, ha="left", va="top",
            color="black", weight="bold")
    y -= head_h
    ax.plot([margin, total_w - margin], [y, y], color=INK, lw=0.8)

    for bi, block in enumerate(blocks):
        band_colour = {"floors": "#eef1f5", "supervised": "#e7eefa",
                       "planned": "#eef2f9", "frozen": "#f4f4f4"}[block.key]
        strip = {"floors": INK_2, "supervised": INK, "frozen": GREY,
                 "planned": INK_3}[block.key]
        band_hh = band_h * (1.15 if block.key in ("frozen", "planned") else 1.0)
        ax.add_patch(Rectangle((margin, y - band_hh), total_w - 2 * margin, band_hh,
                               facecolor=band_colour, edgecolor="none"))
        ax.add_patch(Rectangle((margin, y - band_hh), 0.05, band_hh,
                               facecolor=strip, edgecolor="none"))
        ax.text(margin + 0.14, y - band_hh / 2, block.title, fontsize=6.5,
                ha="left", va="center", color=strip, weight="bold")
        y -= band_hh

        for ri, row in enumerate(block.rows):
            if ri % 2 == 1:
                ax.add_patch(Rectangle((margin, y - row_h), total_w - 2 * margin,
                                       row_h, facecolor="#fafbfd", edgecolor="none"))
            ink = GREY if row.frozen else INK
            ax.text(xs["arm"] + 0.14, y - row_h / 2, row.arm.name, fontsize=7.2,
                    ha="left", va="center", color=ink,
                    weight="normal" if row.frozen else "bold")
            if row.arm.dual:
                ax.text(xs["arm"] + 0.14 + 0.52, y - row_h / 2, "dual", fontsize=5.2,
                        ha="left", va="center", color=ink, style="italic")
            for col in COLUMNS:
                cell = row.cell(col.key)
                cy = y - row_h / 2
                if col.kind == "tick":
                    if cell.tick:
                        ax.plot(centre(col.key), cy, marker="o", ms=3.6,
                                mfc=ink, mec=ink, lw=0)
                    else:
                        ax.plot([centre(col.key) - 0.06, centre(col.key) + 0.06],
                                [cy, cy], color="#c8ccd2", lw=0.9)
                elif col.kind == "text":
                    ax.text(centre(col.key), cy, cell.text, fontsize=6.0,
                            ha="center", va="center", color=ink)
                else:
                    style = dict(fontsize=5.9, ha="center", va="center")
                    if cell.text in (NA, FROZEN):
                        ax.text(centre(col.key), cy, cell.text,
                                color="#9a9a9a" if cell.text == FROZEN else "#b0b0b0",
                                style="italic", **style)
                    else:
                        ax.text(centre(col.key), cy, cell.text, color=ink, **style)
            y -= row_h
        ax.plot([margin, total_w - margin], [y, y], color=RULE, lw=0.8)
        if bi == len(blocks) - 1:
            ax.plot([margin, total_w - margin], [y, y], color=INK, lw=0.8)

    ny = y - 0.16
    for note in _NOTES:
        ax.text(margin, ny, _wrap(note, 196), fontsize=5.5, ha="left", va="top",
                color="#444444", linespacing=1.35)
        ny -= 0.235 * (1 + len(_wrap(note, 196).split("\n")) - 1) + 0.045

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def _wrap(text: str, width: int) -> str:
    """Hard-wrap for the footnote block (matplotlib has no paragraph layout)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def render(out_dir: Path | str, sources: Sequence[Path | str] = NUM_SOURCES,
           dpi: int = 200) -> list[Path]:
    """Write both panels and the provenance JSON; return the files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = matrix_rows(numbers=load_numbers(sources))
    facts = pipeline_facts()
    written = [
        render_pipeline(facts, out_dir / "fig2_pipeline.png", dpi=dpi),
        render_matrix(rows, out_dir / "fig2_ablation_matrix.png", dpi=dpi),
    ]
    prov = out_dir / "fig2_provenance.json"
    prov.write_text(json.dumps(provenance(rows, facts, sources), indent=1) + "\n",
                    encoding="utf-8")
    written.append(prov)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", type=Path, default=BENCHMARK,
                   help="measured numbers (primary source)")
    p.add_argument("--complexity", type=Path, default=COMPLEXITY,
                   help="measured numbers (fallback / cross-check)")
    p.add_argument("--out", type=Path, default=Path("spur_mpm/figures/framework"))
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)
    for path in render(args.out, sources=(args.benchmark, args.complexity),
                       dpi=args.dpi):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
