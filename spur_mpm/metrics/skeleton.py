"""Skeleton topology and vascular morphometry layered on ``metrics/vessel.py``.

Dice and PSNR both answer "how much": how much volume overlaps, how close the
grey levels are. Neither answers the question that decides whether a
reconstruction of microvasculature is usable -- does the *network* survive
thinning? A prediction can thicken every capillary and score a respectable Dice
while merging two vessels into one, and it can bridge a gap and raise Dice while
filling in a bifurcation. Both failures are invisible to intensity metrics and
obvious the moment the mask is skeletonised, which is exactly the check an
anatomist makes by hand in ImageJ (skeletonise, then measure length and
branching).

The mask itself is deliberately **not** redefined here:
:func:`spur_mpm.metrics.vessel.vessel_mask` owns the Otsu + hysteresis decision,
so every number below is measured on the same binary object for every model and
no threshold is ever fitted on a prediction. This module only measures what is
left after thinning -- chain length, branches, junctions, endpoints, radius, and
how those quantities vary with depth.

These are summary statistics over branches, so a single plane is noisy; they are
meant to be aggregated per group (``metrics/statistics.py``) and reported next
to, never instead of, the intensity metrics. Functions that cannot honestly
compute a number return NaN rather than a plausible placeholder, and every such
case is documented at the function that does it.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as _label
from skimage.morphology import skeletonize as _skeletonize

from spur_mpm.metrics.vessel import (mask_vessel_metrics, segment_pair,
                                      vessel_mask)

__all__ = [
    "skeleton",
    "skeleton_length_vox",
    "skeleton_length_density",
    "skeleton_stats",
    "skeleton_overlap",
    "depth_profile",
    "analyse_volume",
    "export_imagej_mask",
]

def _voxel_scales(voxel_um, ndim: int) -> tuple[float, ...]:
    """Per-axis voxel size for ``ndim`` axes; a 3-vector drops its z for 2-D."""
    v = tuple(float(x) for x in np.atleast_1d(voxel_um))
    if len(v) == ndim:
        out = v
    elif len(v) == 3 and ndim == 2:              # single plane: no axial extent
        out = v[1:]
    elif len(v) == 1:
        out = v * ndim
    else:
        raise ValueError(f"voxel_um={voxel_um!r} does not describe a {ndim}-D mask")
    if min(out) <= 0:
        raise ValueError(f"voxel_um must be positive, got {out}")
    return out


def _bbox_slices(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Tight index box around the mask, or ``None`` when it is empty."""
    if not mask.any():
        return None
    box = []
    for axis in range(mask.ndim):
        others = tuple(a for a in range(mask.ndim) if a != axis)
        idx = np.flatnonzero(mask.any(axis=others))
        box.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(box)


def _bbox_volume_um3(mask: np.ndarray, voxel_um) -> float:
    """Volume of the mask's tight bounding box in um^3.

    A 2-D mask has no axial extent, so its box is given one plane of thickness
    (``voxel_um[0]``). That keeps the density below in cubic-millimetre units
    for a single plane as well.
    """
    box = _bbox_slices(mask)
    if box is None:
        return 0.0
    if mask.ndim == 2:
        scales = _voxel_scales(voxel_um, 2)
        volume = float(np.atleast_1d(voxel_um)[0])
    else:
        scales = _voxel_scales(voxel_um, 3)
        volume = 1.0
    for sl, step_um in zip(box, scales):
        volume *= (sl.stop - sl.start) * step_um
    return float(volume)


def _offset_kernel(scales: tuple[float, ...]) -> np.ndarray:
    """Neighbourhood kernel holding the Euclidean length of every step.

    All ``3**ndim - 1`` offsets are enumerated, so the three distinct step
    lengths of a 3-D grid (1, sqrt2, sqrt3) fall out of the table instead of
    being hard-coded; supplying ``scales`` in micrometres gives the same table in
    physical units, which is what makes the length correct for anisotropic
    voxels.
    """
    kernel = np.zeros((3,) * len(scales), dtype=np.float64)
    for offset in itertools.product((-1, 0, 1), repeat=len(scales)):
        if any(offset):
            step = sum((o * s) ** 2 for o, s in zip(offset, scales))
            kernel[tuple(o + 1 for o in offset)] = float(np.sqrt(step))
    return kernel


def _forward_offsets(ndim: int) -> list[tuple[int, ...]]:
    """Half of the 26 (or 8) neighbour offsets: each voxel pair once.

    A pair is visited from exactly one end -- the end whose offset is positive in
    its first non-zero axis -- so no edge is counted twice.
    """
    offsets = []
    for off in itertools.product((-1, 0, 1), repeat=ndim):
        if not any(off):
            continue
        first = next(o for o in off if o)
        if first > 0:
            offsets.append(off)
    return offsets


def _chain_length(thin: np.ndarray, scales: tuple[float, ...] | None = None) -> float:
    """Length of a one-voxel-wide structure, as the spanning tree of its voxels.

    The obvious estimator -- half the summed distance to every neighbour -- is
    correct only for a graph where each voxel has exactly two neighbours. A
    thinned skeleton does not satisfy that: wherever the vessel runs obliquely,
    the neighbouring voxels are also diagonally adjacent to each other, so the
    "shortcut" edges are counted as well. In 2-D that inflates a 45-degree
    staircase by ~6 %; in 3-D a diagonal run has up to 13 near-neighbours per
    voxel and the same formula inflated a real 200-plane stack to **45 km of
    vessel** in a 512 x 512 x 202 volume.

    The length used here is the total edge weight of the **minimum spanning tree**
    of the voxel adjacency graph, which is the geodesic length of the structure:
    for a simple path it is exactly the chain length, and where the path is
    ambiguous (a diagonal step can be taken either way) it takes the shorter one
    rather than adding both. Edges are restricted to the 26-neighbourhood, so a
    pair two voxels apart is never joined, and weights are in ``scales`` units so
    anisotropic voxels are handled.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    m = np.asarray(thin, dtype=bool)
    if not m.any():
        return 0.0
    if scales is None:
        scales = (1.0,) * m.ndim
    shape = np.asarray(m.shape)
    coords = np.argwhere(m)
    n = len(coords)
    if n < 2:
        return 0.0
    flat = np.ravel_multi_index(coords.T, m.shape)
    order = np.argsort(flat)
    flat_sorted = flat[order]

    srcs, dsts, weights = [], [], []
    for off in _forward_offsets(m.ndim):
        nb = coords + np.asarray(off)
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        nb_flat = np.ravel_multi_index(nb[idx].T, m.shape)
        pos = np.searchsorted(flat_sorted, nb_flat)
        np.clip(pos, 0, n - 1, out=pos)
        hit = flat_sorted[pos] == nb_flat
        if not hit.any():
            continue
        srcs.append(order[pos[hit]])
        dsts.append(idx[hit])
        weights.append(np.full(int(hit.sum()),
                               float(np.sqrt(sum((o * s) ** 2
                                                 for o, s in zip(off, scales))))))
    if not srcs:
        return 0.0
    graph = coo_matrix((np.concatenate(weights),
                        (np.concatenate(srcs), np.concatenate(dsts))),
                       shape=(n, n)).tocsr()
    tree = minimum_spanning_tree(graph)
    return float(tree.sum())


def _neighbour_counts(thin: np.ndarray) -> np.ndarray:
    """Number of skeleton neighbours per voxel (8-connected in 2-D, 26 in 3-D)."""
    ones = np.ones((3,) * thin.ndim, dtype=np.uint8)
    counts = ndi.convolve(thin.astype(np.uint8), ones, mode="constant", cval=0)
    return counts.astype(np.int16) - 1           # drop the voxel's own weight


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Thin a binary vessel mask to a one-voxel-wide skeleton.

    Non-zero is foreground. The result is 26-connected and centred inside the
    original structure, so measuring it describes the vessel axis rather than
    its wall -- which is what makes length and branch counts comparable to a
    hand measurement in ImageJ.
    """
    m = np.asarray(mask)
    if not m.any():
        return np.zeros(m.shape, dtype=bool)
    return np.asarray(_skeletonize(m.astype(bool)), dtype=bool)


def skeleton_length_vox(mask: np.ndarray) -> float:
    """Chain length of the mask's skeleton, in voxels.

    Counting voxels would charge one unit for a diagonal step and so inflate
    any vessel that is not axis-aligned. The length here is the sum, over
    skeleton voxels, of half the Euclidean step to each skeleton neighbour:
    an axis-aligned run of N voxels is exactly N-1, a diagonal step counts
    sqrt2 and a 3-D corner sqrt3.
    """
    return _chain_length(skeleton(mask))


def skeleton_length_density(mask: np.ndarray,
                            voxel_um: tuple[float, float, float] = (5.0, 0.98, 0.98)) -> float:
    """Skeleton length in mm per mm^3 of the mask's own bounding box.

    The normaliser is the tight bounding box of the mask, not the field of view,
    so a sparse structure in a mostly empty image is not scored as if it filled
    the volume. For a single 2-D plane the box is one voxel thick
    (``voxel_um[0]``), and the value is then a length per area in the same
    mm^-2 unit -- but it is *not* numerically comparable to a 3-D density: only
    ratios between masks measured with the same ``voxel_um`` are, which is what
    ``length_density_ratio`` in :func:`analyse_volume` reports.

    An empty mask has no bounding box and returns 0.0.
    """
    m = np.asarray(mask, dtype=bool)
    thin = skeleton(m)
    if not thin.any():
        return 0.0
    length_mm = _chain_length(thin, _voxel_scales(voxel_um, m.ndim)) / 1000.0
    volume_mm3 = _bbox_volume_um3(m, voxel_um) / 1e9
    if volume_mm3 <= 0:                          # unreachable for a non-empty mask
        return 0.0
    return float(length_mm / volume_mm3)


def _shift_index(shape: tuple[int, ...], offset: tuple[int, ...]) -> np.ndarray:
    """Flat index of the voxel at ``x + offset`` for every ``x``; -1 outside.

    Letting numpy carry the translation means every neighbour pair in the volume
    is enumerated in a handful of vectorised passes instead of per-object Python
    loops -- the difference between seconds and minutes on a 200-plane stack.
    """
    size = int(np.prod(shape))
    dtype = np.int32 if size < 2 ** 31 - 1 else np.int64
    flat = np.arange(size, dtype=dtype).reshape(shape)
    out = np.full(size, -1, dtype=dtype)
    dst, src = [], []
    for o, n in zip(offset, shape):
        if o >= 0:
            dst.append(slice(0, max(n - o, 0)))
            src.append(slice(o, n))
        else:
            dst.append(slice(-o, n))
            src.append(slice(0, n + o))
    out.reshape(shape)[tuple(dst)] = flat[tuple(src)]
    return out


def _region_lengths(region: np.ndarray,
                    scales: tuple[float, ...] | None = None) -> list[float]:
    """Length of every connected component of ``region``, one entry each.

    Every edge of the voxel adjacency graph is enumerated once (forward offsets
    only) and attributed to the component both its ends belong to. For a
    component whose edges form a tree -- which is what a one-voxel-wide branch is
    -- the sum of its edge lengths *is* its length, so no per-branch graph has to
    be built. Only a component that contains a cycle (a closed loop of vessel)
    needs the spanning-tree correction, and those are rare, so they are the only
    ones measured individually.
    """
    reg = np.asarray(region, dtype=bool)
    if not reg.any():
        return []
    if scales is None:
        scales = (1.0,) * reg.ndim
    # Crop first: the neighbour enumeration allocates a volume-sized index array
    # once per offset, so working on the occupied box rather than the padded
    # field is the difference between a 400 MB pass and a few megabytes.
    box = _bbox_slices(reg)
    if box is not None:
        reg = reg[box]
    labels = _label(reg, connectivity=reg.ndim)
    flat_labels = labels.ravel()
    n_labels = int(labels.max())
    total_per_label = np.zeros(n_labels + 1, dtype=np.float64)
    edges_per_label = np.zeros(n_labels + 1, dtype=np.int64)
    voxels_per_label = np.bincount(flat_labels, minlength=n_labels + 1)
    start = np.flatnonzero(reg.ravel())

    for off in _forward_offsets(reg.ndim):
        step = float(np.sqrt(sum((o * s) ** 2 for o, s in zip(off, scales))))
        j = _shift_index(reg.shape, off)[start]
        ok = j >= 0
        if not ok.any():
            continue
        a, b = start[ok], j[ok]
        la, lb = flat_labels[a], flat_labels[b]
        same = la == lb
        total_per_label += step * np.bincount(la[same], minlength=n_labels + 1)
        edges_per_label += np.bincount(la[same], minlength=n_labels + 1)

    lengths: list[float] = []
    boxes = ndi.find_objects(labels)
    for index in range(1, n_labels + 1):
        e, v = int(edges_per_label[index]), int(voxels_per_label[index])
        if e == 0:
            lengths.append(0.0)
        elif e <= v - 1:                     # acyclic: the edge sum is the length
            lengths.append(float(total_per_label[index]))
        else:                                # a loop: build the graph properly
            box = boxes[index - 1]
            lengths.append(_chain_length(labels[box] == index, scales))
    return lengths


def _connector_length(thin: np.ndarray, junctions: np.ndarray,
                      scales: tuple[float, ...] | None = None) -> float:
    """Length of the edges cut by taking the junctions out of the skeleton.

    Removing the junction voxels partitions the skeleton into branches, but it
    also deletes the edges that *join* a junction to a branch. Ignoring them
    shortens the total: measured on a real stack it fell 22.97 -> 19.14, exactly
    the ``1 + 2*sqrt(2)`` of the connectors that vanished. Each junction-to-branch
    edge is visited once here, so the three parts add up to the whole skeleton.
    """
    thin = np.asarray(thin, dtype=bool)
    if not (thin.any() and junctions.any()):
        return 0.0
    if scales is None:
        scales = (1.0,) * thin.ndim
    box = _bbox_slices(thin)
    if box is not None:
        thin = thin[box]
        junctions = np.asarray(junctions, dtype=bool)[box]
    junction_flat = junctions.ravel()
    thin_flat = thin.ravel()
    start = np.flatnonzero(thin_flat)
    total = 0.0
    for off in _forward_offsets(thin.ndim):
        step = float(np.sqrt(sum((o * s) ** 2 for o, s in zip(off, scales))))
        b = _shift_index(thin.shape, off)[start]
        ok = b >= 0
        a, b = start[ok], b[ok]
        inside = thin_flat[b]
        a, b = a[inside], b[inside]
        # Each unordered pair is enumerated exactly once (forward offsets only),
        # so "exactly one end is a junction" counts every connector once and no
        # junction-junction or branch-branch edge. Orientation plays no part: the
        # pair may show up as junction->branch or as branch->junction.
        keep = junction_flat[a] != junction_flat[b]
        total += step * float(keep.sum())
    return total


def _branch_lengths(thin: np.ndarray, junctions: np.ndarray,
                    scales: tuple[float, ...] | None = None) -> list[float]:
    """Lengths of the segments left once the junction voxels are removed."""
    return _region_lengths(thin & ~junctions, scales)


def skeleton_stats(mask: np.ndarray, voxel_um=None) -> dict:
    """Topology and morphometry of the skeleton of ``mask``.

    Returns
    -------
    n_voxels : foreground voxels of the **mask** (not of the skeleton), so that
        ``bounding_box_fraction`` is a ratio of two mask volumes.
    n_skeleton_voxels : voxels left after thinning.
    n_components : skeleton components, 26-connected in 3-D (8 in 2-D).
    n_junctions : **junctions**, not junction voxels. Voxels with >= 3 skeleton
        neighbours are clustered first, because in 3-D an oblique vessel makes a
        whole band of voxels qualify -- an unreduced count said 2.3 million
        junctions in a single 512 x 512 x 202 stack, i.e. 96 % of the skeleton.
        One cluster is one bifurcation, which is what ImageJ's Analyze Skeleton
        reports after its own reduction pass.
    n_endpoints : skeleton voxels with exactly 1 skeleton neighbour.
    n_branches, total_length_vox, mean/median/max_branch_length_vox : the
        skeleton is cut at its junctions and each remaining segment is measured
        on its own, so a mean branch length says how far apart bifurcations are.
        Lengths are spanning-tree lengths (see ``_chain_length``), so a segment
        is never charged for the diagonal shortcuts its own voxels happen to
        form.
    mean_radius_vox : mean Euclidean distance transform along the skeleton, i.e.
        the half-width of the mask in voxels.
    mean_radius_um : the same converted with ``voxel_um``. NaN when ``voxel_um``
        is None -- an uncalibrated stack must not silently be reported as if its
        voxels were 1 um.
    total_length_um : physical length; NaN when ``voxel_um`` is None.
    bounding_box_fraction : mask volume / volume of its tight bounding box.

    An empty mask gives zero for every count and every length instead of
    raising, because "this plane contains no vessel" is a legitimate and common
    answer that must aggregate with the others rather than abort a run.
    """
    m = np.asarray(mask, dtype=bool)
    thin = skeleton(m)
    n_skeleton = int(thin.sum())
    counts = _neighbour_counts(thin) if n_skeleton else np.zeros(thin.shape, np.int16)
    junction_voxels = thin & (counts >= 3)
    # Cluster the junction voxels: one cluster = one bifurcation. Counting raw
    # voxels makes the number an artefact of how obliquely the vessel runs.
    junction_clusters = (_label(junction_voxels, connectivity=thin.ndim)
                         if junction_voxels.any() else None)
    n_junctions = int(junction_clusters.max()) if junction_clusters is not None else 0
    branch_lengths = _branch_lengths(thin, junction_voxels) if n_skeleton else []
    junction_lengths = _region_lengths(junction_voxels) if n_junctions else []

    box = _bbox_slices(m)
    box_voxels = (int(np.prod([sl.stop - sl.start for sl in box]))
                  if box is not None else 0)

    mean_radius_vox = 0.0
    if n_skeleton:
        edt = ndi.distance_transform_edt(m)
        mean_radius_vox = float(edt[thin].mean())

    if voxel_um is None:
        mean_radius_um = float("nan")
        total_length_um = float("nan")
    else:
        scales = _voxel_scales(voxel_um, m.ndim)
        # Physical radius must come from an anisotropic distance transform.
        # Scaling an isotropic voxel radius by the geometric-mean pitch is not
        # geometrically valid when Z=5 um and XY=0.98 um: even a purely lateral
        # tube would get the wrong physical radius.
        mean_radius_um = (float(ndi.distance_transform_edt(m, sampling=scales)[thin].mean())
                          if n_skeleton else 0.0)
        # Same partition, measured in micrometres so anisotropy is honoured.
        total_length_um = float(
            sum(_branch_lengths(thin, junction_voxels, scales))
            + sum(_region_lengths(junction_voxels, scales))
            + _connector_length(thin, junction_voxels, scales)
            if n_skeleton else 0.0)

    return {
        "n_voxels": int(m.sum()),
        "n_skeleton_voxels": n_skeleton,
        "n_components": int(_label(thin, connectivity=thin.ndim).max()) if n_skeleton else 0,
        "n_junctions": n_junctions,
        "n_endpoints": int((thin & (counts == 1)).sum()),
        "n_branches": len(branch_lengths),
        # Segments + junction clusters + the edges between them: the skeleton is
        # partitioned, every part stays a small graph, and nothing is dropped.
        "total_length_vox": float(sum(branch_lengths) + sum(junction_lengths)
                                   + _connector_length(thin, junction_voxels)),
        "mean_branch_length_vox": float(np.mean(branch_lengths)) if branch_lengths else 0.0,
        "median_branch_length_vox": float(np.median(branch_lengths)) if branch_lengths else 0.0,
        "max_branch_length_vox": float(np.max(branch_lengths)) if branch_lengths else 0.0,
        "mean_radius_vox": mean_radius_vox,
        "mean_radius_um": mean_radius_um,
        "total_length_um": total_length_um,
        "bounding_box_fraction": float(m.sum() / box_voxels) if box_voxels else 0.0,
    }


def _matched_fraction(a: np.ndarray, b: np.ndarray, tolerance_vox: float) -> float:
    """Fraction of ``a``'s skeleton voxels within ``tolerance_vox`` of ``b``'s."""
    if not a.any():
        return 1.0 if not b.any() else 0.0
    if not b.any():
        return 0.0
    distance = ndi.distance_transform_edt(~b)
    return float((distance[a] <= float(tolerance_vox) + 1e-9).mean())


def skeleton_overlap(pred_mask: np.ndarray, ref_mask: np.ndarray,
                     tolerance_vox: float = 1) -> dict:
    """Skeleton-pixel agreement within a Euclidean tolerance, in voxels.

    Exact skeleton coordinates are not a fair criterion under sparse sampling: a
    reconstruction that traces the same vessel half a voxel to the side would be
    penalised on every pixel. ``tolerance_vox`` accepts a match within that
    Euclidean distance, measured with the distance transform of the other
    skeleton, so ``tolerance_vox=0`` is strict coincidence while 1 forgives a
    one-voxel misplacement in any direction.

    Two empty skeletons score 1.0 on all three keys (they agree that there is
    nothing to trace; same convention as :func:`spur_mpm.metrics.vessel.dice`),
    one empty skeleton against a non-empty one scores 0.0.
    """
    sp, sr = skeleton(pred_mask), skeleton(ref_mask)
    precision = _matched_fraction(sp, sr, tolerance_vox)
    recall = _matched_fraction(sr, sp, tolerance_vox)
    total = precision + recall
    f1 = float(2 * precision * recall / total) if total > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _mask_metrics(pred_mask: np.ndarray, ref_mask: np.ndarray, voxel_um,
                  tolerance_vox: float) -> dict:
    """Every per-mask number, computed from two already-segmented masks.

    The overlap numbers come from :func:`mask_vessel_metrics`, the mask-level
    sibling of ``vessel_metrics``, so a single slab and a whole volume are scored
    by identical code and their numbers can be compared.
    """
    out = mask_vessel_metrics(pred_mask, ref_mask, tolerance_vox=tolerance_vox)
    for prefix, mask in (("pred_", pred_mask), ("ref_", ref_mask)):
        for key, value in skeleton_stats(mask, voxel_um=voxel_um).items():
            out[f"{prefix}{key}"] = value
    out.update(skeleton_overlap(pred_mask, ref_mask, tolerance_vox=tolerance_vox))

    pred_density = skeleton_length_density(pred_mask, voxel_um)
    ref_density = skeleton_length_density(ref_mask, voxel_um)
    out["pred_length_density"] = pred_density
    out["ref_length_density"] = ref_density
    both = bool(np.asarray(pred_mask).any()) and bool(np.asarray(ref_mask).any())
    # NaN, not 0 or 1: with a side empty the ratio is undefined, and 0 would
    # read as "every vessel was lost" while 1 would read as "nothing changed".
    out["length_density_ratio"] = (float(pred_density / ref_density)
                                   if both and ref_density > 0 else float("nan"))
    return out


def depth_profile(pred_mask: np.ndarray, ref_mask: np.ndarray, voxel_um,
                  n_slabs: int = 6, tolerance_vox: float = 1) -> list[dict]:
    """The metric bundle per z slab, so a depth-dependent failure stays visible.

    Signal-to-noise falls with depth, so a model can be defensible in the top
    slabs and invent vessels at the bottom; a volume average hides exactly that.
    Slabs are equal index ranges of axis 0 (at most ``n_slabs`` of them, never
    empty), and ``z_lo_um``/``z_hi_um`` are depths from the first plane of the
    array, using ``voxel_um[0]`` as the slab thickness -- the convention of
    ``scripts/profile_depth.py``.

    Each dict carries the keys of :func:`analyse_volume` except the nested
    ``depth_profile`` itself, plus ``z_lo_um`` and ``z_hi_um``. Its
    ``threshold`` key is the binary level used to recover the already-segmented
    mask, not the image threshold, which is applied once before this function.
    ``length_density_ratio`` is NaN in slabs where either mask is empty.

    A single plane has no depth axis, so it is reported as one slab spanning
    one voxel of axial sampling rather than being cut into meaningless bands.
    """
    pm = np.asarray(pred_mask, dtype=bool)
    rm = np.asarray(ref_mask, dtype=bool)
    if pm.shape != rm.shape:
        raise ValueError(f"pred_mask {pm.shape} and ref_mask {rm.shape} differ")
    if int(n_slabs) < 1:
        raise ValueError(f"n_slabs must be >= 1, got {n_slabs}")
    _voxel_scales(voxel_um, pm.ndim)              # validates voxel_um early
    z_um = float(np.atleast_1d(voxel_um)[0])
    if pm.ndim != 3:                              # a plane is already one slab
        n_slabs = 1

    n_z = pm.shape[0]
    # Balanced split of the planes, extras at the top; empty groups (a stack
    # thinner than n_slabs) are dropped rather than reported as zero slabs.
    groups = [g for g in np.array_split(np.arange(n_z), int(n_slabs)) if g.size]
    out = []
    for group in groups:
        lo, hi = int(group[0]), int(group[-1]) + 1
        slab = _mask_metrics(pm[lo:hi], rm[lo:hi], voxel_um, tolerance_vox)
        if slab["n_empty"]:
            # Two empty masks score dice = 1.0 by the usual convention, and a
            # slab that a gate emptied is *not* a perfect reconstruction of a
            # vessel-less slab -- it is a slab with nothing to score. On a real
            # M2 volume the bottom third of the stack is gated out, so averaging
            # slabs would otherwise report a flattering number for the hardest
            # part of the volume. NaN + n_empty=1 says "no number here".
            for key in ("dice", "cl_dice", "f1"):
                if key in slab:
                    slab[key] = float("nan")
        # A single plane spans one voxel of axial sampling, not n_rows of depth.
        slab["z_lo_um"] = 0.0 if pm.ndim != 3 else float(lo) * z_um
        slab["z_hi_um"] = z_um if pm.ndim != 3 else float(hi) * z_um
        out.append(slab)
    return out


def analyse_volume(pred_img: np.ndarray, ref_img: np.ndarray, *,
                   threshold: float | None = None,
                   voxel_um: tuple[float, float, float] = (5.0, 0.98, 0.98),
                   n_slabs: int = 6, tolerance_vox: float = 1,
                   threshold_mode: str = "per_plane",
                   segment: str = "vesselness",
                   max_plane_fraction: float | None = 0.15,
                   return_masks: bool = False) -> dict:
    """The full structural verdict for one predicted volume against its reference.

    Merges, on one pair of masks:

    * ``mask_vessel_metrics`` -- dice, cl_dice, pred_fraction, ref_fraction,
      n_components_pred, n_components_ref;
    * ``skeleton_stats`` of both masks, prefixed ``pred_`` and ``ref_``;
    * ``skeleton_overlap`` -- ``precision``, ``recall``, ``f1`` within
      ``tolerance_vox``;
    * ``pred_length_density`` / ``ref_length_density`` and their ratio
      ``length_density_ratio`` -- how much vessel length the prediction invents
      or loses, independent of how thick it draws it;
    * ``depth_profile`` -- the same bundle per z slab.

    The threshold is fitted on the **reference** image only (Otsu unless
    ``threshold`` is given) and then applied unchanged to both, so the operating
    point never depends on the prediction and two models cannot be compared at
    two different thresholds by accident.

    ``threshold_mode`` decides *where* that Otsu is fitted, and on real MPM stacks
    it matters enough to be reported:

    ``per_plane``  (default) one Otsu per z plane, from that plane of the
        reference, applied to both. Brightness falls off steeply with depth, so a
        single global threshold classifies the surface planes as almost entirely
        foreground; measured on a real stack, global Otsu put **73 % of the
        volume** above threshold, which makes the skeleton of a solid blob and
        every topological count meaningless. Per-plane also matches how
        ``evaluate`` scores a single plane, so the volume table and the plane
        table agree by construction.
    ``global``     one Otsu over the whole reference volume. Kept because it is
        the classic recipe and a reviewer may ask for it, and because on a
        brightness-flat volume the two coincide.

    Planes whose **reference** segmentation covers more than
    ``max_plane_fraction`` (15 %) are emptied in both masks and counted in
    ``n_planes_gated`` -- see :func:`spur_mpm.metrics.vessel.segment_pair`. On a
    real M2 stack those planes are out-of-focus haze and dense bright regions
    where the threshold stops describing vessels, and they dominate the volume
    total if they are scored (a 170 um slab contributed 3437 mm of "vessel" where
    the clean slabs of the same volume contribute 20-40 mm).

    ``segment`` chooses what is thresholded: ``vesselness`` (default) applies the
    Sato tubular response to both images and thresholds that, because a raw
    intensity Otsu marks a third of a real plane as vessel (see
    :func:`spur_mpm.metrics.vessel.vesselness`); ``intensity`` is the old recipe
    and is kept for the comparison table.

    ``length_density_ratio`` is NaN when either mask is empty. The ratio is then
    undefined -- 0 would claim every vessel was lost and 1 would claim nothing
    changed -- but every ratio-free number is still reported, so an empty result
    degrades instead of raising. ``return_masks=True`` adds the two masks under
    ``pred_mask``/``ref_mask``, so an export or a figure can use exactly the
    masks the numbers came from.
    """
    p = np.asarray(pred_img, dtype=np.float32)
    r = np.asarray(ref_img, dtype=np.float32)
    if p.shape != r.shape:
        raise ValueError(f"pred_img {p.shape} and ref_img {r.shape} differ")
    if p.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D plane or a 3-D volume, got {p.ndim}-D")

    seg = segment_pair(p, r, method=segment, threshold=threshold,
                       threshold_mode=threshold_mode,
                       max_fraction=max_plane_fraction)
    pm, rm = seg["pred_mask"], seg["ref_mask"]
    thr = seg["threshold"]

    out = _mask_metrics(pm, rm, voxel_um, tolerance_vox)
    out["threshold_mode"] = threshold_mode
    out["segment_method"] = segment
    out["n_planes_gated"] = seg["n_planes_gated"]
    out["max_plane_fraction"] = seg["max_plane_fraction"]
    if seg["threshold_p10_p90"] is not None:
        out["threshold_p10_p90"] = seg["threshold_p10_p90"]
    # Fill in the image-level threshold truthfully; every other value already
    # comes from the masks above, so nothing is recomputed differently.
    out["threshold"] = thr
    out["pred_fraction"] = float(pm.mean())
    out["ref_fraction"] = float(rm.mean())
    out["depth_profile"] = depth_profile(pm, rm, voxel_um, n_slabs=n_slabs,
                                         tolerance_vox=tolerance_vox)
    if return_masks:
        out["pred_mask"] = pm
        out["ref_mask"] = rm
    return out


def export_imagej_mask(mask: np.ndarray, path) -> str:
    """Write ``mask`` as an 8-bit multipage TIFF that ImageJ opens as a stack.

    The numbers above are only believable if someone can look at the mask, so
    the export is part of the protocol: 0/255 rather than 0/1, written in the
    ImageJ flavour of multipage TIFF so Fiji opens a (Z, Y, X) stack without an
    import dialog and every viewer agrees on which axis is depth. Plain ``.tif``
    only -- the ``.zip`` hyperstack is ImageJ's own container and is not produced
    here, so a ``.zip`` target is rejected instead of quietly writing something
    that looks different in Fiji.
    """
    import tifffile                                  # only needed for export

    target = str(path)
    if target.lower().endswith(".zip"):
        raise ValueError("ImageJ .zip hyperstacks are not written; use a .tif target")
    data = np.where(np.asarray(mask, dtype=bool), np.uint8(255), np.uint8(0))
    tifffile.imwrite(target, data, imagej=True, photometric="minisblack",
                     metadata={"axes": "ZYX" if data.ndim == 3 else "YX"})
    return target
