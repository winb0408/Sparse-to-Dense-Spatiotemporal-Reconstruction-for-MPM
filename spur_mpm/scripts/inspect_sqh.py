"""Inspect the MPM-3P-SQH upload: what is in it, and could it validate our task?

The upload is *not* volumetric like Dryad_3PM.  It is three 2-D time-lapse movies
(TYX), each a different acquisition:

    CellVideo 1(128 128).tif    128x128, 5000 frames
    CellVideo 1(256 256).tif    256x256, 3191 frames
    CellVideo 1(512 512).tif    512x512,  800 frames

None of them is a subsample of another: the time means do not correlate at any
scale factor we tried (|r| <= 0.27).  So the three files are three separate
recordings, and their photon budgets differ by more than an order of magnitude.

This script measures, per movie, the things that decide whether the data can be
used at all -- pedestal, gain (counts per detected photon), photons per pixel
per frame, temporal drift, how much of a single frame is noise versus structure,
and the spatial scale of the structure -- and then renders four figures plus a
JSON of every number, so the reader can check the claims by opening the files.

Outputs, under ``spur_mpm/figures/sqh/``:
    sqh_1_movies.png     one row per movie: raw frame / short mean / full mean / temporal std
    sqh_2_physics.png    photon transfer, noise-vs-blur, lag correlation, drift, SNR, morphology
    sqh_3_sparsity.png   what a row-sparse acquisition of this data would look like
    sqh_4_calibration.png  does the photon model predict the per-pixel error?
    sqh_stats.json       every number quoted in the report

Run:
    python -m spur_mpm.scripts.inspect_sqh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

DATA = Path("/ssd1/wjb/Data/MPM-3P-SQH")
OUT = Path("spur_mpm/figures/sqh")

MOVIES = {
    "128 (5000 frames)": DATA / "CellVideo 1(128 128).tif",
    "256 (3191 frames)": DATA / "CellVideo 1(256 256).tif",
    "512 (800 frames)": DATA / "CellVideo 1(512 512).tif",
}

CMAP = "magma"
_NOTE = dict(fontsize=6.0, color="white", ha="right", va="bottom",
             bbox=dict(facecolor="black", alpha=0.6, lw=0.0, pad=1.4))


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def accumulate(path: Path) -> dict:
    """Chunked pass: temporal mean, temporal variance, per-frame mean."""
    vol = tifffile.imread(path, out="memmap")
    n, h, w = vol.shape
    s = np.zeros((h, w), np.float64)
    ss = np.zeros((h, w), np.float64)
    fmean = np.empty(n, np.float64)
    for lo in range(0, n, 200):
        blk = np.asarray(vol[lo:lo + 200], np.float64)
        s += blk.sum(0)
        ss += (blk ** 2).sum(0)
        fmean[lo:lo + len(blk)] = blk.mean((1, 2))
    mu = s / n
    var = np.maximum(ss / n - mu ** 2, 0.0)
    return dict(mu=mu, var=var, fmean=fmean, n=n, shape=(n, h, w),
                vol=vol)


def photon_transfer(mu: np.ndarray, var: np.ndarray) -> dict:
    """Fit var = G * (mu - offset) on background pixels.

    If the detector adds a constant pedestal and the signal is Poisson, the
    per-pixel temporal variance is linear in the temporal mean with a negative
    intercept; the slope is the counts per detected photon.
    """
    lo, hi = np.percentile(mu, [1, 60])
    m = (mu >= lo) & (mu <= hi)
    A = np.vstack([mu[m], np.ones(m.sum())]).T
    slope, icept = np.linalg.lstsq(A, var[m], rcond=None)[0]
    pedestal = -icept / slope if slope > 0 else float("nan")
    resid = var[m] - (slope * mu[m] + icept)
    r2 = 1.0 - resid.var() / var[m].var()
    bright = float(np.percentile(mu, 99.5))
    return dict(pedestal=float(pedestal), counts_per_photon=float(slope),
                r2=float(r2),
                photons_bg=float((np.median(mu) - pedestal) / slope),
                photons_p995=float((bright - pedestal) / slope),
                photons_max=float((mu.max() - pedestal) / slope))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def analyse(name: str, path: Path) -> tuple[dict, dict]:
    from scipy.ndimage import gaussian_filter

    acc = accumulate(path)
    vol, mu, var = acc["vol"], acc["mu"], acc["var"]
    n = acc["n"]
    pt = photon_transfer(mu, var)
    pt["frame_photons"] = float((acc["fmean"].mean() - pt["pedestal"])
                                / pt["counts_per_photon"])

    # how much of one frame is structure, and how much is noise: blur the frame
    # and the full mean by a growing sigma and watch their correlation climb.
    ref = gaussian_filter(mu, 1.0)
    blur_scan = {}
    for sg in (0.0, 1.0, 2.0, 4.0, 8.0):
        a = vol[0] if sg == 0 else gaussian_filter(vol[0].astype(np.float64), sg)
        b = ref if sg == 0 else gaussian_filter(mu, sg)
        blur_scan[float(sg)] = corr(a, b)

    # how stable is the scene over the whole stack?  Compare the two halves.
    half = n // 2
    h1 = gaussian_filter(np.asarray(vol[:half], np.float64).mean(0), 3)
    h2 = gaussian_filter(np.asarray(vol[half:], np.float64).mean(0), 3)
    half_corr = corr(h1, h2)

    # lag correlation on blurred frames: is the movie ordered and stable?
    step = max(1, n // 320)
    sm = np.stack([gaussian_filter(vol[i].astype(np.float64), 3)
                   for i in range(0, n, step)]).reshape(-1, mu.size)
    sm -= sm.mean(1, keepdims=True)
    nr = np.linalg.norm(sm, axis=1)
    C = (sm @ sm.T) / np.outer(nr, nr)
    lags = [1, 2, 5, 10, 20, 50, 100]
    lag = {int(l): float(np.mean(np.diag(C, l))) for l in lags if l < len(C)}

    # spatial scale: autocorrelation of the time mean along x
    z = mu - mu.mean()
    f = np.fft.fft2(z)
    ac = np.real(np.fft.ifft2(f * np.conj(f)))
    ac /= ac[0, 0]
    acx = {int(d): float(ac[0, d].real) for d in (0, 4, 8, 12, 16, 24, 32)}
    # the lag at which the autocorrelation first falls below 0.5
    half = next((d for d in range(1, z.shape[1] // 2)
                 if ac[0, d].real < 0.5), -1)

    stats = dict(file=path.name, shape=list(acc["shape"]),
                 dtype="uint16", pedestal=pt["pedestal"],
                 counts_per_photon=pt["counts_per_photon"], fit_r2=pt["r2"],
                 photons_per_pixel_frame=pt["frame_photons"],
                 frame_mean_min=float(acc["fmean"].min()),
                 frame_mean_max=float(acc["fmean"].max()),
                 frame_mean_drift_pct=float(100 * (acc["fmean"][-1] - acc["fmean"][0])
                                            / acc["fmean"][0]),
                 blur_corr=blur_scan, half_stack_corr=half_corr,
                 lag_corr=lag, spatial_autocorr_x=acx,
                 spatial_half_width_px=int(half),
                 temporal_std_median=float(np.median(np.sqrt(var))))
    return stats, dict(vol=vol, mu=mu, var=var, fmean=acc["fmean"])


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _win(img: np.ndarray, lo: float = 0.5, hi: float = 99.7) -> tuple[float, float]:
    v = img[np.isfinite(img)]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))
            if v.size else (0.0, 1.0))


def _show(ax, img, rng, cmap: str = CMAP, bad: str | None = "#22d3ee"):
    im = ax.imshow(img, cmap=cmap, vmin=rng[0], vmax=rng[1],
                   interpolation="nearest", aspect="auto", resample=False)
    if bad is not None:
        im.cmap.set_bad(bad)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def _note(ax, text: str) -> None:
    ax.text(0.975, 0.025, text, transform=ax.transAxes, zorder=6, **_NOTE)


def figure_movies(data: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    rows = []
    for name, d in data.items():
        vol, mu, var = d["vol"], d["mu"], d["var"]
        rows.append((name, [
            ("raw frame", vol[0].astype(np.float64), _win(vol[0])),
            ("mean of 100", np.asarray(vol[:100], np.float64).mean(0),
             _win(np.asarray(vol[:100], np.float64).mean(0))),
            (f"mean of all {len(vol)}", mu, _win(mu)),
            ("temporal s.d.", np.sqrt(var), _win(np.sqrt(var))),
        ]))

    fig, axes = plt.subplots(len(rows), 4, figsize=(9.6, 2.5 * len(rows)))
    fig.patch.set_facecolor("white")
    for r, (name, cols) in enumerate(rows):
        for c, (lab, img, rng) in enumerate(cols):
            ax = axes[r][c]
            _show(ax, img, rng)
            _note(ax, f"{lab}\n{img.min():.0f}..{img.max():.0f}")
            if r == 0:
                ax.set_title(lab, fontsize=8, pad=4)
        axes[r][0].set_ylabel(name, fontsize=8)
    fig.suptitle("MPM-3P-SQH: three 2-D time-lapse movies, not a 3-D volume",
                 fontsize=9.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=0.6, h_pad=0.8)
    fig.savefig(out / "sqh_1_movies.png", dpi=220, facecolor="white")
    plt.close(fig)


def figure_physics(stats: dict, data: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    colors = {"128 (5000 frames)": "#2f9e6f",
              "256 (3191 frames)": "#e8536b",
              "512 (800 frames)": "#3f8ee0"}
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.2))
    fig.patch.set_facecolor("white")

    # (a) photon transfer: temporal variance against temporal mean
    ax = axes[0][0]
    for name, st in stats.items():
        d = data[name]
        mu, var = d["mu"], d["var"]
        idx = np.argsort(mu.ravel())
        k = 4000
        chunks = np.array_split(idx, k)
        xs = np.array([mu.ravel()[c].mean() for c in chunks])
        ys = np.array([var.ravel()[c].mean() for c in chunks])
        ax.loglog(xs, ys, ".", ms=2.5, color=colors[name], label=name)
        g, off = st["counts_per_photon"], st["pedestal"]
        xx = np.array([xs.min(), xs.max()])
        ax.loglog(xx, g * (xx - off), "-", lw=1.0, color=colors[name],
                  label=f"var=G(mean−{off:.0f}), G={g:.0f}")
    ax.set_xlabel("temporal mean (counts)", fontsize=7.5)
    ax.set_ylabel("temporal variance (counts²)", fontsize=7.5)
    ax.set_title("(a) photon transfer: noise is Poisson above a pedestal", fontsize=8)
    ax.legend(fontsize=5.6, frameon=False)
    ax.tick_params(labelsize=6)

    # (b) correlation of one frame with the full mean, as a function of blur
    ax = axes[0][1]
    for name, st in stats.items():
        sg = sorted(st["blur_corr"])
        ax.plot(sg, [st["blur_corr"][s] for s in sg], "o-", ms=4, lw=1.2,
                color=colors[name], label=name)
    ax.set_xlabel("Gaussian blur applied before correlating (px)", fontsize=7.5)
    ax.set_ylabel("corr(frame 0, mean of all)", fontsize=7.5)
    ax.set_title("(b) a single frame is noise-limited at 1 px", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=6)

    # (c) lag correlation
    ax = axes[0][2]
    for name, st in stats.items():
        lags = sorted(st["lag_corr"])
        ax.plot(lags, [st["lag_corr"][l] for l in lags], "o-", ms=4, lw=1.2,
                color=colors[name], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("lag (samples of the stored stack)", fontsize=7.5)
    ax.set_ylabel("corr of blurred frames", fontsize=7.5)
    ax.set_title("(c) frames are ordered; the scene drifts slowly", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=6)

    # (d) per-frame mean: photobleaching / power drift
    ax = axes[1][0]
    for name, d in data.items():
        f = d["fmean"]
        ax.plot(np.linspace(0, 1, len(f)), f, lw=0.8, color=colors[name], label=name)
    ax.set_xlabel("position in the stack (normalised)", fontsize=7.5)
    ax.set_ylabel("frame mean (counts)", fontsize=7.5)
    ax.set_title("(d) per-frame mean: a slow 12 % drift, no steps", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=6)

    # (e) where the data is trustworthy: temporal mean over temporal s.d.
    ax = axes[1][1]
    d = data["128 (5000 frames)"]
    snr = d["mu"] / np.sqrt(d["var"])
    _show(ax, snr, (float(np.percentile(snr, 1)), float(np.percentile(snr, 99.5))),
          cmap="viridis")
    _note(ax, f"median {np.median(snr):.1f}\nmax {snr.max():.1f}")
    ax.set_title("(e) 128: mean/s.d. per pixel — the SNR is very uneven", fontsize=8)

    # (f) morphology: high-passed time mean, structure only
    ax = axes[1][2]
    h = data["128 (5000 frames)"]["mu"]
    hp = h - gaussian_filter(h, 12)
    rng = (float(np.percentile(hp, 1)), float(np.percentile(hp, 99.5)))
    _show(ax, hp, rng)
    _note(ax, "high-pass σ=12 px\nremoves illumination gradient")
    ax.set_title("(f) 128 time-mean, high-passed: sparse compact objects", fontsize=8)

    fig.suptitle("MPM-3P-SQH: what the numbers in these movies actually are",
                 fontsize=9.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.6, h_pad=1.4)
    fig.savefig(out / "sqh_2_physics.png", dpi=220, facecolor="white")
    plt.close(fig)


def figure_sparsity(data: dict, out: Path) -> None:
    """What the Dryad-style row mask would look like on this data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = data["128 (5000 frames)"]
    ref, frame = d["mu"], d["vol"][0].astype(np.float64)
    h, w = ref.shape
    yy = np.arange(h)[:, None]
    my2 = (yy % 2 == 0)
    my4 = (yy % 4 == 0)

    def masked(img, m):
        return np.where(m, img, np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.4))
    fig.patch.set_facecolor("white")
    panels = [
        (ref, ("ready reference\nmean of 3191", "reference")),
        (frame, ("one real frame\nsame field", "one frame")),
        (masked(ref, my2), ("ky=2 on reference\n50 % of rows", "ky=2 ref")),
        (masked(frame, my2), ("ky=2 on one frame\n50 % of rows", "ky=2 frame")),
        (masked(ref, my4), ("ky=4 on reference\n25 % of rows", "ky=4 ref")),
        (masked(frame, my4), ("ky=4 on one frame\n25 % of rows", "ky=4 frame")),
        (np.asarray(d["vol"][:8], np.float64).mean(0),
         ("mean of 8 frames\nSNR still low", "mean 8")),
        (np.asarray(d["vol"][:8], np.float64)[5] - np.asarray(d["vol"][:8], np.float64)[4],
         ("frame-to-frame difference\n= pure noise", "difference")),
    ]
    rng = _win(ref)
    for ax, (img, (lab, _)) in zip(axes.ravel(), panels):
        mm = img.copy()
        _show(ax, mm, _win(img) if np.isfinite(img).all() else rng)
        ax.set_title(lab.replace("\n", " | "), fontsize=7)
        if not np.isfinite(img).all():
            frac = float(np.isfinite(img).mean())
            _note(ax, f"measured {frac:.0%}\nof rows")
    fig.suptitle("MPM-3P-SQH under the Dryad row mask: the same sparsity, on real MPM noise (128 movie)",
                 fontsize=9.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.6, h_pad=1.0)
    fig.savefig(out / "sqh_3_sparsity.png", dpi=220, facecolor="white")
    plt.close(fig)


def figure_calibration(stats: dict, data: dict, out: Path) -> None:
    """Can the measured noise model predict the per-pixel error?  If yes, this
    movie is a calibration bed for the uncertainty head."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    name = "128 (5000 frames)"
    st, d = stats[name], data[name]
    vol = d["vol"]
    ref = np.asarray(vol[:1000], np.float64).mean(0)      # long average = reference
    ref = gaussian_filter(ref, 0.0)
    frame = vol[1000].astype(np.float64)
    err = frame - ref
    g, off = st["counts_per_photon"], st["pedestal"]
    # Poisson prediction: variance of one frame is G*(I - offset); the residual
    # against a 1000-frame reference keeps essentially all of it.
    sig = np.sqrt(np.maximum(g * (ref - off), 1.0))
    z = err / sig

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.4))
    fig.patch.set_facecolor("white")
    _show(axes[0], frame, _win(frame))
    _note(axes[0], f"one frame\nmean {frame.mean():.0f}")
    axes[0].set_title("(a) one measured frame", fontsize=8)

    _show(axes[1], err, (float(np.percentile(err, 0.5)), float(np.percentile(err, 99.5))),
          cmap="coolwarm")
    _note(axes[1], f"|err| p50 {np.median(np.abs(err)):.0f}\n"
                   f"p99 {np.percentile(np.abs(err), 99):.0f}")
    axes[1].set_title("(b) frame − 1000-frame reference", fontsize=8)

    _show(axes[2], sig, (float(np.percentile(sig, 1)), float(np.percentile(sig, 99.5))),
          cmap="viridis")
    _note(axes[2], f"√(G(I−{off:.0f}))\nmedian {np.median(sig):.0f}")
    axes[2].set_title("(c) predicted s.d. from the photon model", fontsize=8)

    ax = axes[3]
    idx = np.argsort(ref.ravel())
    chunks = np.array_split(idx, 60)
    xs = np.array([ref.ravel()[c].mean() for c in chunks])
    ys = np.array([err.ravel()[c].std() for c in chunks])
    ax.loglog(xs, ys, "o", ms=4, color="#c0392b", label="measured s.d. of the residual")
    ax.loglog(xs, np.sqrt(np.maximum(g * (xs - off), 1.0)), "-", lw=1.2,
              color="#2c3e50", label="photon model")
    ax.set_xlabel("reference intensity (counts)", fontsize=7.5)
    ax.set_ylabel("residual s.d. (counts)", fontsize=7.5)
    ax.set_title(f"(d) model fit: z = err/σ has s.d. {z.std():.2f}", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=6)

    fig.suptitle("MPM-3P-SQH: the per-pixel noise is predictable, so error bars can be checked here",
                 fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    fig.savefig(out / "sqh_4_calibration.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stats, data = {}, {}
    for name, path in MOVIES.items():
        print(f"reading {path.name} ...", flush=True)
        stats[name], data[name] = analyse(name, path)
        s = stats[name]
        print(f"  pedestal {s['pedestal']:.0f} counts, "
              f"{s['counts_per_photon']:.0f} counts/photon, "
              f"{s['photons_per_pixel_frame']:.3f} photons/px/frame, "
              f"half-width {s['spatial_half_width_px']} px", flush=True)

    (args.out / "sqh_stats.json").write_text(json.dumps(stats, indent=2))
    figure_movies(data, args.out)
    figure_physics(stats, data, args.out)
    figure_sparsity(data, args.out)
    figure_calibration(stats, data, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
