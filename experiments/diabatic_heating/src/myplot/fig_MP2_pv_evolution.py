"""MP2 — the same heating driven into two intensities, iteration by iteration.

[`fig_B2_pv_evolution.py`](../fig_B2_pv_evolution.py) shows the uncompressed spatial
response — azimuthal-mean ΔPV(r, p), columns across the run — for one amplitude and one
vortex. Every other figure in the study collapses that field to a single number per run,
which is what makes the sweeps possible and what hides how the response is actually
*shaped*. This set keeps the shape and varies the two things that matter:

* **rows** — the same heating going into Category 1 (0920) and Category 4 (0921). Both
  are axisymmetric ICs sitting in **0920's environment**, so the only thing that differs
  between the rows is the vortex itself. See [`fig_MP1_axisym_pv.py`](fig_MP1_axisym_pv.py)
  for what the two vortices look like before anything is injected;
* **figures** — one per injected amplitude, 1 K … 10 K (and the cooling runs).

**The columns are iterations of the model operator, not forecast hours.** These are
``snapshot`` runs: the background ū = M(u₀) is computed once at frozen valid time and
every iteration is measured against it, with the prescribed surface channels — lat/lon,
SST, f, terrain, land mask — realigned to ū every time. Nothing drifts, nothing makes
landfall, and both members stay in the identical environment for the whole experiment.
What grows across the columns is the vortex's own response and nothing else. Labelling
these columns in hours would be wrong twice over: there is no clock being advanced, and
the growth is a power iteration, not an evolution in time.

Amplitude is injected over the first 8 iterations at ``amp_K/8`` each; everything to the
right of that is the manifold's own amplification with no further forcing.

Both rows of a figure share one colour scale — otherwise the row comparison, which is
the whole point, would be meaningless. Between figures the scale is per-amplitude by
default (the 1 K response is roughly an order of magnitude smaller than the 10 K one),
so **read the colourbar before comparing amplitudes**. ``--shared-scale`` forces one
limit across the set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402

import _data as D                                                     # noqa: E402
import _snap as SNAP                                                  # noqa: E402
import style as S                                                     # noqa: E402
from fig_MP1_axisym_pv import _title as _member_title                 # noqa: E402
from llat_manifold.diagnostics import _idealized as _id               # noqa: E402

# The controlled ladder, weakest first. Both members carry 0920's environment.
_MEMBERS = (D.STRONG, D.SEV)
# Iterations to draw: dense while the forcing is on, then spread out.
_ITERS = (1, 2, 4, 8, 16, 24, 32, 40)
_AMPS = tuple(range(1, 11))

FIGS = D.FIGS_ROOT / "myplot"
R_MAX_KM = SNAP.R_MAX_KM


def _stem(amp: float) -> str:
    """Output-file stem for an amplitude: 5 -> 'mp2-05', -5 -> 'mp2-m05'."""
    return f"mp2-{'m' if amp < 0 else ''}{abs(amp):02.0f}"


def _collect(amp, ns, members=_MEMBERS):
    """[(row label, {n: (r_km, p_hPa, azimuthal-mean ΔPV)})] for one amplitude."""
    rows = []
    for init in members:
        try:
            rows.append((_member_title(init).replace(" (", "\n("),
                         SNAP.panels(amp, init, ns)))
        except FileNotFoundError as e:
            print(f"[MP2] {amp} K: skipping {init} ({e})")
    return rows


def _limit(rows, ns):
    """Colour limit for a strip — B2's rule, so the two figure sets stay comparable.

    The 99.5th percentile of |ΔPV| over what is actually on screen, not the max: PV
    grows by orders of magnitude toward the stratosphere, so a single cell above the
    plotted window would set the range and blank every panel.
    """
    vals = []
    for _label, panels in rows:
        for n in ns:
            if n not in panels:
                continue
            r_km, p_hpa, az = panels[n]
            vals.append(np.abs(az[SNAP.window(r_km, p_hpa)]).ravel())
    if not vals:
        return 0.0
    return float(np.nanpercentile(np.concatenate(vals), 99.5))


def plot(amp, rows, ns, lim, *, style: str = "note"):
    """One amplitude's two-row strip. ``rows``/``lim`` come from :func:`_collect` /
    :func:`_limit` so a shared colour scale can be decided across the set first."""
    S.apply()
    have = [n for n in ns if all(n in p for _l, p in rows)]
    levels = np.linspace(-lim, lim, 25)

    fig, axes = plt.subplots(len(rows), len(have), squeeze=False, sharey=True,
                             figsize=(2.05 * len(have) + 1.4, 3.5 * len(rows) + 1.0))
    for i, (label, panels) in enumerate(rows):
        for j, n in enumerate(have):
            ax = axes[i, j]
            r_km, p_hpa, az = panels[n]
            cf = ax.contourf(r_km, p_hpa, az, levels=levels, cmap=S.CMAP_PV,
                             extend="both")
            ax.contour(r_km, p_hpa, az, levels=[0], colors="k", linewidths=0.5)
            ax.set_ylim(1000, _id.P_TOP_HPA)
            ax.set_xlim(0, R_MAX_KM)
            ax.tick_params(labelsize=S.FS_TICK - 2)
            if i == 0:
                forced = " (forcing on)" if SNAP.forcing_on(n) else ""
                ax.set_title(f"{SNAP.label(n)}{forced}", fontsize=S.FS_TICK,
                             color=(S.C_SENS if SNAP.forcing_on(n) else "k"))
            if i == len(rows) - 1:
                ax.set_xlabel("radius  (km)", fontsize=S.FS_TICK - 1)
        axes[i, 0].set_ylabel(f"{label}\n\npressure  (hPa)", fontsize=S.FS_TICK)

    cb = fig.colorbar(cf, ax=axes.ravel().tolist(), pad=0.012, fraction=0.02)
    cb.set_label("azimuthal-mean ΔPV  (PVU)", fontsize=S.FS_LABEL - 1, weight="bold")
    fig.suptitle(f"MP2 — heating → PV at {amp:g} K: the same forcing into two "
                 f"intensities  (semi-linear iteration, {D.IC_LABEL[SNAP.IC]}, "
                 f"0920 environment)", wrap=True)
    S.caption(fig, "red = PV generated, blue = PV destroyed, black line = zero; n counts "
                   "iterations of the model operator at frozen valid time — NOT forecast "
                   "hours; forcing is injected over n ≤ 8 and everything to the right is "
                   "the manifold amplifying what it already has; the prescribed surface "
                   "channels (lat/lon, SST, f, terrain, land mask) are realigned to the "
                   "background every iteration, so the environment is identical at every "
                   "n and between the two rows; both rows share one colour scale, each "
                   "amplitude has its own unless --shared-scale was used", style,
              wrap=True)
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--amps", nargs="+", type=int, default=list(_AMPS),
                    help="injected amplitudes in K (default: 1 … 10)")
    ap.add_argument("--members", nargs="+", default=list(_MEMBERS),
                    help="ladder members, top row first (default: %(default)s)")
    ap.add_argument("--iters", nargs="+", type=int, default=list(_ITERS),
                    help="iterations to draw as columns (default: %(default)s)")
    ap.add_argument("--shared-scale", action="store_true",
                    help="one colour limit across every amplitude instead of per-figure")
    ap.add_argument("--outdir", default=None,
                    help="default: figs/myplot/ (files mp2-01_pv_evolution.png … )")
    ap.add_argument("--suffix", default="",
                    help="appended to the output stem, so a variant (e.g. --iters 1 2 3 4 "
                         "--suffix _early) does not overwrite the main set")
    a = ap.parse_args()

    ns = tuple(a.iters)
    outdir = Path(a.outdir) if a.outdir else FIGS
    # Collect first, draw second: a shared colour scale has to see every amplitude
    # before the first figure is drawn.
    strips = {}
    for amp in a.amps:
        rows = _collect(amp, ns, tuple(a.members))
        if not rows:
            print(f"[MP2] {amp} K: no runs, skipping")
            continue
        strips[amp] = (rows, _limit(rows, ns))
    if not strips:
        raise SystemExit("[MP2] no runs found for any requested amplitude")

    shared = max(l for _r, l in strips.values()) if a.shared_scale else None
    for amp, (rows, lim) in strips.items():
        use = shared if shared is not None else lim
        print(f"[MP2] {amp:3d} K: {len(rows)} row(s), |ΔPV| colour limit {use:.3f} PVU"
              f"{' (shared)' if shared is not None else ''}")
        S.save(plot(amp, rows, ns, use or 1.0, style=a.style),
               outdir / f"{_stem(amp)}{a.suffix}_pv_evolution.png",
               style=a.style, pdf=not a.no_pdf)
        plt.close("all")
