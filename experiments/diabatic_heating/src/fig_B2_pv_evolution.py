"""B2 (basics) — pure heating in, PV out: the whole spatial story in one strip.

Every later figure in this study compresses the PV response to a single number per
run, which is what makes the amplitude sweeps and survival ratios possible — and
what hides the thing a reader most needs to see first: the response is a *dipole
that grows and moves*. This figure shows it uncompressed. Rows are treatments,
columns are lead times, each panel the azimuthal-mean ΔPV(r, p) at that hour, on one
shared colour scale so panels are comparable across the whole strip.

Read left to right:

* through hour 24 the heating is still going in (orange column headers): a positive
  PV column grows on the axis (r ≲ 100 km), and — worth noticing, because it is not
  quite the textbook picture — it is not confined below the 600 hPa heating maximum
  but reaches well up through the column;
* after hour 24 nothing more is injected, so everything further right is the
  manifold's own evolution: the positive column persists for the full five days,
  and the negative lobe aloft (200–300 hPa) arrives *late*, deepening only around
  n ≈ 96–120 h. The destruction half of the dipole is not the prompt balanced
  response; it is something the run grows on its own long after the forcing stops.

The second row is the δq-only run at the same nominal amplitude — the same figure
for an injection that adds *no heat at all*. That it is not blank is the entire
reverse-probe argument, in a form that needs no metric to read.

Sections stop at 200 hPa: above that the model's levels thin out and its own error
dominates whatever we injected.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _data as D
import style as S
from llat_manifold import io
from llat_manifold.diagnostics import _idealized as _id
from llat_manifold.diagnostics.response import _azimuthal_mean, analyze_pair

# (family, run tag, row label) — the heating baseline and its zero-heat counterpart.
_ROWS = [
    ("heating_moist", "tseries_5K_120h_init{init}", "inject ΔT\n(heating, q free)"),
    ("dq_measured", "tseriesdq_5K_120h_init{init}", "inject δq only\n(θ̇ = 0)"),
]
_HOURS = (3, 6, 12, 24, 48, 72, 96, 120)


def _panels(run_dir, hours, sigma=5.0):
    """{hour: (r_km, z or p axis, azimuthal-mean ΔPV)} for the requested leads."""
    data = run_dir / "data"
    out = {}
    for h in hours:
        hits = sorted(data.glob(f"delta_continuous_*lead{h:03d}hr.npz"))
        if not hits:
            continue
        d = hits[0]
        c = d.parent / d.name.replace("delta_", "control_")
        _m, aux = analyze_pair(d, c, sigma=sigma)
        r_km, az = _azimuthal_mean(aux["dpv"], aux["lat2d"], aux["lon2d"])
        out[h] = (r_km, aux["p_hpa"], az)
    if not out:
        raise FileNotFoundError(f"no requested leads under {data}")
    return out


def plot(init: str = D.STRONG, hours=_HOURS, *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT, style: str = "note"):
    S.apply()
    rows = []
    for fam, tag, label in _ROWS:
        try:
            run = D.run(fam, tag.format(init=init), ic=ic)
        except FileNotFoundError as e:
            print(f"[B2] skipping row {fam}: {e}")
            continue
        rows.append((label, _panels(run, hours)))
    if not rows:
        raise FileNotFoundError("no 5-day runs available for B2")

    have = [h for h in hours if all(h in p for _l, p in rows)]
    # Scale to what is actually on screen. PV grows by orders of magnitude toward the
    # stratosphere, so including the levels above 200 hPa (which this figure crops)
    # would set the colour range from a cell nobody can see and blank every panel.
    # The 99.5th percentile rather than the max, for the same reason one grid cell
    # should not decide a curve.
    vals = []
    for _l, panels in rows:
        for h in have:
            r_km, p_hpa, az = panels[h]
            win = np.ix_(np.asarray(p_hpa) >= _id.P_TOP_HPA, r_km <= 500.0)
            vals.append(np.abs(az[win]).ravel())
    lim = float(np.nanpercentile(np.concatenate(vals), 99.5)) or 1.0
    levels = np.linspace(-lim, lim, 25)

    fig, axes = plt.subplots(len(rows), len(have), squeeze=False, sharey=True,
                             figsize=(2.05 * len(have) + 1.4, 3.5 * len(rows)))
    t_end = 24
    for i, (label, panels) in enumerate(rows):
        for j, h in enumerate(have):
            ax = axes[i, j]
            r_km, p_hpa, az = panels[h]
            cf = ax.contourf(r_km, p_hpa, az, levels=levels, cmap=S.CMAP_PV,
                             extend="both")
            ax.contour(r_km, p_hpa, az, levels=[0], colors="k", linewidths=0.5)
            ax.set_ylim(1000, _id.P_TOP_HPA)
            ax.set_xlim(0, 500)
            ax.tick_params(labelsize=S.FS_TICK - 2)
            if i == 0:
                forced = " (forcing on)" if h <= t_end else ""
                ax.set_title(f"n = {h} h{forced}", fontsize=S.FS_TICK,
                             color=(S.C_SENS if h <= t_end else "k"))
            if i == len(rows) - 1:
                ax.set_xlabel("radius [km]", fontsize=S.FS_TICK - 1)
        axes[i, 0].set_ylabel(f"{label}\n\npressure [hPa]", fontsize=S.FS_TICK)

    cb = fig.colorbar(cf, ax=axes.ravel().tolist(), pad=0.012, fraction=0.02)
    cb.set_label("azimuthal-mean ΔPV  [PVU]", fontsize=S.FS_LABEL - 1, weight="bold")
    fig.suptitle(f"B2 — heating → PV: the response building and then living on its own  "
                 f"({D.CASE.get(init, init)}, 5 K, {D.IC_LABEL[ic]})")
    S.caption(fig, "red = PV generated, blue = PV destroyed, black line = zero; columns "
                   "up to n = 24 h still have heating going in, everything to the right "
                   "is free evolution; sections stop at 200 hPa where the model's own "
                   "error takes over", style)
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--hours", nargs="+", type=int, default=list(_HOURS))
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/b2_pv_evolution.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "b2_pv_evolution.png"
    S.save(plot(a.init, tuple(a.hours), ic=a.ic, stat=a.stat, style=a.style),
           out, style=a.style, pdf=not a.no_pdf)
