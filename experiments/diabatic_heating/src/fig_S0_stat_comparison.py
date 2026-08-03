"""S0 — why the response scalar changed from a single-cell extremum to a percentile.

Every ΔPV curve in this study reduces a 3-D field to one number inside a fixed box
(the heated core r ≤ 2σ, crossed with the pole's pressure band). The first round did
that with ``np.nanmax`` / ``np.nanmin``: **one grid cell out of ~1300 decided the
entire curve**. This figure shows what changes when the same population is reduced by
its 95th/90th percentile instead — same box, same runs, same bundles, only the
reduction differs.

Three panels per row, one row per case (weak 0917 / strong 0920):

* left, middle — the sweep and the 5-day series under all three reductions;
* right — where the reduction sits (pressure), which is the part the extremum
  really does get wrong: the argmax cell snaps between two or three discrete model
  levels as the amplitude changes, so "the response moves into the core" reads as a
  staircase. A percentile's |ΔPV|-weighted exceedance centroid moves continuously.

The jitter numbers printed in the caption are

    J_abs = mean |y_{i+1} - 2 y_i + y_{i-1}|        (units of the curve)
    J_rel = J_abs / mean |y_i|                      (dimensionless)

second differences along the curve: how much a point departs from the straight line
through its neighbours. Read J_abs and J_rel together — a percentile of a positive
field is a smaller number than its maximum, so a modest J_abs improvement can still
show up as a worse J_rel purely from the change of scale.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _data as D
import style as S

_LS = {"max": ("-", "o", 2.4), "p95": ("--", "s", 1.9), "p90": (":", "^", 1.6)}
_STAT_C = {"max": "#D55E00", "p95": "#0072B2", "p90": "#009E73"}


def jitter(y):
    """(absolute, relative) mean second difference of a curve."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return float("nan"), float("nan")
    j = float(np.mean(np.abs(y[2:] - 2 * y[1:-1] + y[:-2])))
    scale = float(np.mean(np.abs(y)))
    return j, (j / scale if scale else float("nan"))


def _series_curves(fam, tag, init, ic, pole, mode):
    """{stat: (hours, values, pressures)} for one 5-day run."""
    run = str(D.run(fam, tag.format(init=init), ic=ic))
    out = {}
    for s in D.STATS:
        ser = D.pv_response_series(run, stat=s)
        k = D._stat_key(pole, mode, s)
        out[s] = (ser["hour"], ser[k], ser[f"{k}_p"])
    return out


def plot(pole: str = "lowlevel", lead_hr: int = 24, *, ic: str = D.DEFAULT_IC,
         style: str = "note"):
    S.apply()
    mode = "max" if pole == "lowlevel" else "min"
    pts = D.sweep(D.FAMILY["heating_moist"][0], lead_hr, ic=ic, stat="max")
    inits = [i for i in (D.WEAK, D.STRONG) if i in pts]

    fig, axes = plt.subplots(len(inits), 3, figsize=(16.5, 4.6 * len(inits)),
                             squeeze=False)
    notes = []
    for r, init in enumerate(inits):
        amps = np.array([a for a, _ in pts[init]])
        series = _series_curves("heating_moist", "tseries_5K_120h_init{init}",
                                init, ic, pole, mode)
        t_end = D.forcing_end_hours(str(D.run(
            "heating_moist", f"tseries_5K_120h_init{init}", ic=ic)))

        for s in D.STATS:
            ls, mk, lw = _LS[s]
            c = _STAT_C[s]
            k = D._stat_key(pole, mode, s)

            # (a) amplitude sweep
            y = np.array([m[k] for _a, m in pts[init]])
            ja, jr = jitter(y)
            axes[r, 0].plot(amps, y, color=c, ls=ls, lw=lw, marker=mk, ms=4,
                            mfc=c if ls == "-" else "none",
                            label=f"{s}   J={ja:.4f} ({jr:.1%})")

            # (b) 5-day series
            hours, vals, press = series[s]
            ja2, jr2 = jitter(vals)
            axes[r, 1].plot(hours, vals, color=c, ls=ls, lw=lw,
                            label=f"{s}   J={ja2:.4f} ({jr2:.1%})")

            # (c) where the reduction sits, along the sweep
            pk = np.array([m[f"{k}_p"] for _a, m in pts[init]])
            jp, _ = jitter(pk)
            axes[r, 2].plot(amps, pk, color=c, ls=ls, lw=lw, marker=mk, ms=4,
                            mfc=c if ls == "-" else "none",
                            label=f"{s}   J={jp:.1f} hPa, {len(set(np.round(pk,3)))} "
                                  f"distinct")
            notes.append((init, s, ja, jr, ja2, jr2, jp))

        S.zero_line(axes[r, 0])
        axes[r, 0].set_xlabel("Heating amplitude amp_K  [K]")
        axes[r, 0].set_ylabel(f"ΔPV at hour {lead_hr}  [PVU]")
        axes[r, 0].set_title(f"{D.CASE[init]} · amplitude sweep")
        axes[r, 0].set_xlim(left=0)

        S.zero_line(axes[r, 1])
        S.forcing_span(axes[r, 1], t_end, label=(r == 0))
        axes[r, 1].set_xlabel("Iteration n  (hour)")
        axes[r, 1].set_ylabel("ΔPV  [PVU]")
        axes[r, 1].set_title(f"{D.CASE[init]} · 5-day series (5 K)")

        axes[r, 2].set_xlabel("Heating amplitude amp_K  [K]")
        axes[r, 2].set_ylabel("level of the reduction  [hPa]")
        axes[r, 2].set_title(f"{D.CASE[init]} · where it sits")
        axes[r, 2].invert_yaxis()
        axes[r, 2].set_xlim(left=0)
        for ax in axes[r]:
            ax.legend(loc="best", fontsize=S.FS_LEGEND - 1.5)

    S.panel_letters(axes)
    where = "low-level generation (700–1000 hPa)" if pole == "lowlevel" \
        else "upper-level destruction (200–500 hPa)"
    fig.suptitle(f"S0 — extremum vs 95th/90th percentile: {where}  ·  {D.IC_LABEL[ic]}")
    S.caption(fig, "identical box (core r ≤ 2σ × the pole's pressure band) and identical "
                   "bundles — only the reduction differs; J = mean |second difference| "
                   "along the curve (relative J in brackets); the staircase in the right "
                   "panels is the extremum cell snapping between model levels", style)
    fig.tight_layout()

    print(f"\n{'case':12s} {'stat':5s} {'sweep J_abs':>12s} {'J_rel':>8s} "
          f"{'series J_abs':>13s} {'J_rel':>8s} {'level J[hPa]':>13s}")
    for init, s, ja, jr, ja2, jr2, jp in notes:
        print(f"{D.CASE[init]:12s} {s:5s} {ja:12.5f} {jr:7.1%} "
              f"{ja2:13.5f} {jr2:7.1%} {jp:13.2f}")
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--pole", choices=["lowlevel", "upperlevel"], default="lowlevel")
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/s0_stat_comparison_<pole>.png")
    a = ap.parse_args()
    # This figure IS the stat comparison, so it always draws all three; --stat only
    # picks which folder it is filed under (use the one the report leads with).
    out = a.out or D.figs_dir(a.ic, a.stat) / f"s0_stat_comparison_{a.pole}.png"
    S.save(plot(a.pole, a.lead, ic=a.ic, style=a.style), out,
           style=a.style, pdf=not a.no_pdf)
