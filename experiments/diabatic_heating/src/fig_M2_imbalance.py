"""M2 — are the lock-induced collapses dynamical breakdown, or fuel starvation?

A rescue hypothesis for q's uniqueness: if lock-z killed the vortex by dynamical
imbalance (a numerical breakdown) while lock-q killed it by cutting the moisture
supply (a graceful decay), q would still be physically special. This tests it.

Excess gradient-wind imbalance ⟨|v_t − v_gr|⟩_pert − ⟨…⟩_ctrl at 850 hPa,
r = 50–500 km, per iteration, for free heating and the three locks. ≈0 means the
intervention left the vortex in gradient balance (so any collapse is fuel
starvation); a sustained positive excess would mean the lock itself broke the
balance (dynamical inconsistency).

The rescue fails: no lock drives the excess above ~0.3 m/s (< 1 % of the ~40 m/s
vortex). The noisiest run is free heating (largest response); lock-q is the
quietest. All three collapses are loop-severing decay, not numerical breakdown —
so q's specialness rests on collapse depth/timing and the injection experiments,
not on lock-w/z being an artefact.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import _data as D
import style as S

_TREATMENTS = [
    ("heating_moist", "tseries_5K_120h_init{init}", "free heating", "#111111", "-"),
    ("heating_qlock", "tseriesq_5K_120h_init{init}", "δq = 0", S.C_LAT, "-"),
    ("heating_wlock", "tseriesw_5K_120h_init{init}", "δw = 0", S.C_STRONG, "--"),
    ("heating_zlock", "tseriesz_5K_120h_init{init}", "δz = 0", "#009E73", "-."),
]


def plot(init: str = D.STRONG, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    from llat_manifold.diagnostics.response import imbalance_series
    t_end = D.forcing_end_hours(str(D.run("heating_moist",
                                          "tseries_5K_120h_init{}".format(init), ic=ic)))

    fig, ax = plt.subplots(figsize=(11, 6.2))
    S.zero_line(ax)
    S.forcing_span(ax, t_end)
    for fam, tag, lab, c, ls in _TREATMENTS:
        s = imbalance_series(str(D.run(fam, tag.format(init=init), ic=ic)))
        ax.plot(s["hour"], s["excess"], color=c, ls=ls,
                lw=2.4 if fam == "heating_moist" else 1.9, marker="o", ms=2.5, label=lab)
    ax.set_xlabel("Iteration n  (hour)")
    ax.set_ylabel(r"excess $\langle|v_t - v_{gr}|\rangle$ at 850 hPa,"
                  "\nr = 50–500 km  [m s$^{-1}$]")
    ax.set_xlim(s["hour"][0], s["hour"][-1])
    ax.legend(loc="upper right")
    ax.set_title(f"M2 — collapse mode under channel locks: imbalance stays < 1 %  "
                 f"({D.CASE[init]}, 5 K)")
    S.caption(fig, "≈0 = the lock keeps the vortex in gradient balance (collapse = fuel "
                   "starvation); no lock exceeds ~0.3 m/s on a ~40 m/s vortex → all "
                   "collapses are physical decay, not numerical breakdown", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/m2_imbalance_locks.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "m2_imbalance_locks.png"
    S.save(plot(a.init, a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
