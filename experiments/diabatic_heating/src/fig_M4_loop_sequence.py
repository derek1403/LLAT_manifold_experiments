"""M4 mechanism — the T–q–w–z loop as a secondary circulation, and how each lock kills it.

Five loop observables per iteration, each normalised by its own maximum in the free
run: warm core δT (400–700 hPa), core column δq, core ascent −δw, low-level inflow
−δu_r (150–400 km), and moisture import −δ(u_r·q). These map the abstract channels
onto the textbook CISK/WISHE loop: inflow → ascent → latent heating → warm core →
pressure fall → inflow. 2×2 grid: free reference, then δq = 0 / δw = 0 / δz = 0.

Each lock severs the chain at a different node, and the order in which the curves
collapse traces the causal chain. Lock q: warm core grows during forcing, then
loses latent support and goes negative by n≈45. Lock w: almost untouched during
forcing (moisture even peaks), then moisture cannot cash out and bleeds away.
Lock z: mass field frozen, the loop detunes and oscillates most violently — yet M2
shows even this is physical decay, not numerical breakdown.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

# The five observables and their fixed hues.
_OBS = [("warm_core", "warm core δT (400–700)", S.C_STRONG),
        ("moisture", "core column δq", S.C_LAT),
        ("updraft", "core ascent −δw (400–700)", "#E69F00"),
        ("inflow", "low-level inflow −δu_r (150–400 km)", "#009E73"),
        ("q_import", "moisture import −δ(u_r q)", "#CC79A7")]

_PANELS = [("heating_moist", "tseries_5K_120h_init{init}", "free heating (reference)"),
           ("heating_qlock", "tseriesq_5K_120h_init{init}", "δq = 0"),
           ("heating_wlock", "tseriesw_5K_120h_init{init}", "δw = 0"),
           ("heating_zlock", "tseriesz_5K_120h_init{init}", "δz = 0")]


def plot(init: str = D.STRONG, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    from llat_manifold.diagnostics.response import loop_series
    series = [loop_series(str(D.run(fam, tag.format(init=init), ic=ic))) for fam, tag, _ in _PANELS]
    norm = {k: max(1e-12, float(np.max(np.abs(series[0][k])))) for k, _l, _c in _OBS}
    t_end = D.forcing_end_hours(str(D.run("heating_moist",
                                          "tseries_5K_120h_init{}".format(init), ic=ic)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True, sharey=True)
    for ax, s, (_fam, _tag, lab) in zip(axes.ravel(), series, _PANELS):
        S.zero_line(ax)
        S.forcing_span(ax, t_end, label=False)
        for k, name, c in _OBS:
            ax.plot(s["hour"], np.asarray(s[k]) / norm[k], color=c, lw=1.9,
                    marker="o", ms=2.5, label=name)
        ax.set_title(lab)
        ax.set_xlim(s["hour"][0], s["hour"][-1])
    for ax in axes[-1, :]:
        ax.set_xlabel("Iteration n  (hour)")
    for ax in axes[:, 0]:
        ax.set_ylabel("normalised by free-run max")
    axes[0, 0].legend(loc="upper right", fontsize=S.FS_LEGEND - 1.5, ncol=1)
    S.panel_letters(axes)
    fig.suptitle(f"M4 — death sequence of the T–q–w–z loop  ({D.CASE[init]}, 5 K)")
    S.caption(fig, "each series normalised by its own free-run max; the order in which curves "
                   "collapse after a lock traces the causal chain "
                   "(inflow → ascent → latent heating → warm core → inflow)", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/m4_loop_sequence.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "m4_loop_sequence.png"
    S.save(plot(a.init, a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
