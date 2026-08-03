"""H5 routing — does moisture reach PV through δT, or straight to the circulation?

Two independent lock experiments, read as a survival fraction of the free δq-only
response, side by side. Both use the measured δq scaling; the free baseline is
``dq_measured`` (``sweepdq_*``).

* Left — δT = 0 (``dq_tlock``, ``sweepdqtl_*``):  S = ΔPV(δq, δT = 0) / ΔPV(δq).
  δθ ≡ 0, so with PV ∝ (ζ+f)·∂θ/∂p the surviving ΔPV is the vorticity term alone.
  S ≈ 1 ⇒ the response never needed a temperature anomaly.

* Right — δu = δv = 0 (``dq_uvlock``, ``sweepdquv_*``):  S = ΔPV(δq, δu = δv = 0)/ΔPV(δq).
  The vorticity term is now cut instead. S small ⇒ the response lived in the wind.

The two panels interlock pole-by-pole: where one lock kills the response the other
spares it. Low-level generation survives δT = 0 (~100 %) but dies under
δu = δv = 0 (~23 %) → a direct q→vorticity line. Upper-level destruction does the
opposite → a δT-mediated (thermal) pathway. That crossing is the H5 result.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S


def _survival(free_pat, lock_pat, lead_hr, ic, stat):
    """{init: {pole_key: (amps, S)}} = locked ÷ free, per pole."""
    free = D.sweep(free_pat, lead_hr, ic=ic, stat=stat)
    lock = D.sweep(lock_pat, lead_hr, ic=ic, stat=stat)
    out = {}
    for init in sorted(set(free) & set(lock)):
        a, b = dict(free[init]), dict(lock[init])
        amps = sorted(set(a) & set(b))
        out[init] = {key: (np.array(amps),
                           np.array([b[x][key] / a[x][key] for x in amps]))
                     for key, _ in D.poles(stat)}
    return out


def _draw_panel(ax, data, color, stat, *, title, ylab):
    ax.axhspan(0.85, 1.15, color=S.C_GUIDE, alpha=0.13, lw=0, zorder=0)
    ax.axhline(1.0, color="#333333", lw=1.2, zorder=1)
    ax.axhline(0.0, color=S.C_GUIDE, lw=0.8, zorder=1)
    for init, poles in data.items():
        c = color[init]
        for (key, name), ls, mk, lw in ((D.poles(stat)[0], "-", "o", 2.6),
                                        (D.poles(stat)[1], "--", "s", 1.9)):
            amps, surv = poles[key]
            ax.plot(amps, surv, color=c, ls=ls, lw=lw, marker=mk, ms=5,
                    mfc=c if ls == "-" else "none", zorder=3,
                    label=f"{D.CASE.get(init, init)} · {name}")
    ax.set_ylim(-0.05, 1.4)
    ax.set_xlim(left=0)
    ax.set_xlabel("Nominal amplitude amp_K  [K]")
    if ylab:
        ax.set_ylabel("Survival fraction  S")
    ax.set_title(title)


def plot(lead_hr: int = 24, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    tlock = _survival(D.FAMILY["dq_measured"][0], D.FAMILY["dq_tlock"][0],
                      lead_hr, ic, stat)
    uvlock = _survival(D.FAMILY["dq_measured"][0], D.FAMILY["dq_uvlock"][0],
                       lead_hr, ic, stat)
    color = S.case_colors(tlock)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    _draw_panel(axL, tlock, color, stat,
                title="δT = 0  (S = 1 ⇒ no temperature anomaly needed)", ylab=True)
    _draw_panel(axR, uvlock, color, stat,
                title="δu = δv = 0  (S small ⇒ response lives in the wind)", ylab=False)

    # "S = 1" note anchored top-left of the left panel — clear of every curve.
    axL.text(0.03, 1.0, " S = 1 ", transform=axL.get_yaxis_transform(),
             ha="left", va="bottom", fontsize=S.FS_ANNOT, style="italic", color="#333333")

    # The one sentence each panel is really saying, coloured to its own hero curve.
    axL.text(0.5, 0.06, "low-level (solid) survives → q reaches vorticity directly",
             transform=axL.transAxes, ha="center", fontsize=S.FS_ANNOT + 0.5,
             weight="bold", color=S.C_STRONG)
    axR.text(0.5, 0.06, "low-level (solid) dies → that response was carried by the wind",
             transform=axR.transAxes, ha="center", fontsize=S.FS_ANNOT + 0.5,
             weight="bold", color=S.C_STRONG)

    axL.legend(loc="upper right", ncol=1, fontsize=S.FS_LEGEND - 0.5)
    S.panel_letters([axL, axR])
    S.caption(fig, "both locks are heavy interventions — read S as a direction, not an exact "
                   "share; low-level survives δT = 0 but not δu = δv = 0 (direct q→vorticity), "
                   "upper-level does the reverse (δT-mediated)", style)
    fig.suptitle(f"H5 — routing of the moisture-driven PV response  "
                 f"(δq-only, hour {lead_hr})")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/h5_routing_ratio.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "h5_routing_ratio.png"
    S.save(plot(a.lead, a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
