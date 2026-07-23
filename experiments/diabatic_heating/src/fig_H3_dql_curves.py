"""H3 raw view — heating vs latent-equivalent δq, absolute response curves.

The equivalence-ratio figure is the analyst's view; this is the audience's view:
the two ΔPV amplitude curves plotted on top of each other in absolute units.
Solid = heating (ΔT), dashed = latent-equivalent δq (Δq = (c_p/L_v)·ΔT, strictly
equal energy). Left low-level generation, right upper-level destruction.

The strong-vortex low level (orange) is the punchline: the two curves lie on top
of each other across the whole sweep — give the manifold the moisture worth
amp_K of condensation heating, or the heating itself, and it builds the same
low-level PV. Everywhere else the curves separate (upper level: δq overshoots
2–4×; weak low level: δq keeps growing after heating saturates).
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import _data as D
import style as S


def plot(lead_hr: int = 24, style: str = "note"):
    S.apply()
    heat = D.sweep(D.FAMILY["heating_moist"][0], lead_hr)
    dql = D.sweep(D.FAMILY["dq_latent"][0], lead_hr)
    color = S.case_colors(set(heat) | set(dql))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), sharex=True)
    for ax, (key, name) in zip((axL, axR), D.POLES):
        S.zero_line(ax)
        for init in sorted(set(heat) | set(dql)):
            c = color[init]
            for pts, ls, mfc, tag in ((heat.get(init), "-", c, "heating (ΔT)"),
                                      (dql.get(init), "--", "none", "latent-equiv δq")):
                if not pts:
                    continue
                amps = [a for a, _ in pts]
                ax.plot(amps, [m[key] for _, m in pts], color=c, ls=ls, lw=2.2,
                        marker="o" if ls == "-" else "D", ms=5, mfc=mfc,
                        label=f"{D.CASE.get(init, init)} · {tag}")
        ax.set_xlabel("Nominal amplitude amp_K  [K]")
        ax.set_title(name)
        ax.set_xlim(left=0)
    axL.set_ylabel(f"ΔPV at nominal hour {lead_hr}  [PVU]")
    axR.set_ylabel(f"ΔPV at nominal hour {lead_hr}  [PVU]")
    #axL.legend(loc="upper left")
    # 保持對齊左上角，但將基準點往下移動約 10% 的高度
    axL.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95))
    S.panel_letters([axL, axR])
    fig.suptitle("H3 raw view — equal-energy moisture reproduces the heating curve "
                 "(strong low level)")
    S.caption(fig, "strictly equal-energy injection (L_v δq = c_p ΔT); strong low-level curves "
                   "overlie across the sweep — the manifold's cp/Lv conversion at work — while "
                   "every other pathway separates", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=str(D.FIGS / "h3_dql_curves.png"))
    a = ap.parse_args()
    S.save(plot(a.lead, a.style), a.out, style=a.style, pdf=not a.no_pdf)
