"""H2 — can moisture alone drive the PV response, with no heating at all?

The reverse probe. Inject a δq anomaly with the same spatial design as the heating
(vertical profile × centred Gaussian) but θ̇ = 0 (``dq_measured``, measured
scaling). A dry-dynamical model should give ΔPV ≈ 0; anything that appears is
q-channel-routed. Plotted against the heating sweep on the shared nominal-amplitude axis:
solid = heating (ΔT), dashed = δq-only. Left low-level, right upper-level.

At 5 K the δq-only response already reaches 29–91 % of the heating response across
the four poles, and the weak-vortex low level overshoots heating ~3× by 10 K. q is
not a passenger — it is a driver.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import _data as D
import style as S


def plot(lead_hr: int = 24, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    heat = D.sweep(D.FAMILY["heating_moist"][0], lead_hr, ic=ic, stat=stat)
    dq = D.sweep(D.FAMILY["dq_measured"][0], lead_hr, ic=ic, stat=stat)
    color = S.case_colors(set(heat) | set(dq))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), sharex=True)
    for ax, (key, name) in zip((axL, axR), D.poles(stat)):
        S.zero_line(ax)
        for init in sorted(set(heat) | set(dq)):
            c = color[init]
            for pts, ls, mfc, tag in ((heat.get(init), "-", c, "heating (ΔT)"),
                                      (dq.get(init), "--", "none", "δq-only (measured, θ̇=0)")):
                if not pts:
                    continue
                amps = [a for a, _ in pts]
                ax.plot(amps, [m[key] for _, m in pts], color=c, ls=ls, lw=2.2,
                        marker="o" if ls == "-" else "^", ms=5, mfc=mfc,
                        label=f"{D.CASE.get(init, init)} · {tag}")
        ax.set_xlabel("Nominal amplitude amp_K  [K]")
        ax.set_title(name)
        ax.set_xlim(left=0)
    axL.set_ylabel(f"ΔPV at hour {lead_hr}  [PVU]")
    axR.set_ylabel(f"ΔPV at hour {lead_hr}  [PVU]")
    #axL.legend(loc="upper left")
    # 保持對齊左上角，但將基準點往下移動約 10% 的高度
    axL.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95))
    S.panel_letters([axL, axR])
    fig.suptitle("H2 — the reverse probe: heating vs δq-only (measured scaling) "
                 "on one nominal-amplitude axis")
    S.caption(fig, "measured scaling: at nominal amp_K, inject the δq the moist heating run "
                   "itself grew by hour 24 — only 0.37–0.49× the heating's energy; yet δq-only "
                   "reaches 29–91 % of heating at 5 K (weak low-level overshoots ~3× by 10 K)", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/h2_dq_sweep.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "h2_dq_sweep.png"
    S.save(plot(a.lead, a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
