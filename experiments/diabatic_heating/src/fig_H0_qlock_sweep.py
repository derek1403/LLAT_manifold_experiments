"""H0 — is the heating→PV response dry-dynamical, or bound to moisture?

ΔPV dipole scalars at hour 24 vs heating amplitude: moist (heating with q
free, solid) against its δq = 0 twin (dashed). Left panel is
low-level generation (700–1000 hPa max), right is upper-level destruction
(200–500 hPa min). Blue = weak (0917), orange = strong (0920).

The gap between solid and dashed is the moisture-mediated share of the response.
If PV generation by θ̇ were the dry process it is on paper, holding δq = 0 would do
nothing and the curves would overlie. They do — but only below ~2 K; above that
the moist curve grows superlinearly while the δq = 0 one falls back toward
linear. The superlinear amplification is moisture-mediated.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import _data as D
import style as S


def plot(lead_hr: int = 24, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    moist = D.sweep(D.FAMILY["heating_moist"][0], lead_hr, ic=ic, stat=stat)
    qlock = D.sweep(D.FAMILY["heating_qlock"][0], lead_hr, ic=ic, stat=stat)
    color = S.case_colors(set(moist) | set(qlock))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), sharex=True)
    for ax, (key, name) in zip((axL, axR), D.poles(stat)):
        S.zero_line(ax)
        for init in sorted(set(moist) | set(qlock)):
            c = color[init]
            for pts, ls, mfc, tag in ((moist.get(init), "-", c, "moist (q free)"),
                                      (qlock.get(init), "--", "none", "δq = 0")):
                if not pts:
                    continue
                amps = [a for a, _ in pts]
                ax.plot(amps, [m[key] for _, m in pts], color=c, ls=ls, lw=2.2,
                        marker="o" if ls == "-" else "s", ms=5, mfc=mfc,
                        label=f"{D.CASE.get(init, init)} · {tag}")
        ax.set_xlabel("Heating amplitude amp_K  [K]")
        ax.set_title(name)
        ax.set_xlim(left=0)
    axL.set_ylabel(f"ΔPV at hour {lead_hr}  [PVU]")
    axR.set_ylabel(f"ΔPV at hour {lead_hr}  [PVU]")
    #axL.legend(loc="upper left")
    # 保持對齊左上角，但將基準點往下移動約 10% 的高度
    axL.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95))
    S.panel_letters([axL, axR])
    fig.suptitle("H0 — moisture binding of the heating→PV response: moist vs δq = 0")
    S.caption(fig, "curves overlie below ~2 K (response is dry there); above it the gap = "
                   "the moisture-mediated share — the superlinear growth is q-mediated", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/h0_qlock_sweep.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "h0_qlock_sweep.png"
    S.save(plot(a.lead, a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
