"""H3 — is the q→PV gain arbitrary statistics, or the physical energy conversion?

Equivalence ratio ΔPV(latent-equiv δq) / ΔPV(heating) vs amplitude, log-2 axis.
Latent-equivalent δq (``dq_latent``) injects Δq = (c_p/L_v)·ΔT — the moisture whose
complete condensation releases exactly amp_K of heating — so the injection is
strictly equal-energy to the heating sweep. A ratio of 1 then means the manifold
converts moisture and heat at the physical c_p/L_v rate on that pathway; the grey
band is the ±25 % "practically equivalent" zone.

Four curves (init × pole). Only the strong-vortex low level (orange solid) rides 1
across the whole range (0.77–0.99) — the manifold has encoded a near-correct
L_v·δq ↔ c_p·ΔT on its data-richest pathway. Every other pathway's gain is
empirical: strong upper is 2–4×, weak low diverges to ~7×, and where weak upper
dips into the band at 9–10 K it is a shared saturation ceiling, not equivalence.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S


def plot(lead_hr: int = 24, band=(0.75, 1.25), style: str = "note"):
    S.apply()
    heat = D.sweep(D.FAMILY["heating_moist"][0], lead_hr)
    dql = D.sweep(D.FAMILY["dq_latent"][0], lead_hr)
    inits = sorted(set(heat) & set(dql))
    color = S.case_colors(inits)

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.axhspan(band[0], band[1], color=S.C_GUIDE, alpha=0.16, lw=0, zorder=0)
    ax.axhline(1.0, color="#333333", lw=1.3, zorder=1)

    for init in inits:
        a, b = dict(heat[init]), dict(dql[init])
        amps = sorted(set(a) & set(b))
        c = color[init]
        for (key, name), ls, mk, lw in ((D.POLES[0], "-", "o", 2.8),
                                        (D.POLES[1], "--", "s", 1.9)):
            r = [b[x][key] / a[x][key] for x in amps]
            ax.plot(amps, r, color=c, ls=ls, lw=lw, marker=mk, ms=5,
                    mfc=c if ls == "-" else "none", zorder=3,
                    label=f"{D.CASE.get(init, init)} · {name}")

    ax.set_yscale("log", base=2)
    ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8])
    ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2", "3", "4", "6", "8"])
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("Nominal amplitude amp_K  [K]")
    ax.set_ylabel("ΔPV ratio   latent-equiv δq ÷ heating")
    ax.set_title(f"H3 — latent-heat equivalence ratio (nominal hour {lead_hr})")

    # The two findings annotations, now real code (the old PNG had them hand-patched).
    ax.text(0.985, 1.0, " ratio = 1: c_p ΔT ↔ L_v δq equivalence ",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=S.FS_ANNOT, style="italic", color="#333333")
    ax.annotate("strong low-level: 0.77–0.99 across all amplitudes\n"
                "→ energy conversion encoded on this pathway",
                xy=(6.0, 0.99), xytext=(3.0, 0.5), fontsize=S.FS_ANNOT + 0.5,
                weight="bold", color=S.C_STRONG, ha="center",
                arrowprops=dict(arrowstyle="->", color=S.C_STRONG, lw=1.2))
    ax.annotate("weak upper dips into the band at 9–10 K:\n"
                "shared saturation ceiling, not equivalence",
                xy=(9.5, 1.0), xytext=(7.6, 1.9), fontsize=S.FS_ANNOT,
                style="italic", color=S.C_WEAK, ha="center",
                arrowprops=dict(arrowstyle="->", color=S.C_WEAK, lw=1.0))

    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    S.caption(fig, "equal-energy injection (L_v δq = c_p ΔT); a curve on 1 means the manifold "
                   "converts moisture and heat at the physical c_p/L_v rate on that pathway", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=str(D.FIGS / "h3_dql_equivalence_ratio.png"))
    a = ap.parse_args()
    S.save(plot(a.lead, style=a.style), a.out, style=a.style, pdf=not a.no_pdf)
