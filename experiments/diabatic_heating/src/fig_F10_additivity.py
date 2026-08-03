"""Methodology legitimacy — can the response be split into a T-path and a q-path?

Much of the argument decomposes the response into severed halves (δq = 0 = T
path, δq-only = q path) and measures them separately. This checks that the split
is legitimate. On the shared control trajectory, add the ΔPV fields point-by-point
and give the sum the same dipole reduction as the moist field, then form the
synergy factor S = moist ÷ ((δq = 0) + δq-only): S = 1 means the decomposition is
complete, > 1 super-additive (T×q coupling builds response neither half owns), < 1
sub-additive (the two halves double-count a shared resource).

Low-level (solid) and upper-level (dashed) S vs amplitude, per init. Strong
low-level sits on 1 to ~8 K — the split is near-linear on the data-richest pathway,
which legitimises the whole two-halves method. Weak low-level falls to 0.31: its
nonlinear amplifier is a shared resource, so "weak δq overshoots heating 3×" must
not be extrapolated to "both together are stronger still".
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S


def plot(lead_hr: int = 24, band=(0.8, 1.25), style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    from llat_manifold.diagnostics.response import additivity_series
    data = additivity_series(str(D.category(ic)), lead_hr=lead_hr)
    inits = sorted(data)
    color = S.case_colors(inits)

    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.axhspan(band[0], band[1], color=S.C_GUIDE, alpha=0.16, lw=0, zorder=0)
    ax.axhline(1.0, color="#333333", lw=1.3, zorder=1)

    for init in inits:
        rec = data[init]
        amps = np.asarray(rec["amps"])
        c = color[init]
        for pole, name, ls, mk in (("low", "low-level max", "-", "o"),
                                   ("up", "upper-level min", "--", "s")):
            r = np.asarray(rec[f"moist_{pole}"]) / np.asarray(rec[f"sum_{pole}"])
            r = np.where(r > 0, r, np.nan)
            ax.plot(amps, r, color=c, ls=ls, lw=2.6 if pole == "low" else 1.9,
                    marker=mk, ms=5, mfc=c if ls == "-" else "none",
                    label=f"{D.CASE.get(init, init)} · {name}")

    ax.set_yscale("log", base=2)
    ax.set_yticks([0.25, 0.5, 0.75, 1, 1.5, 2])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1", "1.5", "2"])
    ax.set_xlim(left=0)
    ax.set_xlabel("Nominal amplitude amp_K  [K]")
    ax.set_ylabel("synergy factor  S = moist ÷ ((δq = 0) + δq-only)")
    ax.set_title(f"Additivity — can the response be split into a T-path and a q-path?  "
                 f"(hour {lead_hr})")
    ax.text(0.985, 1.0, " S = 1: decomposition complete ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=S.FS_ANNOT, style="italic", color="#333333")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")
    S.caption(fig, "ΔPV fields summed on the shared control before the dipole reduction; strong "
                   "low-level ≈ 1 (split legitimate) but weak low-level falls to 0.31 (the "
                   "amplifier is a shared resource, not additive)", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/f10_additivity.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "f10_additivity.png"
    S.save(plot(a.lead, style=a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
