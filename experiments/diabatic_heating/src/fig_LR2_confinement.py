"""LR2 — how far the thermal response spread, against each vortex's own L_R.

Once the reduction is restricted to the heated band (see ``fig_LR1_metric_audit``),
the question the H4 panel was being asked to answer becomes answerable: does the
strong vortex hold the injected warmth near the axis, as the local Rossby radius
argument says it must?

    L_R = N·H / √(ξη),   ξ = f + 2V_t/r,  η = f + ζ

The two RAGASA stages have essentially identical stratification (N ≈ 1.20e-2 s⁻¹ for
both), so every bit of the L_R difference between them is inertial stability — which
is exactly the variable the theory speaks about.

  (a, b) azimuthal-mean, mass-weighted 400–700 hPa δT vs radius, strong vs weak,
         one panel per injection type, with each case's own L_R marked;
  (c)    cumulative fraction of the column sensible-energy anomaly vs radius, with
         r50 dropped onto the axis — the confinement number in one reading;
  (d)    r50 against L_R over all six runs: the scatter the theory predicts to be
         monotone.

Nothing here is a new model run: it is the existing 5-day series re-reduced.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

_RUNS = [  # (family, tag template, short label)
    ("heating_moist", "tseries_5K_120h_init{init}", "inject ΔT"),
    ("dq_latent", "tseriesdql_5K_120h_init{init}", "inject δq (latent-equiv)"),
    ("dq_measured", "tseriesdq_5K_120h_init{init}", "inject δq (measured)"),
]
_BAND = (400.0, 700.0)
_MARK = {"ragasa": "o", "axisym": "s"}


def _lr_of(run, ic):
    from llat_manifold import io
    from llat_manifold.diagnostics import waves
    from llat_manifold.diagnostics.response import _continuous_pairs
    _lead, _d, c = _continuous_pairs(run)[0]
    up, sfc = io.load_delta_bundle(c)
    return waves.rossby_radius_summary(up, sfc)


def plot(lead_hr: int = 24, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    from llat_manifold.diagnostics.response import (confinement_radii,
                                                    warm_core_profile,
                                                    energy_cumulative)

    S.apply()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    (axA, axB), (axC, axD) = axes

    lr = {init: _lr_of(D.run("heating_moist", f"tseries_5K_120h_init{init}", ic=ic), ic)
          for init in (D.WEAK, D.STRONG)}

    # (a) radial δT profile, heated pair; (b) the same for the equal-energy δq pair
    for ax, (fam, tag, lab) in zip((axA, axB), _RUNS[:2]):
        for init in (D.WEAK, D.STRONG):
            run = str(D.run(fam, tag.format(init=init), ic=ic))
            r, prof = warm_core_profile(run, lead_hr, p_band=_BAND)
            col = S.C_STRONG if init == D.STRONG else S.C_WEAK
            m = confinement_radii(run, lead_hr, p_band=_BAND)
            l_r = lr[init]["L_R_core_km"]
            ax.plot(r, prof, color=col, lw=2.2,
                    label=f"{D.CASE[init]} · r50 = {m['r50']:.0f} km · "
                          f"$L_R$ = {l_r:.0f} km")
            # The weak case's L_R sits beyond the plotted range; an arrow at the edge
            # says "off to the right" instead of a label floating outside the axes.
            if l_r <= 1000:
                ax.axvline(l_r, color=col, lw=1.4, ls="--", alpha=0.85)
            else:
                ax.annotate("", xy=(1000, 0.0), xytext=(930, 0.0),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
                ax.text(925, 0.0, f"$L_R$ = {l_r:.0f} km ", color=col, ha="right",
                        va="bottom", fontsize=S.FS_ANNOT + 1, weight="bold")
        S.zero_line(ax)
        ax.axvspan(0, 278, color=S.C_GUIDE, alpha=0.10, lw=0)
        ax.set_xlim(0, 1000)
        ax.set_xlabel("radius [km]")
        ax.set_ylabel(f"azimuthal-mean δT, {_BAND[0]:.0f}–{_BAND[1]:.0f} hPa  [K]")
        ax.set_title(f"{lab} — radial spread of the warm anomaly")
        ax.legend(loc="upper right")

    # (c) cumulative energy fraction
    for fam, tag, lab in _RUNS:
        for init in (D.WEAK, D.STRONG):
            run = str(D.run(fam, tag.format(init=init), ic=ic))
            m = confinement_radii(run, lead_hr, p_band=_BAND)
            col = S.C_STRONG if init == D.STRONG else S.C_WEAK
            ls = {"heating_moist": "-", "dq_latent": "--", "dq_measured": ":"}[fam]
            rc, frac = energy_cumulative(run, lead_hr)
            axC.plot(rc, frac, color=col, ls=ls, lw=1.9,
                     label=f"{lab} · {D.CASE[init]}")
            axC.plot(m["r50"], 0.5, marker=_MARK.get(ic, "o"), color=col, ms=7,
                     mec="k", mew=0.7, zorder=5)
    axC.axhline(0.5, color=S.C_GUIDE, lw=0.9, ls="-")
    axC.set_xlim(0, 1000)
    axC.set_ylim(0, 1.02)
    axC.set_xlabel("radius [km]")
    axC.set_ylabel("cumulative fraction of the warm anomaly")
    axC.set_title("markers = r50, the radius holding half the anomaly")
    axC.legend(loc="lower right", fontsize=S.FS_LEGEND - 1.2)

    # (d) r50 vs L_R. Only two vortices exist here, so this is a *direction* plot, not
    # a scaling law — one segment per injection type, both ends labelled. The scaling
    # exponent needs the six-vortex sweep in fig_LR3_intensity_sweep.py.
    for fam, tag, lab in _RUNS:
        mk = {"heating_moist": "o", "dq_latent": "s", "dq_measured": "^"}[fam]
        pts = []
        for init in (D.WEAK, D.STRONG):
            run = str(D.run(fam, tag.format(init=init), ic=ic))
            m = confinement_radii(run, lead_hr, p_band=_BAND)
            pts.append((lr[init]["L_R_core_km"], m["r50"]))
            axD.plot(pts[-1][0], pts[-1][1], mk,
                     color=S.C_STRONG if init == D.STRONG else S.C_WEAK,
                     ms=12, mec="k", mew=0.8, zorder=4,
                     label=lab if init == D.WEAK else None)
        axD.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="0.45",
                 lw=1.3, zorder=1)
    axD.plot([], [], "o", color=S.C_WEAK, mec="k", label="weak (0917)")
    axD.plot([], [], "o", color=S.C_STRONG, mec="k", label="strong (0920)")
    axD.set_xscale("log")
    axD.set_yscale("log")
    axD.set_xlabel("local Rossby radius $L_R = NH/\\sqrt{\\xi\\eta}$, core mean  [km]")
    axD.set_ylabel("r50  [km]")
    axD.set_title("confinement vs the adjustment scale\n"
                  "(two vortices only — direction, not a scaling law)")
    axD.legend(loc="upper left", fontsize=S.FS_LEGEND - 1.2)

    S.panel_letters(axes)
    fig.suptitle("LR2 — the strong vortex confines the injected warmth, as $L_R$ "
                 f"requires (5 K runs, nominal hour {lead_hr})\n"
                 f"{D.ic_note(ic, stat)} · $L_R$: weak "
                 f"{lr[D.WEAK]['L_R_core_km']:.0f} km vs strong "
                 f"{lr[D.STRONG]['L_R_core_km']:.0f} km "
                 f"(N differs by <1 %, so this is all inertial stability)")
    S.caption(fig, "r50 = radius containing half the positive column sensible-energy "
                   "anomaly; every pair has the strong vortex more confined than the "
                   "weak one", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=None, help="default: figs/lr/lr2_confinement_<ic>.png")
    a = ap.parse_args()
    out = a.out or (D.FIGS_ROOT / "lr" / f"lr2_confinement_{a.ic}.png")
    S.save(plot(a.lead, style=a.style, ic=a.ic, stat=a.stat), out,
           style=a.style, pdf=not a.no_pdf)
