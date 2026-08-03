"""B1 (basics) — what actually goes into the model.

Before any result means anything, the reader has to be able to picture the
intervention. Every heating run in this study adds the same shape to the
temperature field, scaled by one number ``amp_K``:

    ΔT(p, x, y) = amp_K/8 · V(p) · exp(−r²/2σ²)     once per step, for 8 steps

so the panels are (a) the vertical weight V(p) — the "Deep" profile, a half-sine
that vanishes at 1000 and 200 hPa and peaks near 600 hPa; (b) the horizontal
Gaussian footprint with the σ and 2σ rings marked, 2σ being the core every
diagnostic in the study averages over; (c) the product as an r–z section, in the
units the model actually receives; (d) the time profile — the injection is spread
over the first 24 hours and then stops, and everything after that is free evolution.

Nothing here is a model output. This is the forcing, drawn from the same functions
the driver calls (``perturbations.heating.vertical_profile`` and
``gaussian_centered``), so the figure cannot drift away from what is injected.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _data as D
import style as S
from llat_manifold import layout
from llat_manifold.diagnostics import _idealized as _id
from llat_manifold.perturbations.heating import gaussian_centered, vertical_profile

DEG_KM = 111.32
RES_DEG = 0.25


def plot(amp_K: float = 5.0, sigma: float = 5.0, forcing_steps: int = 8,
         heat_type: str = "Deep", *, ic: str = D.DEFAULT_IC,
         style: str = "note"):
    S.apply()
    p = np.asarray(layout.pressure_levels(), dtype=float)
    npts = 81
    per_step = amp_K / forcing_steps

    v = vertical_profile(heat_type, p)
    g = gaussian_centered(npts, sigma)
    km = (np.arange(npts) - npts // 2) * RES_DEG * DEG_KM
    sig_km = sigma * RES_DEG * DEG_KM

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.9),
                             gridspec_kw={"width_ratios": [0.8, 1.1, 1.25, 1.0]})

    # (a) vertical weight -----------------------------------------------------
    ax = axes[0]
    ax.plot(v, p, color=S.C_SENS, lw=2.6, marker="o", ms=4)
    ax.axvline(0, color=S.C_GUIDE, lw=0.8)
    ax.set_ylim(1000, _id.P_TOP_HPA)
    ax.set_xlabel("vertical weight  V(p)")
    ax.set_ylabel("pressure  [hPa]")
    ax.set_title(f"(a) {heat_type} profile")
    kpk = int(np.argmax(v))
    ax.annotate(f"peak {p[kpk]:.0f} hPa", (v[kpk], p[kpk]), xytext=(-6, 14),
                textcoords="offset points", fontsize=S.FS_ANNOT, color=S.C_NOTE,
                ha="right")

    # (b) horizontal footprint ------------------------------------------------
    ax = axes[1]
    half = 24                       # ±160 km window around the centre
    sl = slice(npts // 2 - half, npts // 2 + half + 1)
    pm = ax.pcolormesh(km[sl], km[sl], g[sl, sl], cmap="Reds", shading="auto")
    th = np.linspace(0, 2 * np.pi, 200)
    for f, ls, lab in ((1, "--", "σ"), (2, "-", "2σ = core")):
        ax.plot(f * sig_km * np.cos(th), f * sig_km * np.sin(th),
                color="k", ls=ls, lw=1.4)
        ax.annotate(f"{lab}  {f * sig_km:.0f} km", (0, f * sig_km),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=S.FS_ANNOT, color="k")
    ax.set_aspect("equal")
    ax.set_xlabel("x  [km]")
    ax.set_ylabel("y  [km]")
    ax.set_title("(b) horizontal footprint")
    fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.02, label="Gaussian weight")

    # (c) the product, in the units the model receives -------------------------
    ax = axes[2]
    field = per_step * v[:, None] * g[npts // 2, None, sl]
    lim = float(np.abs(field).max())
    cf = ax.contourf(km[sl], p, field, levels=np.linspace(0, lim, 21), cmap="Reds")
    ax.contour(km[sl], p, field, levels=np.linspace(0, lim, 21),
               colors="k", linewidths=0.3, alpha=0.4)
    ax.set_ylim(1000, _id.P_TOP_HPA)
    ax.set_xlabel("distance from centre  [km]")     # a west–east cut, not a radius
    ax.set_ylabel("pressure  [hPa]")
    ax.set_title(f"(c) ΔT per step ({per_step:g} K/step)")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02, label="ΔT  [K per step]")

    # (d) time profile ---------------------------------------------------------
    ax = axes[3]
    hours = np.arange(0, 123, 3)
    step_amp = np.where(hours <= forcing_steps * 3, per_step, 0.0)
    step_amp[0] = 0.0
    cum = np.cumsum(step_amp)
    S.forcing_span(ax, forcing_steps * 3, label=False)
    ax.step(hours, step_amp, where="post", color=S.C_SENS, lw=2.0,
            label=f"per step  ({per_step:g} K)")
    ax.plot(hours, cum, color=S.C_TOTAL, lw=2.4, ls="--",
            label=f"cumulative  (→ {amp_K:g} K)")
    ax.set_xlabel("Iteration n  (hour)")
    ax.set_ylabel("injected ΔT  [K]")
    ax.set_title("(d) injection schedule")
    ax.set_xlim(0, 120)
    ax.legend(loc="center right")

    fig.suptitle(f"B1 — the intervention: a {heat_type} heating bump of amp_K = {amp_K:g} K "
                 f"spread over the first {forcing_steps * 3} h")
    S.caption(fig, "drawn from the same vertical_profile() and gaussian_centered() the "
                   "driver calls, so it is the forcing itself, not a redrawing of it; "
                   "after hour 24 nothing more is added and the perturbation evolves freely",
              style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--amp", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--heat-type", default="Deep",
                    choices=["Deep", "Shallow", "Stratiform"])
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/b1_what_was_injected.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "b1_what_was_injected.png"
    S.save(plot(a.amp, a.sigma, heat_type=a.heat_type, ic=a.ic, style=a.style),
           out, style=a.style, pdf=not a.no_pdf)
