"""LR1 — what the H4 "core max δT" number actually measured.

The v1 report reads panel (a) of ``h4_energy_accounting`` as "the weak vortex's core
warms more than the strong one's" (1.49 K vs 1.19 K), which contradicts the local
Rossby radius argument: a strong vortex has high inertial stability, hence a small
L_R, hence it should trap an injected thermal anomaly near the axis and warm *more*.

This figure shows that the contradiction is in the metric, not in the model.

``_core_max_dT`` takes ``np.nanmax`` over a fixed disc of radius 2σ **across all 13
pressure levels**, with no vertical restriction and no averaging. The Deep heating
profile peaks at 600 hPa and is identically zero at 200 hPa — yet the winning cell is
usually found at 100–250 hPa, where the tropopause's static stability turns a small
vertical displacement into a large δT. The number is a tropopause signal competing
with a warm core, and which one wins differs case by case.

  (a) the old all-level number, each bar annotated with the level it was read off;
  (b) the same reduction restricted to 400–700 hPa (the layer the heating occupies),
      where the weak/strong ordering reverses;
  (c) the azimuthal-mean δT radius–pressure section for the heated pair, with both
      the all-level winning cell and the band-restricted one marked, so the reader
      can see the two cells are in different air masses.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

_RUNS = [  # (family, tag template, short label)
    ("heating_moist", "tseries_5K_120h_init{init}", "inject ΔT"),
    ("dq_latent", "tseriesdql_5K_120h_init{init}", "inject δq\n(latent-equiv)"),
    ("dq_measured", "tseriesdq_5K_120h_init{init}", "inject δq\n(measured)"),
]
_BAR_INITS = [(D.STRONG, "strong"), (D.WEAK, "weak")]
_BAND = (400.0, 700.0)


def _bars(ax, x, vals, colors, fmt="+{:.2f} K", annot=None):
    ax.bar(x, vals, color=colors, alpha=0.75, width=0.62)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02 * max(vals), fmt.format(v), ha="center",
                fontsize=S.FS_ANNOT + 1.5, weight="bold")
    if annot is not None:
        for xi, (v, note) in zip(x, zip(vals, annot)):
            ax.text(xi, v * 0.5, note, ha="center", va="center", rotation=90,
                    fontsize=S.FS_ANNOT + 0.5, weight="bold", color="white")
    ax.set_ylim(0, max(vals) * 1.22)


def plot(lead_hr: int = 24, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    from llat_manifold.diagnostics.response import (confinement_radii, _azimuthal_mean,
                                                    _lead_paths)
    from llat_manifold import io, layout

    S.apply()
    labels, old, new, where, colors = [], [], [], [], []
    for fam, tag, lab in _RUNS:
        for init, who in _BAR_INITS:
            run = str(D.run(fam, tag.format(init=init), ic=ic))
            m = confinement_radii(run, lead_hr, p_band=_BAND)
            labels.append(f"{lab}\n{who}")
            old.append(m["dT_all_max"])
            new.append(m["dT_band_max"])
            where.append(f"{m['dT_all_max_p']:.0f} hPa")
            colors.append(S.C_STRONG if who == "strong" else S.C_WEAK)

    fig = plt.figure(figsize=(15.5, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.42, wspace=0.18)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    x = np.arange(len(labels))

    _bars(axA, x, old, colors, annot=where)
    axA.set_xticks(x, labels, fontsize=S.FS_TICK - 1)
    axA.set_ylabel("core max δT  [K]")
    axA.set_title("as published: max over ALL 13 levels\n"
                  "(white text = the level the max was read off)")

    _bars(axB, x, new, colors)
    axB.set_xticks(x, labels, fontsize=S.FS_TICK - 1)
    axB.set_ylabel("core max δT, 400–700 hPa  [K]")
    axB.set_title(f"corrected: restricted to the heated band "
                  f"({_BAND[0]:.0f}–{_BAND[1]:.0f} hPa)")

    for ax in (axA, axB):
        ax.axvline(1.5, color=S.C_GUIDE, lw=0.8, ls=":")
        ax.axvline(3.5, color=S.C_GUIDE, lw=0.8, ls=":")

    # (c) the r–p sections that explain the two bars
    p = np.asarray(layout.pressure_levels(), dtype=float)
    axC = [fig.add_subplot(gs[1, k]) for k in range(2)]
    for ax, (init, who) in zip(axC, _BAR_INITS):
        run = D.run("heating_moist", f"tseries_5K_120h_init{init}", ic=ic)
        d, c = _lead_paths(run, lead_hr)
        up, _ = io.load_delta_bundle(d)
        _cup, csfc = io.load_delta_bundle(c)
        dT = up[..., layout.upper_index("t")]
        r_km, sec = _azimuthal_mean(dT, csfc[:, :, -1], csfc[:, :, -2],
                                    r_max_km=600.0, dr_km=25.0)
        v = np.nanmax(np.abs(sec))
        pm = ax.pcolormesh(r_km, p, sec, cmap=S.CMAP_PV, vmin=-v, vmax=v,
                           shading="nearest")
        ax.contour(r_km, p, sec, levels=[0.0], colors="k", linewidths=0.8)
        ax.axhspan(_BAND[0], _BAND[1], color="k", alpha=0.07, lw=0)

        m = confinement_radii(str(run), lead_hr, p_band=_BAND)
        ax.plot(m["dT_all_max_r"], m["dT_all_max_p"], "kv", ms=11, mfc="none",
                mew=2.2, label=f"all-level max  {m['dT_all_max']:.2f} K "
                               f"@ {m['dT_all_max_p']:.0f} hPa")
        ax.plot(m["dT_band3_max_r"], m["dT_band3_max_p"], "ko", ms=10, mfc="none",
                mew=2.2, label=f"same max, {_BAND[0]:.0f}–{_BAND[1]:.0f} hPa only  "
                               f"{m['dT_band3_max']:.2f} K @ "
                               f"{m['dT_band3_max_p']:.0f} hPa")
        ax.axvline(278.0, color=S.C_GUIDE, lw=1.1, ls="--")
        ax.text(278.0, 205, " 2σ core ", fontsize=S.FS_ANNOT, style="italic",
                color=S.C_NOTE, rotation=90, va="top")
        ax.set_ylim(1000, 200)
        ax.set_xlabel("radius [km]")
        ax.set_title(f"{D.CASE[init]} · inject ΔT · azimuthal-mean δT\n"
                     f"(markers give the 3-D cell's location; shading is the "
                     f"azimuthal mean, so it reads lower)")
        ax.legend(loc="lower right", fontsize=S.FS_LEGEND - 0.5)
        fig.colorbar(pm, ax=ax, pad=0.015).set_label("δT [K]", fontsize=10)
    axC[0].set_ylabel("pressure [hPa]")

    S.panel_letters([axA, axB] + axC)
    fig.suptitle("LR1 — the H4 'core max δT' is a single cell taken over all 13 levels "
                 f"(5 K runs, nominal hour {lead_hr})\n{D.ic_note(ic, stat)}")
    S.caption(fig, "the heating peaks at 600 hPa and is zero at 200 hPa, yet the "
                   "all-level max is repeatedly won by a 100–250 hPa cell; restrict the "
                   "reduction to the heated band and the weak/strong ordering reverses",
              style)
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=None, help="default: figs/lr/lr1_metric_audit.png")
    a = ap.parse_args()
    out = a.out or (D.FIGS_ROOT / "lr" / "lr1_metric_audit.png")
    S.save(plot(a.lead, style=a.style, ic=a.ic, stat=a.stat), out,
           style=a.style, pdf=not a.no_pdf)
