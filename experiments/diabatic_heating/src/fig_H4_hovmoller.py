"""H4 — where does the re-moistening come from? The moisture harvesting loop.

Radius–time Hovmöller of the azimuthal-mean column latent energy Lv·∫δq dm, one
panel per run. Green = moist anomaly, brown = dry. Inward-sloping green streaks =
moisture converging from the environment into the core; a drying (brown) collar
outside the core = the circulation harvesting environmental moisture.

For the strong vortex the core stays moist for five days while the surroundings dry
— the harvesting loop that sustains the response after the injected moisture is
long gone. The heated-core edge (2σ) is marked at the top axis and the injection
window is shaded, so neither label sits on the data.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

# The three 5-day runs that share the moisture story (strong vortex).
_RUNS = [
    ("heating_moist", "tseries_5K_120h_init{init}", "inject ΔT — q free"),
    ("dq_latent", "tseriesdql_5K_120h_init{init}", "inject δq — latent-equivalent"),
    ("dq_measured", "tseriesdq_5K_120h_init{init}", "inject δq — measured"),
]


def _hov(run_dir, dr_km=25.0, r_max_km=900.0):
    """(hours, r_km, hov[time,radius], core_km) of column latent energy."""
    from llat_manifold.diagnostics.response import (
        _continuous_pairs, _azimuthal_mean, _mass_weights, _pert_params)
    from llat_manifold import io, layout
    pairs = _continuous_pairs(run_dir)
    up0, sfc0 = io.load_delta_bundle(pairs[0][2])
    lat2d, lon2d = sfc0[:, :, -1], sfc0[:, :, -2]
    dm = _mass_weights(up0.shape[0])
    q_idx = layout.upper_index("q")
    sigma = float(_pert_params(run_dir)["sigma"])
    cy, cx = lat2d.shape[0] // 2, lat2d.shape[1] // 2
    core_km = abs((lon2d[cy, cx + int(2 * sigma)] - lon2d[cy, cx])
                  * 111.32 * np.cos(np.deg2rad(lat2d[cy, cx])))
    hours, rows, r_km = [], [], None
    for lead, d, _c in pairs:
        up, _ = io.load_delta_bundle(d)
        e_lat = 2.5e6 * np.einsum("kyx,k->yx", up[..., q_idx], dm) / 1e6
        r_km, az = _azimuthal_mean(e_lat[None], lat2d, lon2d, r_max_km=r_max_km, dr_km=dr_km)
        hours.append(lead)
        rows.append(az[0])
    return np.asarray(hours), r_km, np.asarray(rows), core_km


def plot(init: str = D.STRONG, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    panels = [(_hov(str(D.run(fam, tag.format(init=init), ic=ic))), lab) for fam, tag, lab in _RUNS]
    vmax = max(np.nanpercentile(np.abs(hov), 98) for (_, _, hov, _), _ in panels)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.6), sharey=True)
    for ax, ((hours, r_km, hov, core_km), lab) in zip(axes, panels):
        pm = ax.pcolormesh(r_km, hours, hov, cmap=S.CMAP_Q, vmin=-vmax, vmax=vmax,
                           shading="nearest")
        t_end = 24
        ax.axhspan(hours[0], t_end, color="white", alpha=0.0)          # keep limits
        ax.axhline(t_end, color="k", lw=1.0, ls=":")
        ax.axvline(core_km, color="k", lw=1.2, ls="--")
        ax.set_title(lab)
        ax.set_xlabel("radius [km]")
    axes[0].set_ylabel("Iteration n  (hour)")
    # Two annotations, on panel (a) only, inside the field with a white halo.
    axes[0].text(core_km + 15, hours[-1] * 0.97, "2σ core", ha="left", va="top",
                 fontsize=S.FS_ANNOT, style="italic", rotation=90,
                 bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.7))
    axes[0].text(0.98, t_end, " forcing ends ", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=S.FS_ANNOT, style="italic",
                 bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.7))
    S.panel_letters(axes)
    cb = fig.colorbar(pm, ax=list(axes), pad=0.012, shrink=0.9)
    cb.set_label("azimuthal-mean column latent energy  Lv·∫δq dm  [MJ m$^{-2}$]",
                 weight="bold")
    fig.suptitle(f"H4 — moisture Hovmöller: where does the re-moistening come from?  "
                 f"({D.CASE[init]})")
    S.caption(fig, "green = moist, brown = dry; inward-sloping green streaks with a brown collar "
                   "outside the 2σ core = the circulation harvesting environmental moisture", style)
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/h4_moisture_hovmoller.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "h4_moisture_hovmoller.png"
    S.save(plot(a.init, a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
