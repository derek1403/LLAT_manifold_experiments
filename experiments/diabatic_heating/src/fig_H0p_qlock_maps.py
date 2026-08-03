"""H0′ — under q-lock, does the response collapse, or just shrink?

6×3 structure panel for the δq = 0 strong-vortex sweep. Columns are amplitudes
(0.5/2/4/6/8/10 K); rows are the ΔPV field on each column's own upper-min layer,
its low-max layer, and the azimuthal-mean radius–height section. One symmetric
colour scale per row. The 2σ heated core is circled on the map rows and marked on
the r–z row; the extremum is an ×.

The point: even with δq ≡ 0 the pattern stays textbook — a compact low-level
positive core, an upper negative lobe, an outer wave train. What the lock removes
is amplitude and its later growth, not the shape of the balance adjustment. So the
instantaneous adjustment is dry; the nonlinear amplification and maintenance are
moist. And because the shape survives (rather than the field scrambling), this is a
clean physical weakening, not the numerical breakdown H1 worries about.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S


def _core_circle(ax, lat2d, lon2d, sigma, **kw):
    """Draw the 2σ heated-core edge as a true circle in lon/lat."""
    cy, cx = lat2d.shape[0] // 2, lat2d.shape[1] // 2
    clat, clon = lat2d[cy, cx], lon2d[cy, cx]
    r_deg = 2 * sigma * 0.25                       # 2σ grid points × 0.25° spacing
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(clon + r_deg * np.cos(t) / np.cos(np.deg2rad(clat)),
            clat + r_deg * np.sin(t), **kw)


def plot(init: str = D.STRONG, amps=(0.5, 2, 4, 6, 8, 10), lead_hr=24,
         zoom_deg=5.0, style: str = "note", prefix: str = "sweepq",
         suptitle: str | None = None, caption: str | None = None, *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    from llat_manifold.diagnostics import _idealized as _id
    from llat_manifold.diagnostics.response import (
        analyze_pair, _azimuthal_mean, _find_runs)
    from llat_manifold.perturbations.heating import _amp_tag

    cols = []
    for amp in amps:
        name = f"{prefix}_{_amp_tag(amp)}_{lead_hr}h_init{init}"
        run = _find_runs(str(D.category(ic)), name)[0]
        d = sorted((run / "data").glob(f"delta_continuous_*lead{lead_hr:03d}hr.npz"))[0]
        c = d.parent / d.name.replace("delta_", "control_")
        m, aux = analyze_pair(d, c, sigma=5.0, stat=stat)
        r_km, az = _azimuthal_mean(aux["dpv"], aux["lat2d"], aux["lon2d"])
        cols.append({"amp": amp, "m": m, "dpv": aux["dpv"], "az": az, "r_km": r_km,
                     "z_km": np.nanmean(aux["z"], axis=(1, 2)) / 1000.0,
                     "p_hpa": aux["p_hpa"],
                     "lat2d": aux["lat2d"], "lon2d": aux["lon2d"]})

    ny, nx = cols[0]["lat2d"].shape
    cy, cx = ny // 2, nx // 2
    half = int(round(zoom_deg / 0.25))
    ys, xs = slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1)

    def _lim(fields):
        return max(float(np.nanmax(np.abs(f))) for f in fields) or 1.0
    lim_up = _lim([c["dpv"][c["m"]["upperlevel_kji"][0]][ys, xs] for c in cols])
    lim_low = _lim([c["dpv"][c["m"]["lowlevel_kji"][0]][ys, xs] for c in cols])
    lim_az = _lim([c["az"] for c in cols])

    fig, axes = plt.subplots(3, len(cols), figsize=(3.15 * len(cols), 10.6),
                             gridspec_kw={"height_ratios": [1, 1, 1.35]})
    tag_up = "min" if stat == "max" else stat
    tag_low = "max" if stat == "max" else stat
    rows = [("upperlevel", tag_up, lim_up), ("lowlevel", tag_low, lim_low)]
    for j, col in enumerate(cols):
        lat, lon, m = col["lat2d"], col["lon2d"], col["m"]
        for i, (pole, tag, lim) in enumerate(rows):
            ax = axes[i, j]
            k, pj, pi = m[f"{pole}_kji"]
            ax.pcolormesh(lon[ys, xs], lat[ys, xs], col["dpv"][k][ys, xs],
                          cmap=S.CMAP_PV, vmin=-lim, vmax=lim, shading="auto")
            _core_circle(ax, lat, lon, 5.0, color="k", lw=0.9, ls="--", alpha=0.7)
            ax.plot(lon[pj, pi], lat[pj, pi], "x", color="k", ms=7, mew=1.8,
                    markerfacecolor="white")
            ax.set_title(f"{col['amp']:g} K · {m[f'{pole}_p']:.0f} hPa ({tag})",
                         fontsize=S.FS_TICK - 1)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=S.FS_TICK - 3)
            if j:
                ax.set_yticklabels([])
        ax = axes[2, j]
        ax.contourf(col["r_km"], col["z_km"], col["az"],
                    levels=np.linspace(-lim_az, lim_az, 41), cmap=S.CMAP_PV, extend="both")
        core_km = 2 * 5.0 * 0.25 * 111.32 * np.cos(np.deg2rad(lat[cy, cx]))
        ax.axvline(core_km, color="k", lw=0.9, ls="--", alpha=0.7)
        ax.set_ylim(0, _id.z_top_km(col["p_hpa"], col["z_km"]))
        ax.set_title(f"{col['amp']:g} K · azimuthal mean", fontsize=S.FS_TICK - 1)
        ax.tick_params(labelsize=S.FS_TICK - 3)
        ax.set_xlabel("radius [km]", fontsize=S.FS_TICK - 2)
        if j:
            ax.set_yticklabels([])
    axes[0, 0].set_ylabel("lat [°]  (upper ΔPV)", fontsize=S.FS_TICK - 1)
    axes[1, 0].set_ylabel("lat [°]  (low-level ΔPV)", fontsize=S.FS_TICK - 1)
    axes[2, 0].set_ylabel("altitude [km]", fontsize=S.FS_TICK - 1)

    for i, lim in enumerate((lim_up, lim_low, lim_az)):
        sm = plt.cm.ScalarMappable(cmap=S.CMAP_PV, norm=plt.Normalize(-lim, lim))
        cb = fig.colorbar(sm, ax=list(axes[i]), pad=0.008, fraction=0.02)
        cb.set_label("ΔPV  [PVU]", fontsize=S.FS_TICK - 1, weight="bold")

    fig.suptitle(suptitle or (f"H0′ — δq = 0 sweep structure: shape survives, amplitude dies  "
                              f"({D.CASE[init]}, hour {lead_hr})"))
    S.caption(fig, caption if caption is not None else (
        "dashed circle = 2σ heated core, × = where the reduction sits; even with δq = 0 "
        "the pattern is "
        "textbook (compact low core + upper lobe + outer wave train) — the lock takes "
        "amplitude, not the shape of the balance adjustment"), style)
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/h0p_qlock_maps.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "h0p_qlock_maps.png"
    S.save(plot(a.init, style=a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
