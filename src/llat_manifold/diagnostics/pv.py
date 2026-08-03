"""PV–θ and wind-circulation radius–height cross-sections.

Faithful port of idealized_exp/plot_PV2_with_heating_profile_tengential.py and the
u–w quiver block of plot_PV2_tengential.py. Bold labels, the original colormaps/levels,
the left-hand heating-profile panel, and a dual altitude/pressure axis are preserved.

Two figures:
  * :func:`plot_pv_theta_cross_section` — PV shading (Reds absolute / bwr Δ) +
    isentropes + tangential-wind contours, with the heating-profile panel on the left.
  * :func:`plot_wind_circulation_cross_section` — a cleaner companion: tangential-wind
    speed shading + the in-plane (u, −w) secondary-circulation quiver.

PV is nonlinear in (u, v, t); for semi-linear δ bundles pass the matching control as
``baseline_path`` (with ``add_to_baseline=True``) to recover physical Δ-fields.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import io
from . import experiment_title, load_stamp
from . import _idealized as _id


def _heat_type_of(run_dir):
    cfg = load_stamp(run_dir).get("resolved_config", {})
    pert = cfg.get("perturbation", {}) or {}
    return pert.get("heat_type") if pert.get("type") == "heating" else None


def _load(delta_path, baseline_path, add_to_baseline):
    up, sfc = io.load_delta_bundle(delta_path)
    base = None
    if baseline_path is not None:
        bup, bsfc = io.load_delta_bundle(baseline_path)
        if add_to_baseline:
            # Reconstruct the absolute state ū+δ on BOTH upper and surface. The surface
            # must be added too — it carries Coriolis f and lat/lon, which are locked to
            # zero in a pure δ and would otherwise give a degenerate PV grid.
            up = up + bup
            sfc = sfc + bsfc
        base = (bup, bsfc)
    return up, sfc, base


def plot_pv_theta_cross_section(delta_path, out_png=None, *, baseline_path=None,
                                add_to_baseline=False, heat_type=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as pe

    delta_path = Path(delta_path)
    run_dir = delta_path.parent.parent
    if heat_type is None:
        heat_type = _heat_type_of(run_dir)

    up, sfc, base = _load(delta_path, baseline_path, add_to_baseline)
    u, v, t, z, w, f, lat2d, lon2d, p = _id.fields_from_bundle(up, sfc)
    pv = _id.calculate_pv_spherical(u, v, t, f, p, lat2d, lon2d)
    radius_2d, z_km, cy = _id.radius_height(z, lat2d, lon2d)
    theta = _id.theta_slice(t, p, cy)
    v_slice = v[:, cy, :]
    mean_z_km = np.nanmean(z_km, axis=1)
    # Ceiling at 200 hPa: above it the model's levels thin out and its own
    # error dominates the injected response (_idealized.P_TOP_HPA).
    z_min, z_max = float(np.nanmin(z_km)), _id.z_top_km(p, mean_z_km)

    # Layout: optional heating-profile panel on the left.
    fig = plt.figure(figsize=(13, 7) if heat_type else (12, 7))
    if heat_type:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 10], wspace=0.15)
        ax_prof = fig.add_subplot(gs[0])
        _id.add_heating_profile_panel(ax_prof, heat_type, p, mean_z_km, z_min, z_max)
        ax = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(1, 1, 1)

    if base is not None:
        bu, bv, bt, bz, bw, bf, blat, blon, bp = _id.fields_from_bundle(base[0], base[1])
        pv0 = _id.calculate_pv_spherical(bu, bv, bt, bf, bp, blat, blon)
        r0, z0, cy0 = _id.radius_height(bz, blat, blon)
        theta0 = _id.theta_slice(bt, bp, cy0)
        cf = ax.contourf(radius_2d, z_km, pv[:, cy, :] - pv0[:, cy0, :],
                         levels=np.linspace(-2, 2, 50), cmap="bwr", extend="both")
        cbar = fig.colorbar(cf, ax=ax, pad=0.10, aspect=30)
        cbar.set_ticks(np.arange(-2, 3, 1))
        cbar.set_label("Delta Potential Vorticity [PVU]", fontsize=12, weight="bold")
        # baseline (dashed) + perturbed (solid) isentropes
        ax.contour(r0, z0, theta0, levels=np.arange(290, 400, 4), colors="black",
                   linestyles="--", linewidths=2.0, alpha=0.8)
        dv = np.abs(v_slice) - np.abs(bv[:, cy0, :])
        cw = ax.contour(radius_2d, z_km, dv, levels=np.arange(-25, 25, 2),
                        colors="limegreen", linewidths=2.0, alpha=0.8,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.clabel(cw, inline=True, fontsize=10, fmt="%+1.0f m/s", rightside_up=True)
    else:
        cf = ax.contourf(radius_2d, z_km, pv[:, cy, :], levels=np.linspace(0, 8, 9),
                         cmap="Reds", extend="max")
        cbar = fig.colorbar(cf, ax=ax, pad=0.10, aspect=30)
        cbar.set_ticks(np.arange(0, 9, 2))
        cbar.set_label("Potential Vorticity [PVU]", fontsize=12, weight="bold")
        cw = ax.contour(radius_2d, z_km, np.abs(v_slice), levels=np.arange(0, 80, 5),
                        colors="deepskyblue", linewidths=2.0, alpha=0.8)
        ax.clabel(cw, inline=True, fontsize=10, fmt="%1.0f m/s", rightside_up=True)

    cs = ax.contour(radius_2d, z_km, theta, levels=np.arange(290, 400, 4),
                    colors="black", linewidths=2.0, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=10, fmt="%1.0f K", rightside_up=True)

    _finish_axes(ax, fig, z_min, z_max, p, mean_z_km, ticker,
                 title=experiment_title(run_dir, prefix="PV–θ:"))
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[pv] wrote {out_png}")
    return fig


def plot_wind_circulation_cross_section(delta_path, out_png=None, *, heat_type=None):
    """Companion figure: tangential-wind shading + (u, −w) secondary-circulation quiver."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec

    delta_path = Path(delta_path)
    run_dir = delta_path.parent.parent
    if heat_type is None:
        heat_type = _heat_type_of(run_dir)

    up, sfc, _ = _load(delta_path, None, False)
    u, v, t, z, w, f, lat2d, lon2d, p = _id.fields_from_bundle(up, sfc)
    radius_2d, z_km, cy = _id.radius_height(z, lat2d, lon2d)
    mean_z_km = np.nanmean(z_km, axis=1)
    # Ceiling at 200 hPa: above it the model's levels thin out and its own
    # error dominates the injected response (_idealized.P_TOP_HPA).
    z_min, z_max = float(np.nanmin(z_km)), _id.z_top_km(p, mean_z_km)

    u_slice = u[:, cy, :]
    w_slice = -w[:, cy, :] * 5.0     # scale ω for visibility (as in the original quiver)
    vt = np.abs(v[:, cy, :])

    fig = plt.figure(figsize=(13, 7) if heat_type else (12, 7))
    if heat_type:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 10], wspace=0.15)
        ax_prof = fig.add_subplot(gs[0])
        _id.add_heating_profile_panel(ax_prof, heat_type, p, mean_z_km, z_min, z_max)
        ax = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(1, 1, 1)

    cf = ax.contourf(radius_2d, z_km, vt, levels=np.arange(0, 82, 4),
                     cmap="YlOrRd", extend="max")
    cbar = fig.colorbar(cf, ax=ax, pad=0.10, aspect=30)
    cbar.set_label("Tangential Wind Speed [m s$^{-1}$]", fontsize=12, weight="bold")

    sk = (slice(None), slice(None, None, 3))     # thin the arrows west-east
    q = ax.quiver(radius_2d[sk], z_km[sk], u_slice[sk], w_slice[sk],
                  scale=500, width=0.002, headwidth=3, headlength=4,
                  color="navy", alpha=0.8)
    ax.quiverkey(q, X=0.85, Y=-0.13, U=10, label="10 m s$^{-1}$", labelpos="E",
                 coordinates="axes", fontproperties={"size": 11})
    ax.quiverkey(q, X=0.95, Y=-0.13, U=10, label="1 Pa s$^{-1}$", angle=90,
                 labelpos="E", coordinates="axes", fontproperties={"size": 11})

    _finish_axes(ax, fig, z_min, z_max, p, mean_z_km, ticker,
                 title=experiment_title(run_dir, prefix="Tangential wind & secondary circ.:"))
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[pv] wrote {out_png}")
    return fig


def _finish_axes(ax, fig, z_min, z_max, p, mean_z_km, ticker, *, title):
    ax.set_ylim(z_min, z_max)
    ax.set_xlim(-750, 750)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlabel("Radius from Storm Center [km]", fontsize=13, weight="bold")
    ax.set_ylabel("Altitude [km]", fontsize=13, weight="bold")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(mean_z_km)
    ax2.set_yticklabels([f"{int(pp)}" for pp in p])
    ax2.set_ylabel("Pressure [hPa]", fontsize=13, weight="bold", color="#333333")


# Back-compat alias.
def theta_cross_section(delta_path, out_png=None, **kw):
    return plot_pv_theta_cross_section(delta_path, out_png=out_png)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PV–θ / wind-circulation cross-sections")
    ap.add_argument("delta_npz")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--add-to-baseline", action="store_true")
    ap.add_argument("--wind", action="store_true", help="draw the wind-circulation companion")
    ap.add_argument("--out", default="PV_Theta_tengential.png")
    a = ap.parse_args()
    if a.wind:
        plot_wind_circulation_cross_section(a.delta_npz, a.out)
    else:
        plot_pv_theta_cross_section(a.delta_npz, a.out, baseline_path=a.baseline,
                                    add_to_baseline=a.add_to_baseline)
