"""Divergence–θ radius–height cross-section (ported from idealized_exp/plot_div_uw.py).

Spherical horizontal divergence (10⁻⁵ s⁻¹) on the central west–east cross-section,
overlaid with isentropes — the secondary-circulation / in-&-outflow signature of the
perturbation (config `div_Theta_uv_..._True.png`). A ``baseline`` gives Δ-divergence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import io, layout
from . import experiment_title, load_stamp
from . import _idealized as _id


def _heat_type_of(run_dir):
    cfg = load_stamp(run_dir).get("resolved_config", {})
    pert = cfg.get("perturbation", {}) or {}
    return pert.get("heat_type") if pert.get("type") == "heating" else None

# DLAMPty sub-domain spacing in metres (kept for the simple gradient helper below).
_DX_M = 0.25 * 111_000.0


def horizontal_divergence(dup: np.ndarray) -> np.ndarray:
    """Quick Cartesian ∂u/∂x+∂v/∂y per level (full spherical version in _idealized)."""
    u = dup[:, :, :, layout.upper_index("u")]
    v = dup[:, :, :, layout.upper_index("v")]
    return np.gradient(u, _DX_M, axis=2) + np.gradient(v, _DX_M, axis=1)


def plot_div_theta_cross_section(delta_path, out_png=None, *, baseline_path=None,
                                 add_to_baseline=False, heat_type=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec

    delta_path = Path(delta_path)
    run_dir = delta_path.parent.parent
    if heat_type is None:
        heat_type = _heat_type_of(run_dir)
    up, sfc = io.load_delta_bundle(delta_path)
    base = None
    if baseline_path is not None:
        bup, bsfc = io.load_delta_bundle(baseline_path)
        if add_to_baseline:
            # Add the background to BOTH fields: the surface carries lat/lon used for
            # the spherical metric (locked to zero in a pure δ).
            up = up + bup
            sfc = sfc + bsfc
        base = (bup, bsfc)

    u, v, t, z, w, f, lat2d, lon2d, p = _id.fields_from_bundle(up, sfc)
    div = _id.calculate_divergence(u, v, lat2d, lon2d)
    radius_2d, z_km, cy = _id.radius_height(z, lat2d, lon2d)
    theta = _id.theta_slice(t, p, cy)
    z_min, z_max = float(np.nanmin(z_km)), 15.0
    mean_z_km = np.nanmean(z_km, axis=1)

    fig = plt.figure(figsize=(13, 7) if heat_type else (12, 7))
    if heat_type:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 10], wspace=0.15)
        ax_prof = fig.add_subplot(gs[0])
        _id.add_heating_profile_panel(ax_prof, heat_type, p, mean_z_km, z_min, z_max)
        ax = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(1, 1, 1)
    field = div[:, cy, :]
    label = "Divergence [10$^{-5}$ s$^{-1}$]"
    if base is not None:
        bu, bv, bt, bz, bw, bf, blat, blon, bp = _id.fields_from_bundle(base[0], base[1])
        div0 = _id.calculate_divergence(bu, bv, blat, blon)
        field = field - div0[:, blat.shape[0] // 2, :]
        label = "Δ " + label
    # Fixed, round colorbar (10^-5 s^-1) so plots from different ICs are comparable.
    vmax = 10.0
    cf = ax.contourf(radius_2d, z_km, np.clip(field, -vmax, vmax),
                     levels=np.arange(-vmax, vmax + 0.001, 1.0), cmap="bwr_r", extend="both")
    cbar = fig.colorbar(cf, ax=ax, pad=0.10, aspect=30)
    cbar.set_ticks(np.arange(-vmax, vmax + 1, 2))
    cbar.set_label(label, fontsize=12, weight="bold")

    cs = ax.contour(radius_2d, z_km, theta, levels=np.arange(290, 400, 4),
                    colors="black", linewidths=1.6, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=9, fmt="%1.0f K")

    ax.set_ylim(z_min, z_max)
    ax.set_xlim(-750, 750)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlabel("Radius from Storm Center [km]", fontsize=13, weight="bold")
    ax.set_ylabel("Altitude [km]", fontsize=13, weight="bold")
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(mean_z_km)
    ax2.set_yticklabels([f"{int(pp)}" for pp in p])
    ax2.set_ylabel("Pressure [hPa]", fontsize=13, weight="bold", color="#333333")
    ax.set_title(experiment_title(run_dir, prefix="Divergence–θ:"), fontsize=13, weight="bold")

    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[divergence] wrote {out_png}")
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="divergence–θ cross-section of a bundle")
    ap.add_argument("delta_npz")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--add-to-baseline", action="store_true")
    ap.add_argument("--out", default="div_Theta_uv.png")
    a = ap.parse_args()
    plot_div_theta_cross_section(a.delta_npz, a.out, baseline_path=a.baseline,
                                 add_to_baseline=a.add_to_baseline)
