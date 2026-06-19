"""Axisymmetric wind-balance profiles — LLAT tangential vs gradient vs geostrophic.

Faithful port of ``idealized_exp/wind_r/wind_r_LLAT.py`` (reference figures
``2025091900_LLAT_wind_profile_024h.png`` / ``2025092000_..._024h.png``). For a bundle
it builds the azimuthal-mean tangential wind at the surface, 850 and 500 hPa and
compares it against the diagnostic gradient-wind and geostrophic-wind profiles derived
from the mean pressure/geopotential — a direct test of how close the model's vortex
sits to gradient-wind balance (the expected TC balance) versus geostrophic balance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .. import io, layout
from . import experiment_title

OMEGA = 7.2921e-5
RD = 287.05
R_EARTH = 6371000.0


def _radial_mean(var_2d, r_2d, bins, r_1d):
    idx = np.digitize(r_2d, bins)
    out = np.zeros(len(r_1d))
    for i in range(1, len(bins)):
        m = (idx == i)
        out[i - 1] = np.nanmean(var_2d[m]) if np.any(m) else np.nan
    valid = ~np.isnan(out)
    if not np.all(valid) and np.any(valid):
        out = np.interp(np.arange(len(out)), np.arange(len(out))[valid], out[valid])
    return out


def _winds_surface(P, T, r, f):
    rho = P / (RD * T)
    dP_dr = np.gradient(P, r)
    vg = (1.0 / (f * rho)) * dP_dr
    vgr = 0.5 * (-r * f + np.sqrt(np.maximum((r * f) ** 2 + 4 * (r / rho) * dP_dr, 0)))
    return vg, vgr


def _winds_isobaric(Phi, r, f):
    dPhi_dr = np.gradient(Phi, r)
    vg = (1.0 / f) * dPhi_dr
    vgr = 0.5 * (-r * f + np.sqrt(np.maximum((r * f) ** 2 + 4 * r * dPhi_dr, 0)))
    return vg, vgr


def tangential_radial(dup):
    """Azimuthal-mean (V_t, V_r) vs (level, radius) — kept for general use."""
    u = dup[:, :, :, layout.upper_index("u")]
    v = dup[:, :, :, layout.upper_index("v")]
    nz, ny, nx = u.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    theta = np.arctan2(yy - cy, xx - cx)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    nb = int(min(cy, cx))
    rbin = np.clip(r.astype(int), 0, nb - 1).ravel()
    vt = np.zeros((nz, nb)); vr = np.zeros((nz, nb)); cnt = np.zeros(nb)
    np.add.at(cnt, rbin, 1.0)
    for k in range(nz):
        np.add.at(vt[k], rbin, (-u[k] * np.sin(theta) + v[k] * np.cos(theta)).ravel())
        np.add.at(vr[k], rbin, (u[k] * np.cos(theta) + v[k] * np.sin(theta)).ravel())
    cnt = np.maximum(cnt, 1.0)
    return vt / cnt, vr / cnt


def plot_wind_balance_profile(delta_path, out_png=None):
    """LLAT vs gradient vs geostrophic tangential-wind profiles at sfc/850/500 hPa."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta_path = Path(delta_path)
    run_dir = delta_path.parent.parent
    upper, sfc = io.load_delta_bundle(delta_path)

    ny, nx = sfc.shape[0], sfc.shape[1]
    ic, jc = ny // 2, nx // 2
    f_center = sfc[:, :, layout.surface_index("f")][ic, jc]
    center_lat = np.degrees(np.arcsin(np.clip(f_center / (2 * OMEGA), -1, 1)))
    dx = 0.25 * (np.pi / 180.0) * R_EARTH * np.cos(np.radians(center_lat))
    dy = 0.25 * (np.pi / 180.0) * R_EARTH

    y_idx, x_idx = np.indices((ny, nx))
    x_dist = (x_idx - jc) * dx
    y_dist = (ic - y_idx) * dy
    r_2d = np.sqrt(x_dist ** 2 + y_dist ** 2)
    theta_2d = np.arctan2(y_dist, x_dist)

    lev = layout.pressure_levels()
    k850, k500 = lev.index(850), lev.index(500)
    u10 = sfc[:, :, layout.surface_index("u10")]
    v10 = sfc[:, :, layout.surface_index("v10")]
    t2m = sfc[:, :, layout.surface_index("t2m")]
    sp = sfc[:, :, layout.surface_index("sp")]
    ui, vi, zi = layout.upper_index("u"), layout.upper_index("v"), layout.upper_index("z")
    u850, v850, z850 = upper[k850, :, :, ui], upper[k850, :, :, vi], upper[k850, :, :, zi]
    u500, v500, z500 = upper[k500, :, :, ui], upper[k500, :, :, vi], upper[k500, :, :, zi]

    vt10 = -u10 * np.sin(theta_2d) + v10 * np.cos(theta_2d)
    vt850 = -u850 * np.sin(theta_2d) + v850 * np.cos(theta_2d)
    vt500 = -u500 * np.sin(theta_2d) + v500 * np.cos(theta_2d)

    dr = 10000.0
    bins = np.arange(0, r_2d.max() + dr, dr)
    r_1d = 0.5 * (bins[:-1] + bins[1:])
    r_km = r_1d / 1000.0
    rm = lambda a: _radial_mean(a, r_2d, bins, r_1d)  # noqa: E731

    vt10_1d, sp_1d, t2m_1d = rm(vt10), rm(sp), rm(t2m)
    vt850_1d, z850_1d = rm(vt850), rm(z850)
    vt500_1d, z500_1d = rm(vt500), rm(z500)
    r_safe = np.where(r_1d == 0, 1e-5, r_1d)

    vg_850, vgr_850 = _winds_isobaric(z850_1d, r_safe, f_center)
    vg_500, vgr_500 = _winds_isobaric(z500_1d, r_safe, f_center)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(r_km, vt10_1d, color="black", ls="-", lw=2, label="LLAT ($v_t$) - Sfc")
    ax.plot(r_km, vt850_1d, color="black", ls="--", lw=2, label="LLAT ($v_t$) - 850hPa")
    ax.plot(r_km, gaussian_filter(vgr_850, 2), color="blue", ls="--", lw=2,
            label="Gradient ($v_{gr}$) - 850hPa")
    ax.plot(r_km, gaussian_filter(vg_850, 2), color="red", ls="--", lw=2,
            label="Geostrophic ($v_g$) - 850hPa")
    ax.plot(r_km, gaussian_filter(vt500_1d, 2), color="black", ls="-.", lw=2,
            label="LLAT ($v_t$) - 500hPa")
    ax.plot(r_km, gaussian_filter(vgr_500, 2), color="blue", ls="-.", lw=2,
            label="Gradient ($v_{gr}$) - 500hPa")
    ax.plot(r_km, gaussian_filter(vg_500, 2), color="red", ls="-.", lw=2,
            label="Geostrophic ($v_g$) - 500hPa")

    ax.set_title(experiment_title(run_dir, prefix="Axisymmetric wind balance:"),
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Radius (km)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tangential Wind Speed (m/s)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, r_km.max())
    ax.set_ylim(0, 50)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", fontsize=11, ncol=3)
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[wind_profile] wrote {out_png}")
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LLAT/gradient/geostrophic wind profile")
    ap.add_argument("delta_npz")
    ap.add_argument("--out", default="LLAT_wind_profile.png")
    a = ap.parse_args()
    plot_wind_balance_profile(a.delta_npz, a.out)
