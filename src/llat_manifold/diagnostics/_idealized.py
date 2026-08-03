"""Ported core of the idealized-experiment cross-section plots.

Faithful port of the spherical PV and divergence computations and the radius–height
cross-section styling from
``/wk2/pc/AI_models/RegionalCouple_AI/idealized_exp/plot_PV2_tengential.py`` and
``plot_div_uw.py`` (configs `div_Theta_uv_..._True.png`, `PV_Theta_tengential_...png`).

Operates on the DLAMPty arrays exactly as those scripts do:
  upper (13, 81, 81, 6) with channels u=0, v=1, t=2, q=3, z=4 (geopotential), w=5;
  surface (81, 81, 20) with Coriolis f=10, lon=−2, lat=−1.
A ``baseline`` (0 K / control) bundle may be supplied to plot perturbation (Δ) fields.
"""
from __future__ import annotations

import numpy as np

from .. import layout

G = 9.80665
RD = 287.0
CP = 1004.0
A_EARTH = 6371000.0


# --------------------------------------------------------------------------- #
# Physics (ported verbatim in substance)
# --------------------------------------------------------------------------- #
def calculate_pv_spherical(u, v, t, f, p_hpa, lat_2d, lon_2d):
    """Ertel PV on isobaric levels over a 0.25° spherical grid, in PVU."""
    p_pa = np.array(p_hpa) * 100.0
    p_3d = p_pa[:, None, None]
    theta = t * (100000.0 / p_3d) ** (RD / CP)

    lat_rad = np.deg2rad(lat_2d)
    lon_rad = np.deg2rad(lon_2d)
    if np.abs(lat_2d[-1, 0] - lat_2d[0, 0]) > np.abs(lat_2d[0, -1] - lat_2d[0, 0]):
        y2, x2 = 0, 1
    else:
        y2, x2 = 1, 0
    y3, x3 = y2 + 1, x2 + 1

    dy = A_EARTH * np.gradient(lat_rad, axis=y2)
    dx = A_EARTH * np.cos(lat_rad) * np.gradient(lon_rad, axis=x2)
    dy3 = np.where(dy[None] == 0, 1e-10, dy[None])
    dx3 = np.where(dx[None] == 0, 1e-10, dx[None])

    dtheta_dp = np.gradient(theta, p_pa, axis=0)
    du_dp = np.gradient(u, p_pa, axis=0)
    dv_dp = np.gradient(v, p_pa, axis=0)
    dtheta_dy = np.gradient(theta, axis=y3) / dy3
    du_dy = np.gradient(u, axis=y3) / dy3
    dtheta_dx = np.gradient(theta, axis=x3) / dx3
    dv_dx = np.gradient(v, axis=x3) / dx3

    zeta = dv_dx - du_dy
    term1 = dv_dp * dtheta_dx - du_dp * dtheta_dy
    term2 = (zeta + f[None]) * dtheta_dp
    return -G * (term1 + term2) * 1e6


def calculate_divergence(u, v, lat_2d, lon_2d):
    """Spherical horizontal divergence, in 10⁻⁵ s⁻¹ (positive = divergence)."""
    lat_rad = np.deg2rad(lat_2d)
    lon_rad = np.deg2rad(lon_2d)
    if np.abs(lat_2d[-1, 0] - lat_2d[0, 0]) > np.abs(lat_2d[0, -1] - lat_2d[0, 0]):
        y2, x2 = 0, 1
    else:
        y2, x2 = 1, 0
    y3, x3 = y2 + 1, x2 + 1
    dy = A_EARTH * np.gradient(lat_rad, axis=y2)
    dx = A_EARTH * np.cos(lat_rad) * np.gradient(lon_rad, axis=x2)
    dy3 = np.where(dy[None] == 0, 1e-10, dy[None])
    dx3 = np.where(dx[None] == 0, 1e-10, dx[None])
    du_dx = np.gradient(u, axis=x3) / dx3
    dv_dy = np.gradient(v, axis=y3) / dy3
    return (du_dx + dv_dy) * 1e5


# --------------------------------------------------------------------------- #
# Field extraction from a bundle
# --------------------------------------------------------------------------- #
def fields_from_bundle(upper, sfc):
    """Pull (u, v, t, z[m], w, f, lat_2d, lon_2d, p_hpa) from DLAMPty arrays."""
    u = upper[:, :, :, layout.upper_index("u")]
    v = upper[:, :, :, layout.upper_index("v")]
    t = upper[:, :, :, layout.upper_index("t")]
    z = upper[:, :, :, layout.upper_index("z")] / G       # geopotential -> height (m)
    w = upper[:, :, :, layout.upper_index("w")]
    f = sfc[:, :, layout.surface_index("f")]
    lon_2d = sfc[:, :, -2]
    lat_2d = sfc[:, :, -1]
    return u, v, t, z, w, f, lat_2d, lon_2d, layout.pressure_levels()


def _center_slice(arr3d, lat_2d, lon_2d):
    """West–east slice through the domain centre: returns (radius_2d_km, z?, slice)."""
    cy = lat_2d.shape[0] // 2
    return arr3d[:, cy, :], cy


def radius_height(z, lat_2d, lon_2d):
    """Build (radius_2d_km, z_km_2d) along the central west–east cross-section."""
    cy = lat_2d.shape[0] // 2
    cx = lat_2d.shape[1] // 2
    z_km = z[:, cy, :] / 1000.0
    lon_slice = lon_2d[cy, :]
    center_lat = lat_2d[cy, cx]
    center_lon = lon_2d[cy, cx]
    radius_km = (lon_slice - center_lon) * 111.32 * np.cos(np.deg2rad(center_lat))
    radius_2d = np.tile(radius_km, (z_km.shape[0], 1))
    return radius_2d, z_km, cy


def theta_slice(t, p_hpa, cy):
    p_2d = np.array(p_hpa)[:, None]
    return t[:, cy, :] * (1000.0 / p_2d) ** (RD / CP)


# Plot ceiling for every vertical section in the repo. Above ~200 hPa the model's
# own error grows and the levels thin out (250/200/150/100/50 hPa), so anything
# drawn up there is dominated by model noise rather than the response we injected.
# Single source of truth: change it here and every r–z / r–p figure follows.
P_TOP_HPA = 200.0


def z_top_km(p_hpa, mean_z_km) -> float:
    """Altitude [km] of :data:`P_TOP_HPA`, for figures whose y axis is height.

    ``mean_z_km`` is the domain-mean geopotential height of each pressure level, so
    the cap lands on the same physical surface as a pressure-axis figure's 200 hPa.
    """
    p = np.asarray(p_hpa, dtype=float)
    z = np.asarray(mean_z_km, dtype=float)
    order = np.argsort(p)                       # np.interp needs ascending x
    return float(np.interp(P_TOP_HPA, p[order], z[order]))


def heating_vertical_profile(heat_type: str):
    """The Deep/Shallow/Stratiform V(P) curve + a LaTeX title (port of the panel)."""
    p_plot = np.linspace(200, 1000, 100)
    pn = (p_plot - 200) / 800.0
    if heat_type == "Deep":
        v = np.sin(pn * np.pi)
        title = r"$\sin\left(\frac{P-200}{800}\pi\right)$"
    elif heat_type == "Stratiform":
        v = np.sin(2 * pn * np.pi)
        title = r"$\sin\left(2\frac{P-200}{800}\pi\right)$"
    elif heat_type == "Shallow":
        v = (np.sin(pn * np.pi) - np.sin(2 * pn * np.pi)) / np.sqrt(2)
        title = "Shallow"
    else:
        v = np.zeros_like(p_plot)
        title = ""
    return p_plot, v, title


def add_heating_profile_panel(ax_prof, heat_type, p_hpa, mean_z_km, z_min, z_max):
    """Draw the left-hand vertical heating-profile panel, aligned to the main axis.

    Faithful port of plot_PV2_with_heating_profile_tengential.py: the profile's Y axis
    is the same physical height as the cross-section, with pressure tick labels.
    """
    p_plot, v_p, title_str = heating_vertical_profile(heat_type)
    z_plot = np.interp(p_plot, p_hpa, mean_z_km)
    ax_prof.plot(np.insert(v_p, 0, 0), np.insert(z_plot, 0, 21),
                 color="tab:blue", linewidth=2)
    ax_prof.set_ylim(z_min, z_max)
    ax_prof.set_xlim(-1.2, 1.2)
    ax_prof.set_xticks([-1, 0, 1])
    ax_prof.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_prof.grid(True, linestyle="--", alpha=0.5)
    ax_prof.set_yticks(mean_z_km)
    ax_prof.set_yticklabels([f"{int(p)}" for p in p_hpa])
    ax_prof.set_ylabel("Pressure [hPa]", fontsize=11, weight="bold", color="#333333")
    ax_prof.set_xlabel("Heating", fontsize=11, weight="bold")
    ax_prof.set_title(title_str, fontsize=10, pad=10)
    ax_prof.tick_params(axis="both", which="major", labelsize=8)
