"""Analytic ring vortex in gradient-wind and hydrostatic balance.

Construction (all on a fine 1-D radial grid, then interpolated to the 2-D domain):

  1. ζ(r): three-region smoothed vorticity ring after Schubert et al. (1999) /
     Hendricks et al. (2009) — ``γ·ζ_ring`` in the eye (hollowness γ), ``ζ_ring``
     on the annulus [r1, r2] (thickness δ = r1/r2), ~0 outside, with cubic
     Hermite transition zones.
  2. V(r) = (1/r)∫₀ʳ ζ r′ dr′, then a smooth outer taper to zero at ``r_cut``
     (confines the circulation inside the 20°×20° regional domain; the taper
     implies a weak negative-ζ skirt, as in zero-circulation ring setups).
     The profile is rescaled so max V = ``vmax``.
  3. Vertical structure V(r,p) = V(r)·F(p) with a typhoon-like decay profile
     F(p) (1 in the low troposphere, →0 above ~150 hPa).
  4. Per level, gradient-wind balance  ∂Φ′/∂r = V²/r + fV  is integrated inward
     from the outer edge (f = f(center), an f-plane construction) → Φ′(r,p).
  5. Hydrostatic balance  T′ = −(1/R_d)·∂Φ′/∂ln p  → warm core consistent with
     the decaying wind field. msl′ = ρ_sfc · Φ′(1000 hPa).

DLAMPty units: z channel is **geopotential (m² s⁻²)** (v57_5d.yaml), msl/sp in
Pa — Φ′ is added to z directly, no g factor.
"""
from __future__ import annotations

import numpy as np

RD = 287.05
OMEGA = 7.2921e-5
R_EARTH = 6371000.0
DEG2M = np.pi / 180.0 * R_EARTH


def smooth_step(s: np.ndarray) -> np.ndarray:
    """Cubic Hermite S-curve: 0 for s≤0, 1 for s≥1, 3s²−2s³ between."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def ring_zeta_profile(r: np.ndarray, r1: float, r2: float, *,
                      gamma: float = 0.2, trans_frac: float = 0.15,
                      d_min: float = 0.0) -> np.ndarray:
    """Unit-amplitude three-region ring ζ(r): γ in the eye, 1 on [r1, r2], 0 outside.

    ``trans_frac`` sets the transition half-width as a fraction of the ring
    width; ``d_min`` (same units as r) keeps the transition resolvable on the
    0.25° grid (an under-resolved jump imprints a spurious m=4 grid signal).
    Both are capped so the two transitions never overlap.
    """
    if not 0 < r1 < r2:
        raise ValueError(f"need 0 < r1 < r2, got r1={r1}, r2={r2}")
    d = max(trans_frac * (r2 - r1), d_min)
    d = min(d, 0.45 * (r2 - r1))          # keep r1+d < r2−d
    w1 = smooth_step((r - (r1 - d)) / (2 * d))   # eye → ring
    w2 = smooth_step((r - (r2 - d)) / (2 * d))   # ring → environment
    return gamma * (1.0 - w1) + w1 * (1.0 - w2)


def banded_zeta_profile(r: np.ndarray, bands, trans: float) -> np.ndarray:
    """ζ(r) as a sum of smooth annular bands — radially *alternating* vorticity.

    ``bands`` = list of ``(r_in, r_out, amp)`` (same length unit as ``r``;
    ``amp`` relative, may be **negative** for sign reversals). Each band is a
    Hermite-smoothed top hat of edge half-width ``trans``; overlapping
    transitions blend by summation. ``r_in <= 0`` means the band starts at the
    centre (no inner transition). This is the generalization the Rayleigh
    (barotropic-instability) criterion cares about: multiple sign changes of
    dζ/dr along the radius.
    """
    z = np.zeros_like(r)
    for r_in, r_out, amp in bands:
        if r_out <= r_in:
            raise ValueError(f"band needs r_out > r_in, got ({r_in}, {r_out})")
        inner = (np.ones_like(r) if r_in <= 0
                 else smooth_step((r - (r_in - trans)) / (2 * trans)))
        outer = 1.0 - smooth_step((r - (r_out - trans)) / (2 * trans))
        z += float(amp) * inner * outer
    return z


def v_from_zeta(r: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Tangential wind V(r) = (1/r)∫₀ʳ ζ r′ dr′ (trapezoid; V(0)=0)."""
    integrand = zeta * r
    circ = np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                            * np.diff(r))))
    r_safe = np.where(r > 0, r, 1.0)
    v = circ / r_safe
    v[r == 0] = 0.0
    return v


def outer_taper(r: np.ndarray, r_cut: float, taper_frac: float = 0.25) -> np.ndarray:
    """Smooth 1→0 taper ending at ``r_cut`` (width ``taper_frac``·r_cut)."""
    d = taper_frac * r_cut
    return 1.0 - smooth_step((r - (r_cut - d)) / d)


def default_vertical_profile(p_hpa) -> np.ndarray:
    """Typhoon-like normalized decay F(p): 1 for p ≥ 850 hPa, →0 at 150 hPa."""
    p = np.asarray(p_hpa, dtype=float)
    return smooth_step((p - 150.0) / (850.0 - 150.0))


def gradient_wind_phi(r: np.ndarray, v: np.ndarray, f: float) -> np.ndarray:
    """Φ′(r) from ∂Φ′/∂r = V²/r + fV, integrated so Φ′(r_max) = 0."""
    r_safe = np.where(r > 0, r, 1.0)
    integrand = v * v / r_safe + f * v
    integrand[r == 0] = f * v[0]
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                           * np.diff(r))))
    return cum - cum[-1]


def hydrostatic_t_prime(phi_levels: np.ndarray, p_hpa) -> np.ndarray:
    """T′(level, r) = −(1/R_d)·∂Φ′/∂ln p on the (nz, nr) level–radius plane."""
    lnp = np.log(np.asarray(p_hpa, dtype=float))
    return -np.gradient(phi_levels, lnp, axis=0) / RD


def build_ring_1d(*, gamma: float, delta: float, rmw_deg: float, vmax: float,
                  lat_center_deg: float, p_hpa, vertical_weights=None,
                  r_cut_deg: float | None = None, trans_frac: float = 0.15,
                  min_trans_deg: float = 0.3, dr_deg: float = 0.02,
                  r_max_deg: float = 16.0, zeta_bands=None) -> dict:
    """Balanced axisymmetric ring on a fine radial grid.

    Returns dict with 1-D ``r`` (m) and per-level 2-D (nz, nr) arrays
    ``v``, ``phi``, ``t`` plus 1-D ``v_sfc-level`` helpers and scalars.
    ``delta`` = r1/r2 (ring thickness), ``rmw_deg`` = r2 (outer ring radius, deg).

    ``zeta_bands`` (optional) replaces the single (γ, δ)-ring with an
    **alternating multi-band** ζ(r): list of ``[r_in_deg, r_out_deg, amp_rel]``
    (amp may be negative). γ/δ are ignored then; ``rmw_deg`` should be set to
    the outermost band's ``r_out_deg`` (used only for r_cut default + naming).
    """
    p_hpa = np.asarray(p_hpa, dtype=float)
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat_center_deg))
    r2 = rmw_deg * DEG2M
    r1 = delta * r2
    if r_cut_deg is None:
        r_cut_deg = min(3.0 * rmw_deg, 8.5)
    if r_cut_deg <= rmw_deg * 1.2:
        raise ValueError(f"r_cut_deg={r_cut_deg} too close to rmw_deg={rmw_deg}")
    r = np.arange(0.0, r_max_deg * DEG2M + 1.0, dr_deg * DEG2M)

    if zeta_bands is not None:
        bands_m = [(b[0] * DEG2M, b[1] * DEG2M, b[2]) for b in zeta_bands]
        zeta = banded_zeta_profile(r, bands_m, trans=min_trans_deg * DEG2M)
    else:
        zeta = ring_zeta_profile(r, r1, r2, gamma=gamma, trans_frac=trans_frac,
                                 d_min=min_trans_deg * DEG2M)
    v = v_from_zeta(r, zeta) * outer_taper(r, r_cut_deg * DEG2M)
    v *= vmax / np.abs(v).max()

    F = (np.asarray(vertical_weights, dtype=float) if vertical_weights is not None
         else default_vertical_profile(p_hpa))
    if F.shape != p_hpa.shape:
        raise ValueError(f"vertical_weights length {F.shape} != levels {p_hpa.shape}")

    v_levels = F[:, None] * v[None, :]                       # (nz, nr)
    phi_levels = np.stack([gradient_wind_phi(r, v_levels[k], f0)
                           for k in range(len(p_hpa))])      # (nz, nr)
    t_levels = hydrostatic_t_prime(phi_levels, p_hpa)        # (nz, nr)

    return {"r": r, "zeta": zeta, "v": v, "v_levels": v_levels,
            "phi_levels": phi_levels, "t_levels": t_levels, "F": F, "f0": f0,
            "r1": r1, "r2": r2, "r_cut": r_cut_deg * DEG2M, "vmax": float(vmax)}


def domain_polar_geometry(lats: np.ndarray, lons: np.ndarray,
                          center: tuple[float, float] | None = None):
    """Local-Cartesian (r, θ) about ``center`` (lat, lon; default domain center).

    Returns ``(r_2d [m], theta_2d, lat_c, lon_c, f0)``; x east, y north.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if center is None:
        lat_c = float(lats[len(lats) // 2])
        lon_c = float(lons[len(lons) // 2])
    else:
        lat_c, lon_c = float(center[0]), float(center[1])
    lon2d, lat2d = np.meshgrid(lons, lats)
    x = (lon2d - lon_c) * DEG2M * np.cos(np.radians(lat_c))
    y = (lat2d - lat_c) * DEG2M
    r_2d = np.sqrt(x * x + y * y)
    theta_2d = np.arctan2(y, x)
    f0 = 2.0 * OMEGA * np.sin(np.radians(lat_c))
    return r_2d, theta_2d, lat_c, lon_c, f0


def profile_to_2d(r_1d: np.ndarray, prof_1d: np.ndarray, r_2d: np.ndarray) -> np.ndarray:
    """Interpolate a radial profile onto the 2-D domain (0 beyond the grid)."""
    return np.interp(r_2d, r_1d, prof_1d, right=0.0)
