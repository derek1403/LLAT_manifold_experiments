"""Outward-propagation diagnostics: gravity-wave radiation & Rossby-radius adjustment.

The LLAT grid is too coarse to resolve polygonal-eyewall wavenumbers directly, so the
physical focus is the *consequences* that do reach our scales:

1. **Gravity-wave radiation** — a radius–time (Hovmöller) plot of an axisymmetric-mean
   perturbation field (default: DLAMPty surface MSL anomaly). An outward-tilting
   phase line gives the apparent radial phase speed of the radiating waves.
2. **Rossby-radius adjustment** — overlay the local Rossby radius of deformation
   L_R = N H / f (a single representative value here) so the perturbation's horizontal
   spreading can be compared against the adjustment scale that separates a
   gravity-wave-dominated (r < L_R) from a balanced (r > L_R) response.

First cut; reads the per-lead delta bundles produced by the continuous driver.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import io, layout
from . import experiment_title


def _radial_mean(field2d: np.ndarray, nbins: int | None = None) -> np.ndarray:
    """Azimuthal mean about the domain centre -> profile vs radius (grid units)."""
    ny, nx = field2d.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    nbins = nbins or int(min(cy, cx))
    rbin = np.clip(r.astype(int), 0, nbins - 1)
    prof = np.zeros(nbins)
    cnt = np.zeros(nbins)
    np.add.at(prof, rbin.ravel(), field2d.ravel())
    np.add.at(cnt, rbin.ravel(), 1.0)
    return prof / np.maximum(cnt, 1.0)


def _lead_of(path: Path) -> int:
    m = re.search(r"lead(\d+)hr", path.name)
    return int(m.group(1)) if m else -1


def rossby_radius_km(N=1e-2, H=1.0e4, f=5e-5) -> float:
    """First-baroclinic Rossby radius L_R = N H / f, in km (rough representative).

    The *environmental* scale: it knows nothing about the vortex sitting in the
    domain. For anything inside a typhoon use :func:`local_rossby_radius`, whose
    denominator is the vortex-modified inertial frequency rather than f.
    """
    return (N * H / f) / 1000.0


# --------------------------------------------------------------------------- #
# Vortex-modified (local) Rossby radius
#
# The classical L_R = NH/f is the adjustment scale of a resting atmosphere. Inside
# a vortex the restoring rotation is not f but the inertial frequency of the swirl,
#
#     I² = ξ η ,      ξ = f + 2V_t/r   (modified Coriolis / centrifugal)
#                     η = f + ζ        (absolute vorticity)
#
# so the local adjustment scale is L_R = N H / √(ξη). Because I grows fast toward a
# strong vortex's core, L_R collapses there: heating deposited inside L_R is retained
# locally as warming, heating deposited outside it is radiated away as gravity waves
# and lost. This is the scale that decides whether an intense storm confines an
# injected thermal anomaly while a weak one lets it spread — the quantitative form of
# the "efficiency of diabatic heating" argument (Schubert & Hack 1982; Hack & Schubert
# 1986; Vigh & Schubert 2009).
#
# Everything here is a *control-state* diagnostic: it characterises the vortex the
# perturbation was injected into, never the perturbation itself.
# --------------------------------------------------------------------------- #
_OMEGA = 7.2921e-5
_G = 9.80665
_RD = 287.05
_CP = 1004.0
_KAPPA = _RD / _CP
_H_SCALE_M = 1.0e4          # tropospheric depth used for N·H (10 km, conventional)


def brunt_vaisala(t_field, p_hpa) -> np.ndarray:
    """N [s⁻¹] per level from a temperature field, along the pressure axis.

    ``N² = −(g²ρ/θ)·∂θ/∂p`` with ``ρ = p/(R_d T)``, i.e. the isobaric form of
    ``N² = (g/θ)·∂θ/∂z``. ``t_field`` is ``(nz, ...)``; the result has the same
    shape. Negative (statically unstable) values are clipped to a small positive
    floor so ``√`` and the division in :func:`local_rossby_radius` stay finite —
    the affected cells are flagged by the floor value, not silently hidden.
    """
    t = np.asarray(t_field, dtype=float)
    p = np.asarray(p_hpa, dtype=float)[: t.shape[0]] * 100.0     # Pa
    shape = (-1,) + (1,) * (t.ndim - 1)
    theta = t * (100000.0 / p.reshape(shape)) ** _KAPPA
    dtheta_dp = np.gradient(theta, p, axis=0)
    rho = p.reshape(shape) / (_RD * t)
    n2 = -(_G ** 2) * rho / theta * dtheta_dp
    return np.sqrt(np.maximum(n2, 1e-8))


def inertial_frequency(upper, sfc, *, dr_km: float | None = None):
    """(r [km], I [s⁻¹] per level) — azimuthal-mean √(ξη) of a control state.

    ``ξ = f + 2V_t/r`` and ``η = f + ζ`` are both azimuthally averaged before the
    product, which is the axisymmetric balanced-vortex convention. ``f`` is taken
    analytically as ``2Ω sin(lat)`` at the domain centre (f-plane, matching the
    idealized-vortex builder in :mod:`..idealized_vortex.balance`) rather than
    from the model's ``f`` channel, so the same formula applies to real and
    synthetic ICs.

    The innermost bin is excluded from the ``2V_t/r`` term (it is 0/0 there) and
    refilled from the next bin, where ``V_t ∝ r`` makes ``2V_t/r → ζ(0)``.
    """
    from ..idealized_vortex.axisymmetric import radial_frame, radial_mean
    from ..idealized_vortex.azimuthal import relative_vorticity

    lon2d, lat2d = sfc[:, :, -2], sfc[:, :, -1]
    lats, lons = sfc[:, 0, -1], sfc[0, :, -2]
    r2d, theta2d, edges, centers = radial_frame(lat2d, lon2d, dr_km=dr_km)
    cy, cx = lat2d.shape[0] // 2, lat2d.shape[1] // 2
    f0 = 2.0 * _OMEGA * np.sin(np.deg2rad(lat2d[cy, cx]))

    ui, vi = layout.upper_index("u"), layout.upper_index("v")
    r_m = np.maximum(centers, centers[0]) * 1000.0
    xi, eta = [], []
    for k in range(upper.shape[0]):
        u2, v2 = upper[k, :, :, ui], upper[k, :, :, vi]
        vt = radial_mean(-u2 * np.sin(theta2d) + v2 * np.cos(theta2d), r2d, edges)
        z = radial_mean(relative_vorticity(u2, v2, lats, lons), r2d, edges)
        x = f0 + 2.0 * vt / r_m
        x[0] = f0 + z[0]                       # 2V_t/r → ζ on the axis
        xi.append(x)
        eta.append(f0 + z)
    xi, eta = np.stack(xi), np.stack(eta)
    # An inertially unstable ring (ξη < 0) has no adjustment scale; floor it at the
    # planetary value so L_R saturates at the environmental NH/f instead of blowing up.
    return centers, np.sqrt(np.maximum(xi * eta, f0 ** 2))


def local_rossby_radius(upper, sfc, *, p_band=(400.0, 700.0),
                        h_scale_m: float = _H_SCALE_M, dr_km: float | None = None):
    """(r [km], L_R(r) [km]) — vortex-modified Rossby radius of a control state.

    ``L_R = N·H/√(ξη)``, with ``N`` the mass-weighted mean over ``p_band`` of the
    domain-mean Brunt–Väisälä frequency and ``ξη`` from :func:`inertial_frequency`,
    likewise band-averaged. ``p_band`` defaults to 400–700 hPa — the layer the Deep
    heating profile peaks in, so the number describes the adjustment scale the
    injected anomaly actually experiences.
    """
    from .response import _mass_weights

    p = np.asarray(layout.pressure_levels(), dtype=float)[: upper.shape[0]]
    band = (p >= p_band[0]) & (p <= p_band[1])
    dm = _mass_weights(upper.shape[0])

    n_prof = brunt_vaisala(upper[..., layout.upper_index("t")], p).mean(axis=(1, 2))
    n_band = float(np.average(n_prof[band], weights=dm[band]))

    r_km, inertial = inertial_frequency(upper, sfc, dr_km=dr_km)
    i_band = np.average(inertial[band], axis=0, weights=dm[band])
    return r_km, (n_band * h_scale_m / i_band) / 1000.0


def rossby_radius_summary(upper, sfc, *, r_core_km: float = 278.0,
                          p_band=(400.0, 700.0),
                          h_scale_m: float = _H_SCALE_M) -> dict:
    """Scalars describing one control state's adjustment scale.

    ``L_R_core`` — the single number to regress a confinement radius against — is
    ``N·H/⟨I⟩`` with ``⟨I⟩`` the mean inertial frequency over the 2σ heated core,
    **not** the mean of the ``L_R(r)`` profile over the same disc. Those are not the
    same thing and the difference is not cosmetic: ``L_R ∝ 1/I``, so a plain mean of
    the profile is dominated by the outermost bins, where the vortex has decayed (and
    a tapered vortex even has a weak anticyclonic skirt) and ``L_R`` is enormous. A
    50 m/s vortex scores ~1000 km that way and ~400 km this way; only the second one
    describes the scale the injected anomaly sits in.

    ``L_R_min`` is the tightest point of the profile (at/near the RMW) and
    ``L_R_env`` the domain-edge value, which approaches the environmental ``NH/f``
    because the vortex contributes nothing out there.
    """
    r_km, l_r = local_rossby_radius(upper, sfc, p_band=p_band, h_scale_m=h_scale_m)
    core = r_km <= r_core_km
    _r, inertial = inertial_frequency(upper, sfc)
    p = np.asarray(layout.pressure_levels(), dtype=float)[: upper.shape[0]]
    band = (p >= p_band[0]) & (p <= p_band[1])
    from .response import _mass_weights
    dm = _mass_weights(upper.shape[0])
    n_band = float(np.average(
        brunt_vaisala(upper[..., layout.upper_index("t")], p).mean(axis=(1, 2))[band],
        weights=dm[band]))
    i_core = float(np.average(inertial[band][:, core], axis=0,
                              weights=dm[band]).mean())
    return {
        "r_km": r_km,
        "L_R_km": l_r,
        "L_R_core_km": n_band * h_scale_m / i_core / 1000.0,
        "L_R_min_km": float(np.nanmin(l_r)),
        "L_R_env_km": float(l_r[-1]),
        "I_core_s1": i_core,
        "N_band_s1": n_band,
    }


def _nice_vmax(value: float) -> float:
    """Round a positive magnitude up to a clean 1/2/5 × 10^k for fixed colorbars."""
    if value <= 0 or not np.isfinite(value):
        return 1.0
    exp = np.floor(np.log10(value))
    frac = value / 10 ** exp
    nice = 1.0 if frac <= 1 else 2.0 if frac <= 2 else 5.0 if frac <= 5 else 10.0
    return nice * 10 ** exp


def hovmoller(out_dir, surface_var="msl", out_png=None):
    """Radius–time Hovmöller of an axisymmetric-mean surface anomaly.

    Scans the delta bundles in ``out_dir`` (one per lead), builds the radial profile
    of the chosen DLAMPty surface variable's perturbation, and stacks them by lead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    files = sorted([p for p in out_dir.glob("delta_*lead*hr.npz")], key=_lead_of)
    if not files:
        raise FileNotFoundError(f"No 'delta_*lead*hr.npz' bundles in {out_dir}")

    s_idx = layout.surface_index(surface_var)
    res_km = float(layout.config.layout()["grid"]["resolution_deg"]) * 111.0

    leads, profiles = [], []
    for p in files:
        _, dsfc = io.load_delta_bundle(p)
        profiles.append(_radial_mean(dsfc[:, :, s_idx]))
        leads.append(_lead_of(p))
    H = np.array(profiles)                      # (ntime, nradius)
    radius_km = np.arange(H.shape[1]) * res_km
    leads = np.array(leads)

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = _nice_vmax(np.nanmax(np.abs(H)))
    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(radius_km, leads, np.clip(H, -vmax, vmax), levels=levels,
                     cmap="RdBu_r", extend="both")
    ax.axvline(rossby_radius_km(), color="k", ls="--", lw=1.2,
               label=f"$L_R$ ≈ {rossby_radius_km():.0f} km")
    ax.set_xlabel("Radius (km)", fontsize=12, weight="bold")
    ax.set_ylabel("Lead time (h)", fontsize=12, weight="bold")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))
    cbar.set_label(f"Δ{surface_var} (axisym. mean)", fontsize=11, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(experiment_title(out_dir, prefix="radius–time:"), fontsize=11, weight="bold")

    if out_png is not None:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"[waves] wrote {out_png}")
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="radius–time Hovmöller of a surface anomaly")
    ap.add_argument("out_dir", help="experiment output dir with delta bundles")
    ap.add_argument("--var", default="msl")
    ap.add_argument("--out", default="hovmoller.png")
    a = ap.parse_args()
    hovmoller(a.out_dir, a.var, a.out)
