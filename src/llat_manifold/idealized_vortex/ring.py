"""Ring-vortex IC perturbation (balanced annular vorticity ring + noise seed).

Subclasses the shared :class:`llat_manifold.perturbations.base.Perturbation` ABC
but is **not** registered in the shared registry — :mod:`.driver_vortex` builds
it directly, so ``perturbations/registry.py`` and the ``vortex.py`` scaffold
stay untouched.

``apply_ic`` adds, on top of the (quiescent) background state:

  * u, v      : balanced ring tangential wind V(r)·F(p) (cyclonic, NH)
  * z         : gradient-wind Φ′(r,p)  (geopotential, m² s⁻²)
  * t         : hydrostatic warm-core T′(r,p)
  * u10, v10  : 0.8 × lowest-level ring wind
  * msl, sp   : ρ_sfc·Φ′(1000 hPa)
  * u, v      : + small white-noise seed (amplitude ``noise_amp`` m/s, fixed
                RNG ``seed``) to break axisymmetry — the barotropic-instability
                trigger; q/w/tcwv and all statics are left unchanged.

Parameters (all yaml-config keys): ``gamma`` (hollowness ζ_eye/ζ_ring),
``delta`` (thickness r1/r2), ``rmw_deg`` (outer ring radius r2, degrees),
``vmax`` (m/s), ``noise_amp``, ``seed``, ``r_cut_deg``, ``trans_frac``,
``center`` ([lat, lon], default domain center).
"""
from __future__ import annotations

import numpy as np

from .. import layout
from ..perturbations.base import Perturbation, StepContext
from . import balance


def _num_tag(x: float) -> str:
    """Filename-safe number: 2.5 -> '2p5', 0.7 -> '0p7'."""
    return f"{x:g}".replace(".", "p").replace("-", "m")


class RingVortexPerturbation(Perturbation):
    """Balanced annular-vorticity-ring IC + white-noise asymmetry seed."""

    def __init__(self, *, gamma: float = 0.2, delta: float = 0.7,
                 rmw_deg: float = 2.5, vmax: float = 35.0,
                 noise_amp: float = 0.05, seed: int = 0,
                 r_cut_deg: float | None = None, trans_frac: float = 0.15,
                 center=None, sfc_wind_factor: float = 0.8, zeta_bands=None):
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma (hollowness) must be in [0,1), got {gamma}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta (r1/r2) must be in (0,1), got {delta}")
        if zeta_bands is not None and rmw_deg < max(b[1] for b in zeta_bands):
            raise ValueError("with zeta_bands, set rmw_deg = outermost r_out_deg "
                             "(drives r_cut default and the diagnostics' ring band)")
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.rmw_deg = float(rmw_deg)
        self.vmax = float(vmax)
        self.noise_amp = float(noise_amp)
        self.seed = int(seed)
        self.r_cut_deg = None if r_cut_deg is None else float(r_cut_deg)
        self.trans_frac = float(trans_frac)
        self.center = None if center is None else (float(center[0]), float(center[1]))
        self.sfc_wind_factor = float(sfc_wind_factor)
        self.zeta_bands = ([[float(x) for x in b] for b in zeta_bands]
                           if zeta_bands is not None else None)
        if self.zeta_bands is not None:
            self.param_tag = (f"ringbands{len(self.zeta_bands)}"
                              f"_r{_num_tag(rmw_deg)}_v{_num_tag(vmax)}")
        else:
            self.param_tag = (f"ring_g{_num_tag(gamma)}_d{_num_tag(delta)}"
                              f"_r{_num_tag(rmw_deg)}_v{_num_tag(vmax)}")

    # ------------------------------------------------------------------ #
    def apply_ic(self, upper, surface, ctx: StepContext):
        p_hpa = np.asarray(layout.pressure_levels(), dtype=float)
        nz = upper.shape[0]
        r_2d, theta_2d, lat_c, lon_c, f0 = balance.domain_polar_geometry(
            ctx.dlam_lats, ctx.dlam_lons, self.center)

        ring = balance.build_ring_1d(
            gamma=self.gamma, delta=self.delta, rmw_deg=self.rmw_deg,
            vmax=self.vmax, lat_center_deg=lat_c, p_hpa=p_hpa[:nz],
            r_cut_deg=self.r_cut_deg, trans_frac=self.trans_frac,
            zeta_bands=self.zeta_bands)

        ui, vi = layout.upper_index("u"), layout.upper_index("v")
        ti, zi = layout.upper_index("t"), layout.upper_index("z")

        # unit tangential vector (counterclockwise / cyclonic NH): (−sinθ, cosθ)
        sin_t, cos_t = np.sin(theta_2d), np.cos(theta_2d)
        for k in range(nz):
            v_2d = balance.profile_to_2d(ring["r"], ring["v_levels"][k], r_2d)
            upper[k, :, :, ui] += -v_2d * sin_t
            upper[k, :, :, vi] += v_2d * cos_t
            upper[k, :, :, zi] += balance.profile_to_2d(
                ring["r"], ring["phi_levels"][k], r_2d)
            upper[k, :, :, ti] += balance.profile_to_2d(
                ring["r"], ring["t_levels"][k], r_2d)

        # surface: msl'/sp' from Φ′(1000 hPa); 10-m wind = reduced lowest level.
        k_low = int(np.argmax(p_hpa[:nz]))                      # 1000 hPa
        phi_low = balance.profile_to_2d(ring["r"], ring["phi_levels"][k_low], r_2d)
        i_msl, i_sp = layout.surface_index("msl"), layout.surface_index("sp")
        i_t2m = layout.surface_index("t2m")
        rho_sfc = float(np.mean(surface[:, :, i_msl])
                        / (balance.RD * np.mean(surface[:, :, i_t2m])))
        surface[:, :, i_msl] += rho_sfc * phi_low
        surface[:, :, i_sp] += rho_sfc * phi_low

        v_low = balance.profile_to_2d(ring["r"], ring["v_levels"][k_low], r_2d)
        surface[:, :, layout.surface_index("u10")] += \
            self.sfc_wind_factor * (-v_low * sin_t)
        surface[:, :, layout.surface_index("v10")] += \
            self.sfc_wind_factor * (v_low * cos_t)

        # white-noise asymmetry seed on u/v, confined inside the ring circulation
        if self.noise_amp > 0.0:
            rng = np.random.default_rng(self.seed)
            mask = balance.outer_taper(r_2d, ring["r_cut"])
            F = ring["F"]
            for k in range(nz):
                amp = self.noise_amp * F[k]
                if amp == 0.0:
                    continue
                upper[k, :, :, ui] += amp * rng.standard_normal(r_2d.shape) * mask
                upper[k, :, :, vi] += amp * rng.standard_normal(r_2d.shape) * mask
        return upper, surface
