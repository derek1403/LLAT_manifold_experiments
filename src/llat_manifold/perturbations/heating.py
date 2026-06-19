"""Diabatic-heating perturbation (DLAMPty upper-T column; no FCNv2).

A localized axisymmetric Gaussian heating with a Deep/Shallow/Stratiform vertical
profile, added to the upper-level temperature column. Faithful to
``idealized_exp/LLAT_add_heat_onestep.py`` (the heating field; T index 2, σ in grid
points, vertical weight over 200–1000 hPa).

Two injection styles, selected by config:
  * ``injection: ic``        — add the bump once to the t=0 field (``apply_ic``);
  * ``injection: per_step``  — return the same-shaped bump as the per-step forcing ``f``
    for the first ``forcing_steps`` steps (``apply_step``). ``amp_mode`` chooses the
    per-step amplitude: ``each`` = full ``amp_K`` every step (snapshot's constant
    forcing f), ``spread`` = ``amp_K / forcing_steps``.
"""
from __future__ import annotations

import numpy as np

from .. import layout
from .base import Perturbation, StepContext

_HEATING_PMIN, _HEATING_PMAX = 200, 1000


def vertical_profile(heat_type: str, pressure_levels) -> np.ndarray:
    """Vertical weight V(P) per level (Deep / Stratiform / Shallow), 0 outside 200–1000 hPa."""
    p = np.asarray(pressure_levels, dtype=float)
    v = np.zeros_like(p)
    mask = (p >= _HEATING_PMIN) & (p <= _HEATING_PMAX)
    pn = (p[mask] - _HEATING_PMIN) / (_HEATING_PMAX - _HEATING_PMIN)
    if heat_type == "Deep":
        v[mask] = np.sin(pn * np.pi)
    elif heat_type == "Stratiform":
        v[mask] = np.sin(2 * pn * np.pi)
    elif heat_type == "Shallow":
        v[mask] = (np.sin(pn * np.pi) - np.sin(2 * pn * np.pi)) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown heat_type {heat_type!r} (Deep/Shallow/Stratiform)")
    return v


def gaussian_centered(npts: int, sigma: float) -> np.ndarray:
    """Gaussian on an ``npts``x``npts`` grid centred at (npts//2, npts//2); σ in grid points."""
    c = npts // 2
    yy, xx = np.meshgrid(np.arange(npts), np.arange(npts), indexing="ij")
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma ** 2))


class HeatingPerturbation(Perturbation):
    """Gaussian diabatic heating on the upper-T column; IC bump or per-step forcing."""

    def __init__(self, *, injection: str = "ic", amp_K: float = 10.0,
                 heat_type: str = "Deep", sigma: float = 5.0,
                 forcing_steps: int = 1, amp_mode: str = "each",
                 npts: int | None = None):
        if injection not in ("ic", "per_step"):
            raise ValueError("injection must be 'ic' or 'per_step'")
        if amp_mode not in ("spread", "each"):
            raise ValueError("amp_mode must be 'spread' or 'each'")
        self.injection = injection
        self.amp_K = float(amp_K)
        self.heat_type = heat_type
        self.sigma = float(sigma)
        self.forcing_steps = int(forcing_steps)
        self.amp_mode = amp_mode
        self.npts = npts or int(layout.config.layout()["grid"]["dlampty_npts"])
        if injection == "ic":
            self.param_tag = f"{int(round(amp_K)):03d}K_{heat_type}_icbump"
        else:
            self.param_tag = f"{int(round(amp_K))}K_{heat_type}_{forcing_steps}steps"

    def _unit_field(self, nz: int) -> np.ndarray:
        """Unit-amplitude heating field (nz, ny, nx) = V(P) ⊗ horizontal Gaussian."""
        v = vertical_profile(self.heat_type, layout.pressure_levels())[:nz]
        h_xy = gaussian_centered(self.npts, self.sigma)
        return v[:, None, None] * h_xy[None, :, :]

    def apply_ic(self, upper, surface, ctx: StepContext):
        if self.injection != "ic":
            return upper, surface
        t_idx = layout.upper_index("t")
        upper[:, :, :, t_idx] += self.amp_K * self._unit_field(upper.shape[0])
        return upper, surface

    def apply_step(self, f_upper, f_surface, ctx: StepContext):
        if self.injection != "per_step" or ctx.fore_i > self.forcing_steps:
            return f_upper, f_surface
        amp = self.amp_K if self.amp_mode == "each" else self.amp_K / self.forcing_steps
        t_idx = layout.upper_index("t")
        f_upper[:, :, :, t_idx] += amp * self._unit_field(f_upper.shape[0])
        return f_upper, f_surface
