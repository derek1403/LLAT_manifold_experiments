"""Moisture-only (δq) perturbation — the reverse probe of the q-lock experiments.

Injects a specific-humidity anomaly with the same spatial design as the heating
(vertical profile × centred horizontal Gaussian) but NO temperature bump: θ̇ = 0,
so a dry-dynamical model should generate (almost) no PV. If the manifold routes
its thermodynamic response through the q channel, PV will move anyway — the
two-sided evidence for the moisture binding seen in the q-lock runs.

Two amplitude scalings, both parameterised by a nominal ``amp_K`` so the runs
share an x-axis with the heating sweep:

* **measured** — pass ``gkg_per_K`` + a 13-level ``profile`` measured from the
  moist heating runs' actual δq (see
  ``experiments/diabatic_heating/configs/dq_measured.yaml``): at nominal amp_K,
  inject the δq the moist amp_K heating run actually grew by nominal hour 24.
* **latent-equivalent** — omit ``gkg_per_K``/``profile`` → cp/Lv ≈ 0.402 g/kg
  per K (the moisture whose complete condensation would release amp_K of
  heating), vertical profile from ``heat_type`` (Deep/Shallow/Stratiform).

``injection``/``amp_mode``/``forcing_steps`` follow HeatingPerturbation exactly.
"""
from __future__ import annotations

import numpy as np

from .. import layout
from .base import Perturbation, StepContext
from .heating import _amp_tag, gaussian_centered, vertical_profile

CP, LV = 1004.0, 2.5e6
LATENT_GKG_PER_K = CP / LV * 1.0e3        # ≈ 0.402 g/kg per equivalent K


class MoisturePerturbation(Perturbation):
    """Gaussian δq bump on the upper-q column; IC bump or per-step forcing."""

    def __init__(self, *, injection: str = "per_step", amp_K: float = 5.0,
                 gkg_per_K: float | None = None, profile=None,
                 heat_type: str = "Deep", sigma: float = 5.0,
                 forcing_steps: int = 1, amp_mode: str = "each",
                 offset_pts=(0, 0), npts: int | None = None):
        if injection not in ("ic", "per_step"):
            raise ValueError("injection must be 'ic' or 'per_step'")
        if amp_mode not in ("spread", "each"):
            raise ValueError("amp_mode must be 'spread' or 'each'")
        self.injection = injection
        self.amp_K = float(amp_K)
        self.measured = profile is not None
        if self.measured and gkg_per_K is None:
            raise ValueError("a measured profile needs an explicit gkg_per_K")
        self.gkg_per_K = float(gkg_per_K) if gkg_per_K is not None else LATENT_GKG_PER_K
        self.profile = None if profile is None else np.asarray(profile, dtype=float)
        self.heat_type = heat_type
        self.sigma = float(sigma)
        self.forcing_steps = int(forcing_steps)
        self.amp_mode = amp_mode
        # Injection centre offset from the domain centre (grid points, +y south→north
        # per array orientation, +x eastward) — the off-vortex state-dependence probe.
        self.offset_pts = (int(offset_pts[0]), int(offset_pts[1]))
        self.npts = npts or int(layout.config.layout()["grid"]["dlampty_npts"])
        amp_tag, kind = _amp_tag(self.amp_K), "dqM" if self.measured else "dqL"
        if self.offset_pts != (0, 0):
            kind += f"off{self.offset_pts[0]}x{self.offset_pts[1]}".replace("-", "m")
        if injection == "ic":
            self.param_tag = f"{amp_tag:0>4s}_{kind}_icbump"
        else:
            self.param_tag = f"{amp_tag}_{kind}_{forcing_steps}steps"

    def _unit_field(self, nz: int) -> np.ndarray:
        """Unit-amplitude δq field (nz, ny, nx) in g/kg per equivalent K."""
        if self.measured:
            v = self.profile[:nz]
        else:
            v = vertical_profile(self.heat_type, layout.pressure_levels())[:nz]
        c = self.npts // 2
        cy, cx = c + self.offset_pts[0], c + self.offset_pts[1]
        yy, xx = np.meshgrid(np.arange(self.npts), np.arange(self.npts), indexing="ij")
        h_xy = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * self.sigma ** 2))
        return v[:, None, None] * h_xy[None, :, :]

    def _bump_kgkg(self, nz: int, amp_K: float) -> np.ndarray:
        return amp_K * self.gkg_per_K * 1.0e-3 * self._unit_field(nz)

    def apply_ic(self, upper, surface, ctx: StepContext):
        if self.injection != "ic":
            return upper, surface
        q_idx = layout.upper_index("q")
        upper[:, :, :, q_idx] += self._bump_kgkg(upper.shape[0], self.amp_K)
        return upper, surface

    def apply_step(self, f_upper, f_surface, ctx: StepContext):
        if self.injection != "per_step" or ctx.fore_i > self.forcing_steps:
            return f_upper, f_surface
        amp = self.amp_K if self.amp_mode == "each" else self.amp_K / self.forcing_steps
        q_idx = layout.upper_index("q")
        f_upper[:, :, :, q_idx] += self._bump_kgkg(f_upper.shape[0], amp)
        return f_upper, f_surface
