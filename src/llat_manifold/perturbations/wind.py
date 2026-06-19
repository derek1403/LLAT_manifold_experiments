"""Wind initial-perturbation (initial-condition intervention) — scaffold.

Port target: ``semi-linear/.../exp_wind_init``. Adds a balanced/unbalanced wind
(``u``/``v``) perturbation to the upper-level columns at t=0. Left as a scaffold
until the wind category is built out.
"""
from __future__ import annotations

from .base import Perturbation, StepContext


class WindPerturbation(Perturbation):
    def __init__(self, *, amp: float = 1.0):
        self.amp = float(amp)
        self.param_tag = f"wind_{amp:g}"

    def apply_ic(self, upper, surface, ctx: StepContext):
        raise NotImplementedError(
            "WindPerturbation.apply_ic is a scaffold; port the u/v-perturbation here.")
