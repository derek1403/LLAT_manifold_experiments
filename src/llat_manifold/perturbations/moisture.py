"""Moisture initial-perturbation (initial-condition intervention) — scaffold.

Port target: ``semi-linear/.../exp_moisture_init``. Adds a humidity (``q``)
perturbation to the upper-level column (or surface ``tcwv``) at t=0. Left as a
scaffold until the moisture category is built out.
"""
from __future__ import annotations

from .base import Perturbation, StepContext


class MoisturePerturbation(Perturbation):
    def __init__(self, *, amp: float = 1.0):
        self.amp = float(amp)
        self.param_tag = f"moist_{amp:g}"

    def apply_ic(self, upper, surface, ctx: StepContext):
        raise NotImplementedError(
            "MoisturePerturbation.apply_ic is a scaffold; port the q-perturbation here.")
