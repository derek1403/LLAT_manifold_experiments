"""Idealized-vortex initial condition (initial-condition intervention) — scaffold.

Replaces the t=0 fields with a precomputed axisymmetric vortex (port target:
``inference_two_way_FCNv2_idealized_vortex_all.py`` + the ``idealize_vortex`` input
arrays). Implemented as ``apply_ic`` overwriting upper/surface channels from the
idealized arrays. Left as a scaffold until the vortex category is built out.
"""
from __future__ import annotations

from .base import Perturbation, StepContext


class VortexPerturbation(Perturbation):
    def __init__(self, *, vortex_name: str = "idealize_vortex"):
        self.vortex_name = vortex_name
        self.param_tag = f"vortex_{vortex_name}"

    def apply_ic(self, upper, surface, ctx: StepContext):
        raise NotImplementedError(
            "VortexPerturbation.apply_ic is a scaffold; port the idealize_vortex "
            "overwrite of upper/surface channels here.")
