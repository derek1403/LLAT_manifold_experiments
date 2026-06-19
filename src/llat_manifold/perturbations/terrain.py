"""Terrain modification (initial-condition intervention) — scaffold.

Claims the static surface channels ``hgt`` and ``landmask`` (dynamic masking) so a
flattened / removed-terrain field persists through the run. The actual orography
edit (e.g. zero out ``hgt`` in a box, set ``landmask`` to ocean) is left as a TODO
to be filled in when this category is implemented.
"""
from __future__ import annotations

from .base import Perturbation, StepContext


class TerrainPerturbation(Perturbation):
    def __init__(self, *, mode: str = "remove"):
        self.mode = mode
        self.param_tag = f"terrain_{mode}"

    def apply_ic(self, upper, surface, ctx: StepContext):
        raise NotImplementedError(
            "TerrainPerturbation.apply_ic is a scaffold; implement the hgt/landmask "
            "edit for this experiment category.")

    def claimed_static_vars(self) -> list[str]:
        return ["hgt", "landmask"]
