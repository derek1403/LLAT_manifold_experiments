"""Sea-surface-temperature modification (initial-condition intervention).

Demonstrates *dynamic static masking*: SST (``sst_filled``) is a static surface
channel that is normally locked to the control. This experiment claims it, so the
driver leaves it free — the warmed/cooled SST persists through the run instead of
being reset every step.

Minimal first implementation: add a uniform ΔSST over the DLAMPty sub-domain at t=0.
Extend with spatial masks (e.g. warm-pool patches) as needed.
"""
from __future__ import annotations

from .. import layout
from .base import Perturbation, StepContext


class SSTPerturbation(Perturbation):
    def __init__(self, *, delta_K: float = 1.0):
        self.delta_K = float(delta_K)
        sign = "p" if delta_K >= 0 else "m"
        self.param_tag = f"sst_{sign}{abs(delta_K):g}K"

    def apply_ic(self, upper, surface, ctx: StepContext):
        surface[:, :, layout.surface_index("sst_filled")] += self.delta_K
        return upper, surface

    def claimed_static_vars(self) -> list[str]:
        # Take ownership of SST so it is NOT reset to the control each step.
        return ["sst_filled"]
