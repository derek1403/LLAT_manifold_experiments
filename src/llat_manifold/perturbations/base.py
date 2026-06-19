"""The ``Perturbation`` injection interface (the "override" unit).

Each physics experiment subclasses :class:`Perturbation` and implements one or both
injection points on the **DLAMPty** state (upper + surface). There is no FCNv2 field:
the experiments run the regional model alone (see :mod:`llat_manifold.operators`).

  * :meth:`apply_ic`   — modify the t=0 fields directly (SST/terrain edits, vortex
    replacement, a one-shot heating bump). Called once, right after
    ``DLAMPty.IC_from_xarray_to_npy`` builds the arrays.
  * :meth:`apply_step` — return the per-step constant forcing ``f`` added inside the
    iteration (continuous diabatic heating, nudging). Called every step with a
    :class:`StepContext`; it receives zero arrays and returns ``f`` (the driver adds
    it to ``base + δ`` per the perturbation-evolution recursion).

Dynamic static masking: :meth:`claimed_static_vars` lists the static surface
variables this experiment takes over, so the driver releases them from the default
lock (see :mod:`llat_manifold.layout`).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np


@dataclass
class StepContext:
    """Per-step information handed to :meth:`Perturbation.apply_step`."""
    fore_i: int                 # 1-based step (or iteration) index
    lead_hr: int                # forecast lead time in hours (frozen modes: 0)
    initial_time: _dt.datetime
    dlam_lats: np.ndarray       # DLAMPty sub-domain latitudes (length 81)
    dlam_lons: np.ndarray       # DLAMPty sub-domain longitudes (length 81)


class Perturbation:
    """Base class: an identity perturbation. Subclasses override the hooks."""

    #: short, filesystem-safe tag identifying the parameter set (for output names).
    param_tag: str = "none"

    def apply_ic(self, upper, surface, ctx: StepContext):
        """Return possibly-modified ``(upper, surface)`` t=0 fields."""
        return upper, surface

    def apply_step(self, f_upper, f_surface, ctx: StepContext):
        """Return the per-step forcing ``(f_upper, f_surface)`` (added to base + δ)."""
        return f_upper, f_surface

    def claimed_static_vars(self) -> list[str]:
        """Static surface-variable names this experiment owns (released from lock)."""
        return []
