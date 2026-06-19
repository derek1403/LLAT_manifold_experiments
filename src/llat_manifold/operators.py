"""The model operator ``M`` — one clean 3-hour DLAMPty step.

The LLAT semi-linear / perturbation experiments operate on a **single regional model
(DLAMPty), a single time step**. There is deliberately **no FCNv2 call and no 7.5°
two-way coupling**: ``M`` is just the exact 3-hour DLAMPty inference. This is the
primitive iterated by :mod:`llat_manifold.driver`.

    ū   = M(u₀)                          background (one step)
    u′ᵢ = M(base + u′ᵢ₋₁ + f) − ū        finite-time nonlinear perturbation evolution

(See ``docs/perturbation_method.md``.) The earlier 6-hour coupled "super operator" has
been removed; idealized experiments want the clean regional evolution.

Time handling:
  * ``advance_time=False`` (snapshot): never call ``changing_additional_information``;
    the operator is iterated at one frozen valid time.
  * ``advance_time=True`` (continuous / forward): advance the time-encoded channels to
    ``valid_time`` after the step.

Static-variable locking: when ``lock_ref`` / ``lock_idx`` are given, the listed
DLAMPty-surface channels are reset to the reference after the step (keeps a
perturbation run's non-prognostic fields identical to the background; see
:mod:`llat_manifold.layout`).
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

STEP_HOURS = 3        # one DLAMPty step = 3 hours


def load_models(dlampty_device: str | None = None):
    """Initialise and return the DLAMPty regional model (from Part 1's package)."""
    from regional_couple.inference import models as rc_models  # editable Part 1
    return rc_models.load_dlampty()


def _lock(surface: np.ndarray, lock_ref, lock_idx) -> None:
    """In-place reset of the locked surface channels to the reference."""
    if lock_ref is None or not lock_idx:
        return
    for idx in lock_idx:
        surface[:, :, idx] = lock_ref[:, :, idx]


def m_operator(upper, surface, dlampty, *, advance_time: bool = False,
               valid_time: _dt.datetime | None = None,
               lock_ref=None, lock_idx=()):
    """Apply ``M`` once: a single 3-hour DLAMPty step. Inputs are not mutated.

    Returns ``(upper', surface')``. ``valid_time`` is required when
    ``advance_time`` is True.
    """
    if advance_time and valid_time is None:
        raise ValueError("advance_time=True requires valid_time")

    up, sfc = dlampty.predict_one_step(np.ascontiguousarray(upper),
                                       np.ascontiguousarray(surface))
    _lock(sfc, lock_ref, lock_idx)
    if advance_time:
        up, sfc = dlampty.changing_additional_information(up, sfc, valid_time)
    return up, sfc
