"""Forcing–response curves (scaffold).

Port target: the ``Heating_vs_DeltaP_Response`` / ``RatePrecip_vs_DeltaP_Response``
figures under ``idealized_exp/pic/`` and ``plot_vorticity.py``. Collects a scalar
response (e.g. central-pressure drop Δp_min) across a sweep of perturbation
amplitudes and plots response vs forcing — the model's sensitivity curve, the most
direct probe of how the LLAT manifold reacts to a given physical intervention.

This module provides the response extractor; the sweep orchestration reads the per
-amplitude experiment outputs (each stamped with its config) and assembles the curve.
"""
from __future__ import annotations

import numpy as np

from .. import io, layout


def central_pressure_drop(delta_path) -> float:
    """Minimum (most negative) MSL perturbation in a delta bundle, in the field's units.

    A proxy for intensification: how much the storm's central pressure dropped due to
    the perturbation. Uses the DLAMPty surface ``msl`` channel.
    """
    _, dsfc = io.load_delta_bundle(delta_path)
    return float(np.min(dsfc[:, :, layout.surface_index("msl")]))
