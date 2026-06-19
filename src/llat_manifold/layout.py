"""Variable-index helpers and the static-variable lock set (with dynamic masking).

All channel-index knowledge comes from ``config/model_layout.yaml`` (itself a copy
of the DLAMPty model yaml). Nothing else in the codebase should hard-code an index.

The *static lock* is the set of DLAMPty-surface channels that are not prognostic
(``sst_filled``, ``f``, ``solar``, ``hgt``, ``landmask``, the diurnal/day-of-year
encodings) plus the appended lat/lon space-info channels. In a perturbation run the
driver resets these to the control after every model step and zeros them in the
delta, which prevents a spurious static dipole from contaminating the perturbation.

*Dynamic masking*: an experiment that deliberately modifies a static field (e.g. an
SST-warming run owns ``sst_filled``; a terrain run owns ``hgt``/``landmask``) declares
those names via ``Perturbation.claimed_static_vars()``. :func:`active_lock_indices`
then drops them from the lock so they are free to differ from the control.
"""
from __future__ import annotations

from . import config


def _dlampty() -> dict:
    return config.layout()["dlampty"]


def surface_vars() -> list[str]:
    return list(_dlampty()["surface_vars"])


def upper_vars() -> list[str]:
    return list(_dlampty()["upper_vars"])


def pressure_levels() -> list[int]:
    return list(_dlampty()["pressure_levels"])


def surface_index(name: str) -> int:
    """Index of a DLAMPty surface variable on the last array axis."""
    return surface_vars().index(name)


def upper_index(name: str) -> int:
    """Index of a DLAMPty upper variable on the last array axis."""
    return upper_vars().index(name)


def fcnv2_index(key: str) -> int:
    """FCNv2 global-field channel index, e.g. ``t2m_index`` -> 4."""
    return int(config.layout()["fcnv2"][key])


def space_info_indices() -> list[int]:
    """Negative indices of the appended lat/lon channels (may be empty)."""
    return list(_dlampty().get("space_info_indices", []))


def default_static_indices() -> list[int]:
    """All surface channels locked by default: static fields + lat/lon space-info.

    Positive indices (resolved from names) come first, then the negative
    space-info indices, matching how the legacy scripts addressed them.
    """
    names = _dlampty()["static_surface_vars"]
    idx = [surface_index(n) for n in names]
    return idx + space_info_indices()


def active_lock_indices(claimed_static_vars=()) -> list[int]:
    """Lock indices in force for a run, after dynamic masking.

    ``claimed_static_vars`` is an iterable of surface-variable *names* the
    experiment takes over (released from the lock). The special token
    ``"space_info"`` releases the lat/lon channels too.
    """
    claimed = set(claimed_static_vars or ())
    release_space_info = "space_info" in claimed
    claimed_idx = {surface_index(n) for n in claimed if n != "space_info"}

    out: list[int] = []
    for i in default_static_indices():
        if i < 0:
            if release_space_info:
                continue
        elif i in claimed_idx:
            continue
        out.append(i)
    return out
