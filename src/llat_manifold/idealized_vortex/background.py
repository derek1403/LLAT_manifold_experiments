"""Quiescent horizontally-uniform background for the ring-vortex experiment.

The shared driver always uses a *real analysis* IC that (by construction of the
TC-following domain) contains a real typhoon — useless as the environment of an
idealized instability run. This module flattens such an analysis into a
quiescent environment:

  * upper u, v, w                → 0
  * upper t, q, z                → horizontal domain mean per level
  * u10, v10, tp                 → 0
  * t2m, d2m, msl, sp, tcwv,
    mtnlwrf, sst_filled          → horizontal domain mean (uniform ocean)
  * hgt, landmask                → 0 (uniform ocean)
  * f, solar, time encodings,
    lat/lon space-info           → kept (f keeps β; statics stay locked)

Saved as one npz (keys ``dlampty_upper``/``dlampty_sfc`` + ``lats``/``lons``/
``init_time``) loaded by :mod:`.driver_vortex` via the ``background_npz`` config
key. Static-channel consistency: the flattened statics live in the background
itself, so the driver's default lock keeps them fixed — no ``claimed_static_vars``
needed (snapshot mode never recomputes time encodings).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config, driver, layout

_ZERO_UPPER = ("u", "v", "w")
_MEAN_UPPER = ("t", "q", "z")
_ZERO_SFC = ("u10", "v10", "tp")
_MEAN_SFC = ("t2m", "d2m", "msl", "sp", "tcwv", "mtnlwrf", "sst_filled")
_FLAT_ZERO_SFC = ("hgt", "landmask")


def make_quiescent(state: "driver.InitialState") -> "driver.InitialState":
    """Return a new InitialState with the flattened quiescent environment."""
    up = state.upper.copy()
    sfc = state.surface.copy()

    for name in _ZERO_UPPER:
        up[:, :, :, layout.upper_index(name)] = 0.0
    for name in _MEAN_UPPER:
        i = layout.upper_index(name)
        up[:, :, :, i] = up[:, :, :, i].mean(axis=(1, 2), keepdims=True)

    for name in _ZERO_SFC + _FLAT_ZERO_SFC:
        sfc[:, :, layout.surface_index(name)] = 0.0
    for name in _MEAN_SFC:
        i = layout.surface_index(name)
        sfc[:, :, i] = sfc[:, :, i].mean()

    return driver.InitialState(up, sfc, state.dlam_lats, state.dlam_lons,
                               state.initial_time)


def save_background(state: "driver.InitialState", path) -> str:
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, dlampty_upper=state.upper, dlampty_sfc=state.surface,
        lats=state.dlam_lats, lons=state.dlam_lons,
        init_time=state.initial_time.strftime("%Y%m%d%H"))
    return path


def load_background(path) -> "driver.InitialState":
    import datetime as _dt
    with np.load(config._resolve(str(path))) as z:
        return driver.InitialState(
            z["dlampty_upper"], z["dlampty_sfc"], z["lats"], z["lons"],
            _dt.datetime.strptime(str(z["init_time"]), "%Y%m%d%H"))


def default_background_path(tc_id: str, init_str: str) -> Path:
    return (config.output_root() / "idealized_vortex" / "backgrounds"
            / f"quiescent_{tc_id}_{init_str}.npz")


def build_and_save(tc_id: str, init_str: str, dlampty, out_path=None) -> str:
    """Load the real analysis IC, flatten it, save the quiescent background."""
    state = driver.load_initial_state(tc_id, init_str, dlampty)
    quiet = make_quiescent(state)
    out = out_path or default_background_path(tc_id, init_str)
    return save_background(quiet, out)
