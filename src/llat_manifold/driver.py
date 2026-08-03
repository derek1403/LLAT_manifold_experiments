"""Perturbation-evolution driver over the DLAMPty model operator ``M`` (one 3h step).

No FCNv2, no coupling — the experiments are a clean single-model, single-step
evolution (see :mod:`llat_manifold.operators`).

Modes
-----
``snapshot`` (the LLAT semi-linear power iteration, frozen valid time)
    Background ``ū = M(u₀)`` computed once. Then, with constant forcing ``f`` added
    every iteration:

        u′₁ = M(u₀ + f) − ū
        u′ᵢ = M(ū + u′ᵢ₋₁ + f) − ū      (i ≥ 2)

    i.e. the base inside ``M`` is the true initial field on the first iteration and the
    once-evolved background ``ū`` thereafter; the departure is always measured from
    ``ū``. This amplifies the fastest-growing finite-time structure.

``continuous`` (time-marching)
    A control trajectory ``Iₙ = M(Iₙ₋₁)`` and a perturbed trajectory advance together
    with per-step forcing; ``δ = M(I + δ + f) − M(I)`` is peeled each step. Time
    advances (``changing_additional_information``).

``forward``
    A single perturbed trajectory ``A = M(A)`` from a modified initial condition; saves
    the full state. Faithful to ``LLAT_add_heat_onestep.py``.

Static-variable locking (with dynamic masking) keeps non-prognostic surface channels
equal to the background and zero in ``δ`` unless an experiment claims them.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import xarray as xr

from . import config, io, layout, operators
from .operators import m_operator, STEP_HOURS
from .perturbations import build_perturbation
from .perturbations.base import StepContext


class InitialState:
    """Background DLAMPty initial fields + grid/time metadata for one init time."""

    def __init__(self, upper, surface, dlam_lats, dlam_lons, initial_time):
        self.upper = upper
        self.surface = surface
        self.dlam_lats = dlam_lats
        self.dlam_lons = dlam_lons
        self.initial_time = initial_time


def load_initial_state(tc_id: str, init_str: str, dlampty) -> InitialState:
    """Load the DLAMPty combined-NetCDF background IC (no FCNv2)."""
    grid = config.layout()["grid"]
    ds = xr.open_dataset(str(config.dlampty_ic_path(tc_id, init_str)))
    ds = ds.isel(latitude=np.arange(*grid["ic_lat_slice"]),
                 longitude=np.arange(*grid["ic_lon_slice"]))
    upper, surface = dlampty.IC_from_xarray_to_npy(ds)
    initial_time = _dt.datetime.strptime(init_str, "%Y%m%d%H")
    return InitialState(upper, surface, ds.latitude.values, ds.longitude.values,
                        initial_time)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _zero_locked(surface, active_idx) -> None:
    for idx in active_idx:
        surface[:, :, idx] = 0.0


def _align_statics(surface, ref_surface, active_idx) -> None:
    for idx in active_idx:
        surface[:, :, idx] = ref_surface[:, :, idx]


def _f32(*arrs):
    return tuple(np.ascontiguousarray(a, dtype=np.float32) for a in arrs)


def _upper_lock_indices(cfg):
    """Upper-air channels whose δ is pinned to the control every step.

    ``lock_upper_vars: [q]`` denies the perturbation any moisture co-evolution: the
    perturbed trajectory re-enters each step with the control's q (δq ≡ 0), so the
    response that survives is the part *not* mediated by the moisture channel — a
    probe of how tightly the manifold binds thermodynamics to moisture.
    """
    return [layout.upper_index(v) for v in (cfg.get("lock_upper_vars") or [])]


def _zero_upper_locked(d_up, up_lock) -> None:
    for idx in up_lock:
        d_up[:, :, :, idx] = 0.0


def _resolve_state(cfg, dlampty) -> InitialState:
    """The IC a run starts from: the real analysis, or a prebuilt idealized one.

    ``ic_npz`` (alias ``background_npz``, the key ``driver_vortex`` already uses)
    points at a saved :class:`InitialState` — e.g. the azimuthally averaged vortex
    from ``scripts/make_axisymmetric_ic.py``, or a quiescent background. Building
    the IC ahead of time rather than transforming it here keeps every run in a
    sweep reading the identical bytes, and keeps the transform out of the hot loop.
    """
    p = cfg.get("ic_npz") or cfg.get("background_npz")
    if p:
        from .idealized_vortex.background import load_background
        return load_background(p)
    return load_initial_state(str(cfg["tc_id"]), str(cfg["init_time"]), dlampty)


def _ctx(state, fore_i, lead_hr):
    return StepContext(fore_i=fore_i, lead_hr=lead_hr, initial_time=state.initial_time,
                       dlam_lats=state.dlam_lats, dlam_lons=state.dlam_lons)


def _ic_delta(state, pert, active_idx):
    """Initial perturbation δ0 from an IC intervention (zero for per-step forcing)."""
    a_up = state.upper.copy()
    a_sfc = state.surface.copy()
    a_up, a_sfc = pert.apply_ic(a_up, a_sfc, _ctx(state, 0, 0))
    d_up = a_up - state.upper
    d_sfc = a_sfc - state.surface
    _zero_locked(d_sfc, active_idx)
    return d_up, d_sfc


def _forcing(state, pert, active_idx, fore_i, lead_hr, up_shape, sfc_shape):
    """The per-step constant forcing f for this step (zeros if none)."""
    f_up = np.zeros(up_shape, dtype=np.float32)
    f_sfc = np.zeros(sfc_shape, dtype=np.float32)
    f_up, f_sfc = pert.apply_step(f_up, f_sfc, _ctx(state, fore_i, lead_hr))
    _zero_locked(f_sfc, active_idx)
    return f_up, f_sfc


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def run(cfg: dict, dlampty, out_dir) -> list[str]:
    """Run one experiment described by ``cfg`` and return the saved file paths."""
    mode = cfg["mode"]
    pert = build_perturbation(cfg.get("perturbation", {}))
    active_idx = layout.active_lock_indices(pert.claimed_static_vars())
    state = _resolve_state(cfg, dlampty)
    bundle_dir = io.data_dir(out_dir)

    if mode == "snapshot":
        return _run_snapshot(cfg, dlampty, bundle_dir, state, pert, active_idx)
    if mode == "continuous":
        return _run_continuous(cfg, dlampty, bundle_dir, state, pert, active_idx)
    if mode == "forward":
        return _run_forward(cfg, dlampty, bundle_dir, state, pert, active_idx)
    raise ValueError(f"Unknown mode {mode!r} (snapshot/continuous/forward)")


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _run_snapshot(cfg, dlampty, out_dir, state, pert, active_idx):
    iterations = int(cfg["iterations"])
    up_lock = _upper_lock_indices(cfg)
    init_s = state.initial_time.strftime("%Y%m%d%H")

    u0_up, u0_sfc = _f32(state.upper, state.surface)
    # Background ū = M(u₀), frozen time.
    ubar_up, ubar_sfc = m_operator(u0_up, u0_sfc, dlampty, advance_time=False)
    ubar_up, ubar_sfc = _f32(ubar_up, ubar_sfc)
    # Save the background so diagnostics can form physical absolute/Δ fields (ū + δ).
    io.save_delta_bundle(out_dir / "background", ubar_up, ubar_sfc)

    d_up, d_sfc = _ic_delta(state, pert, active_idx)
    d_up, d_sfc = _f32(d_up, d_sfc)
    _zero_upper_locked(d_up, up_lock)

    saved = []
    for it in range(1, iterations + 1):
        f_up, f_sfc = _forcing(state, pert, active_idx, it, 0, u0_up.shape, u0_sfc.shape)
        base_up, base_sfc = (u0_up, u0_sfc) if it == 1 else (ubar_up, ubar_sfc)

        A_up = np.ascontiguousarray(base_up + d_up + f_up, dtype=np.float32)
        A_sfc = np.ascontiguousarray(base_sfc + d_sfc + f_sfc, dtype=np.float32)
        _align_statics(A_sfc, ubar_sfc, active_idx)

        u_up, u_sfc = m_operator(A_up, A_sfc, dlampty, advance_time=False,
                                 lock_ref=ubar_sfc, lock_idx=active_idx)
        d_up = u_up - ubar_up
        d_sfc = u_sfc - ubar_sfc
        _zero_locked(d_sfc, active_idx)
        _zero_upper_locked(d_up, up_lock)

        base = io.delta_basename("snapshot", pert.param_tag, init_s, it)
        base = base.replace(f"lead{it:03d}hr", f"iter{it:03d}")
        saved.append(io.save_delta_bundle(out_dir / base, d_up, d_sfc))
        print(f"[snapshot] iter {it:02d}/{iterations} | |δT|max="
              f"{np.abs(d_up[:, :, :, layout.upper_index('t')]).max():.3f} | saved {base}")
    return saved


def _run_continuous(cfg, dlampty, out_dir, state, pert, active_idx):
    total_steps = int(cfg["total_steps"])
    up_lock = _upper_lock_indices(cfg)
    init_s = state.initial_time.strftime("%Y%m%d%H")

    I_up, I_sfc = _f32(state.upper, state.surface)
    d_up, d_sfc = _f32(*_ic_delta(state, pert, active_idx))
    _zero_upper_locked(d_up, up_lock)

    saved = []
    for fore_i in range(1, total_steps + 1):
        lead_hr = fore_i * STEP_HOURS
        t = state.initial_time + _dt.timedelta(hours=lead_hr)
        f_up, f_sfc = _forcing(state, pert, active_idx, fore_i, lead_hr,
                               I_up.shape, I_sfc.shape)

        A_up = np.ascontiguousarray(I_up + d_up + f_up, dtype=np.float32)
        A_sfc = np.ascontiguousarray(I_sfc + d_sfc + f_sfc, dtype=np.float32)
        _align_statics(A_sfc, I_sfc, active_idx)

        Ip_up, Ip_sfc = m_operator(I_up, I_sfc, dlampty, advance_time=True, valid_time=t)
        Ap_up, Ap_sfc = m_operator(A_up, A_sfc, dlampty, advance_time=True, valid_time=t,
                                   lock_ref=Ip_sfc, lock_idx=active_idx)
        d_up = Ap_up - Ip_up
        d_sfc = Ap_sfc - Ip_sfc
        _zero_locked(d_sfc, active_idx)
        _zero_upper_locked(d_up, up_lock)
        I_up, I_sfc = Ip_up, Ip_sfc

        base = io.delta_basename("continuous", pert.param_tag, init_s, lead_hr)
        # Save the control trajectory alongside δ so diagnostics can reconstruct
        # physical absolute/Δ fields (control + δ) at the matching lead.
        io.save_delta_bundle(out_dir / base.replace("delta_", "control_"), Ip_up, Ip_sfc)
        saved.append(io.save_delta_bundle(out_dir / base, d_up, d_sfc))
        print(f"[continuous] lead {lead_hr:03d}h | saved {base}")
    return saved


def _run_forward(cfg, dlampty, out_dir, state, pert, active_idx):
    if _upper_lock_indices(cfg):
        raise ValueError("lock_upper_vars needs a control trajectory to pin δ to; "
                         "use continuous or snapshot mode")
    fore_hour = int(cfg["fore_hour"])
    n_steps = fore_hour // STEP_HOURS
    init_s = state.initial_time.strftime("%Y%m%d%H")

    d_up, d_sfc = _ic_delta(state, pert, active_idx)
    A_up = np.ascontiguousarray(state.upper + d_up, dtype=np.float32)
    A_sfc = np.ascontiguousarray(state.surface + d_sfc, dtype=np.float32)

    saved = [io.save_delta_bundle(
        out_dir / io.delta_basename("forward", pert.param_tag, init_s, 0), A_up, A_sfc)]
    for fore_i in range(1, n_steps + 1):
        lead_hr = fore_i * STEP_HOURS
        t = state.initial_time + _dt.timedelta(hours=lead_hr)
        A_up, A_sfc = m_operator(A_up, A_sfc, dlampty, advance_time=True, valid_time=t)
        base = io.delta_basename("forward", pert.param_tag, init_s, lead_hr)
        saved.append(io.save_delta_bundle(out_dir / base, A_up, A_sfc))
        print(f"[forward] lead {lead_hr:03d}h | saved {base}")
    return saved
