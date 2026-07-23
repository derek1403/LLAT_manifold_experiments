"""Snapshot driver for the ring-vortex experiment (isolated from ``driver.py``).

⚠ This module **copies the snapshot-loop math** of
``llat_manifold.driver._run_snapshot`` (kept byte-for-byte where possible and
verified by ``tests/test_vortex_offline.py``). It exists so the shared driver
stays untouched (user requirement). If a bug is ever fixed in
``driver._run_snapshot``, mirror it here.

Additions over the shared snapshot mode:

  1. ``background_npz`` config key — load a custom quiescent background
     (:mod:`.background`) instead of the real-analysis IC.
  2. ``boundary_relax: {width: 8, alpha: 1.0}`` — after each iteration the
     outer ``width``-pixel frame of the perturbation δ is multiplied by
     (1−alpha); alpha=1 pins the frame to the background (the regional model
     free-runs otherwise; pattern after Part 1's ``era5_oneway._apply_boundary``).
  3. The perturbation is built directly as :class:`.ring.RingVortexPerturbation`
     (``perturbation.type: ring_vortex``) — the shared registry is not touched.
"""
from __future__ import annotations

import numpy as np

from .. import io, layout
from ..driver import (_align_statics, _f32, _forcing, _ic_delta, _zero_locked,
                      load_initial_state)
from ..operators import m_operator
from . import background as bg
from .ring import RingVortexPerturbation


def build_perturbation(pert_cfg: dict) -> RingVortexPerturbation:
    kwargs = dict(pert_cfg or {})
    ptype = kwargs.pop("type", "ring_vortex")
    if ptype != "ring_vortex":
        raise ValueError(f"driver_vortex only builds 'ring_vortex', got {ptype!r}")
    return RingVortexPerturbation(**kwargs)


def _relax_frame(d_up, d_sfc, width: int, alpha: float) -> None:
    """In-place: damp the outer ``width``-pixel frame of δ by (1−alpha)."""
    if width <= 0 or alpha <= 0.0:
        return
    keep = 1.0 - float(alpha)
    w = int(width)
    for sl in ((slice(None, w), slice(None)), (slice(-w, None), slice(None)),
               (slice(None), slice(None, w)), (slice(None), slice(-w, None))):
        d_up[:, sl[0], sl[1], :] *= keep
        d_sfc[sl[0], sl[1], :] *= keep


def run(cfg: dict, dlampty, out_dir) -> list[str]:
    """Run the ring-vortex snapshot experiment; returns saved bundle paths."""
    if cfg.get("mode", "snapshot") != "snapshot":
        raise ValueError("driver_vortex supports snapshot mode only")
    pert = build_perturbation(cfg.get("perturbation", {}))
    active_idx = layout.active_lock_indices(pert.claimed_static_vars())

    if cfg.get("background_npz"):
        state = bg.load_background(cfg["background_npz"])
    else:
        state = load_initial_state(str(cfg["tc_id"]), str(cfg["init_time"]), dlampty)

    brelax = cfg.get("boundary_relax") or {}
    b_width = int(brelax.get("width", 0))
    b_alpha = float(brelax.get("alpha", 1.0))

    bundle_dir = io.data_dir(out_dir)
    iterations = int(cfg["iterations"])
    init_s = state.initial_time.strftime("%Y%m%d%H")

    # ---- snapshot loop: mirrors driver._run_snapshot (frozen valid time) ---- #
    u0_up, u0_sfc = _f32(state.upper, state.surface)
    ubar_up, ubar_sfc = m_operator(u0_up, u0_sfc, dlampty, advance_time=False)
    ubar_up, ubar_sfc = _f32(ubar_up, ubar_sfc)
    io.save_delta_bundle(bundle_dir / "background", ubar_up, ubar_sfc)

    d_up, d_sfc = _f32(*_ic_delta(state, pert, active_idx))
    _relax_frame(d_up, d_sfc, b_width, b_alpha)

    ui = layout.upper_index("u")
    saved = []
    for it in range(1, iterations + 1):
        f_up, f_sfc = _forcing(state, pert, active_idx, it, 0,
                               u0_up.shape, u0_sfc.shape)
        base_up, base_sfc = (u0_up, u0_sfc) if it == 1 else (ubar_up, ubar_sfc)

        A_up = np.ascontiguousarray(base_up + d_up + f_up, dtype=np.float32)
        A_sfc = np.ascontiguousarray(base_sfc + d_sfc + f_sfc, dtype=np.float32)
        _align_statics(A_sfc, ubar_sfc, active_idx)

        u_up, u_sfc = m_operator(A_up, A_sfc, dlampty, advance_time=False,
                                 lock_ref=ubar_sfc, lock_idx=active_idx)
        d_up = u_up - ubar_up
        d_sfc = u_sfc - ubar_sfc
        _zero_locked(d_sfc, active_idx)
        _relax_frame(d_up, d_sfc, b_width, b_alpha)

        base = io.delta_basename("snapshot", pert.param_tag, init_s, it)
        base = base.replace(f"lead{it:03d}hr", f"iter{it:03d}")
        saved.append(io.save_delta_bundle(bundle_dir / base, d_up, d_sfc))
        print(f"[vortex-snapshot] iter {it:02d}/{iterations} | "
              f"|δu|max={np.abs(d_up[:, :, :, ui]).max():6.2f} m/s | saved {base}")
    return saved
