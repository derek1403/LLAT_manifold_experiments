"""Shared data layer for the snapshot (semi-linear power-iteration) runs.

Every figure in ``myplot/`` was first written against ``continuous`` runs, which store
a *control and a delta per lead* and let a TC-following domain march the storm across
the map. The experiments this study is actually about are ``snapshot`` runs, and their
storage — and their physics — differ in ways every figure has to respect:

* **one background, not a control per step.** ``driver._run_snapshot`` computes
  ū = M(u₀) once with ``advance_time=False``, writes it as ``background.npz``, and
  measures every iteration against *that*. So the reference field, the radial frame,
  and the background PV are computed **once** and reused — there is no per-iteration
  control to load.
* **nothing moves.** Each iteration is realigned to **u₀** (``_align_statics(A_sfc,
  u0_sfc, …)``, ``lock_ref=u0_sfc``), so lat/lon, SST, f, terrain and the land mask are
  byte-identical at every n *and* equal to the analysis rather than to DLAMPty's own
  prediction of them. The vortex never travels, never makes landfall, and the two ladder
  members stay in the *same* environment for the whole experiment — which is the control
  the ladder was designed around.
* **the base inside M is u₀ at every iteration**, so δ_n = M(u₀ + δ_{n-1} + f) − M(u₀)
  is a repeated application of one operator about one state. f = 0 therefore gives
  δ ≡ 0; whatever an amp = 0 run contains is the noise floor (:func:`null_run`) and
  nothing is subtracted. Both properties date from 2026-08-04; see
  ``docs/perturbation_method.md`` §3.1.1 for what they replaced and why it mattered.
* **n is an iteration count, not a clock.** There is no valid time to advance, so a
  figure must never label these columns in hours. :func:`label` exists to make that
  hard to get wrong.

The public surface is deliberately shaped like the continuous loaders it replaces:
:func:`panels` returns ``{n: (r_km, p_hPa, azimuthal-mean ΔPV)}``, the same triple
``fig_B2_pv_evolution._panels`` returns, so the drawing code above it barely changes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import _data as D
from llat_manifold import io, layout
from llat_manifold.diagnostics import _idealized as _id
from llat_manifold.idealized_vortex import axisymmetric as ax
from llat_manifold.perturbations.heating import _amp_tag

FAMILY = "heating_moist"
TAG = "tseries_{amp}_iter{iters}_init{init}"
IC = "axisym"
ITERS = 40                     # iterations the sweep was run for
FORCING_ITERS = 8              # amp_K is spread over the first 8 iterations
R_MAX_KM = 500.0
NULL_AMP = 0                   # the zero-forcing run — the noise floor, see null_run()


def run_dir(amp, init, *, iters: int = ITERS, family: str = FAMILY) -> Path:
    """Locate one snapshot run, e.g. tseries_5K_iter40_init2025092000."""
    return D.run(family, TAG.format(amp=_amp_tag(amp), iters=iters, init=init), ic=IC)


def null_run(run) -> Path | None:
    """The amp = 0 twin of a run — its **noise floor** — or None if this *is* it.

    Since the 2026-08-04 driver fix (``docs/perturbation_method.md`` §3.1.1) the base
    inside M is u₀ at every iteration, so f = 0 gives δ ≡ 0 analytically and this run
    contains nothing but float32 round-off amplified by the iteration: 1.5e-5 K at
    n = 1 rising to 2.8e-2 K at n = 40, a growth of about 1.2 per iteration. That is
    orders of magnitude below a 5 K response, so it is **not subtracted** — it is the
    level below which a feature means nothing, and figures quote it rather than remove
    it.

    Before that fix the amp = 0 run was not a noise floor at all: iterations i ≥ 2 were
    based on ū rather than u₀, so every δ carried ``M(ū) − ū`` and its descendants. That
    drift is forcing-independent, hence sign-independent, hence indistinguishable from
    nonlinearity in a ±A test — and it dominated: the old 0 K and 5 K runs agreed to
    within 2 % (16.645 vs 16.640 K at n = 40). Those runs are archived under
    ``outputs/diabatic_heating_axisym/_archive_snapshot_pre_u0fix/``.
    """
    run = Path(run)
    parts = run.name.split("_")
    if len(parts) < 2:
        return None
    if parts[1] == _amp_tag(NULL_AMP):
        return None
    parts[1] = _amp_tag(NULL_AMP)
    twin = run.parent / "_".join(parts)
    return twin if (twin / "data" / "background.npz").exists() else None


def label(n: int) -> str:
    """Column label for an iteration. Never 'hours' — these runs have no clock."""
    return f"n = {n}"


def forcing_on(n: int, forcing_iters: int = FORCING_ITERS) -> bool:
    return n <= forcing_iters


@lru_cache(maxsize=32)
def _background(run: Path):
    """(upper, surface) of the frozen background ū — the reference for every n."""
    p = Path(run) / "data" / "background.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — is this a snapshot run?")
    return io.load_delta_bundle(p)


@lru_cache(maxsize=32)
def _frame(run: Path):
    """(r2d, θ2d, bin edges, bin centres) — fixed, because the background is fixed."""
    _up, sfc = _background(run)
    return ax.radial_frame(sfc[:, :, -1], sfc[:, :, -2])


@lru_cache(maxsize=32)
def _pv_background(run: Path):
    """Ertel PV of ū, in PVU. Computed once per run and reused by every iteration."""
    up, sfc = _background(run)
    u, v, t, _z, _w, f, lat2d, lon2d, p_hpa = _id.fields_from_bundle(up, sfc)
    return _id.calculate_pv_spherical(u, v, t, f, p_hpa, lat2d, lon2d)


def iterations(run) -> list[int]:
    """Which iterations this run actually has on disk, ascending."""
    out = []
    for p in sorted((Path(run) / "data").glob("delta_snapshot_*_iter*.npz")):
        try:
            out.append(int(p.stem.rsplit("iter", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def delta(run, n: int):
    """(δ upper, δ surface) at iteration ``n``."""
    hits = sorted((Path(run) / "data").glob(f"delta_snapshot_*_iter{n:03d}.npz"))
    if not hits:
        raise FileNotFoundError(f"no iteration {n} bundle under {run}")
    return io.load_delta_bundle(hits[0])


def azimuthal(field3d, run):
    """Azimuthal mean of a (level, y, x) field on the run's fixed radial frame."""
    r2d, _th, edges, centers = _frame(Path(run))
    return centers, np.stack([ax.radial_mean(field3d[k], r2d, edges)
                              for k in range(field3d.shape[0])])


def pressure():
    return np.asarray(layout.pressure_levels(), dtype=float)


def _raw_dpv(run, n: int) -> np.ndarray:
    """PV(ū + δ_n) − PV(ū), before the null drift is removed.

    Both bundles are added before the PV is taken — the surface delta is what carries
    the grid the metric needs, and in a pure δ the prescribed channels are zero, so
    ``ū + δ`` restores exactly the background's f and lat/lon.
    """
    b_up, b_sfc = _background(Path(run))
    d_up, d_sfc = delta(run, n)
    u, v, t, _z, _w, f, lat2d, lon2d, p_hpa = _id.fields_from_bundle(
        b_up + d_up, b_sfc + d_sfc)
    return _id.calculate_pv_spherical(u, v, t, f, p_hpa, lat2d, lon2d) \
        - _pv_background(Path(run))


def dpv_field(run, n: int, *, subtract_null: bool = False) -> np.ndarray:
    """ΔPV(level, y, x) at iteration ``n``, in PVU, attributable to the forcing.

    Nothing is subtracted: with the base pinned to u₀ the iteration has no
    forcing-independent drift to remove, and the amp = 0 run is a noise floor to quote,
    not a background to peel off (:func:`null_run`). ``subtract_null=True`` is kept only
    for re-reading the archived pre-fix runs, where it was mandatory.
    """
    raw = _raw_dpv(run, n)
    if not subtract_null:
        return raw
    twin = null_run(run)
    return raw if twin is None else raw - _raw_dpv(twin, n)


def noise_floor(run, n: int) -> float:
    """RMS ΔPV (PVU) of the amp = 0 twin at iteration ``n`` — 0.0 if there is none.

    What a figure quotes to say "below this, a feature is round-off".
    """
    twin = null_run(run)
    return 0.0 if twin is None else float(np.sqrt((_raw_dpv(twin, n) ** 2).mean()))


def panels(amp, init, ns, *, iters: int = ITERS, family: str = FAMILY,
           subtract_null: bool = False):
    """{n: (r_km, p_hPa, azimuthal-mean ΔPV)} — the shape the continuous loaders had."""
    run = run_dir(amp, init, iters=iters, family=family)
    p = pressure()
    out = {}
    for n in ns:
        try:
            r_km, az = azimuthal(dpv_field(run, n, subtract_null=subtract_null), run)
        except FileNotFoundError:
            continue
        out[n] = (r_km, p, az)
    if not out:
        raise FileNotFoundError(f"no requested iterations under {run}")
    return out


def window(r_km, p_hpa, r_max: float = R_MAX_KM):
    """Index box for the plotted region — 200 hPa ceiling, r ≤ r_max."""
    return np.ix_(np.asarray(p_hpa) >= _id.P_TOP_HPA, np.asarray(r_km) <= r_max)


__all__ = ["FAMILY", "TAG", "IC", "ITERS", "FORCING_ITERS", "R_MAX_KM",
           "run_dir", "label", "forcing_on", "iterations", "delta", "azimuthal",
           "pressure", "dpv_field", "panels", "window", "null_run", "noise_floor",
           "NULL_AMP"]
