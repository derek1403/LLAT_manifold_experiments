"""Axisymmetric idealized vortex built by azimuthally averaging a real analysis IC.

The heating experiments were first run on the raw RAGASA analysis. That state
carries everything a real case carries — monsoon trough, subtropical ridge, an
elliptical eyewall, a sheared outflow — so a "the vortex responded" claim always
has to argue past the possibility that some asymmetric feature of *that particular
day* did the work. Averaging the analysis around the storm centre removes exactly
that degree of freedom while keeping the storm's own radial structure: the same
RMW, the same warm core, the same secondary circulation, no preferred azimuth.

What gets averaged, and what does not
-------------------------------------
Winds are averaged as a **vector field in cylindrical coordinates**: (u, v) is
decomposed into tangential and radial components about the centre, each is
averaged over azimuth, and the pair is rebuilt. Averaging u and v as scalars
instead would annihilate the vortex — the tangential wind changes sign across the
centre, so its Cartesian components integrate to ~zero around a ring.

    upper  u, v            → v̄_t(r), v̄_r(r), rebuilt   (v̄_r kept: the secondary
                                                          circulation is physics,
                                                          not asymmetry)
    upper  t, q, z, w      → radial mean per level
    sfc[0:2]   u10, v10    → same vector decomposition
    sfc[2:9]   t2m, d2m, msl, sp, tcwv, tp, mtnlwrf   → radial mean
    sfc[9:]    sst_filled, f, solar, hgt, landmask, diurnal/doy, lon, lat
                           → LEFT ALONE

That last row is deliberate. Those channels carry information tied to a place on
the Earth, not to a radius from the storm: f must keep its real meridional
gradient (β), the time encodings are recomputed from lat/lon each step anyway, and
the surface boundary is the real one. **Consequence to state plainly in any write-up
using these runs**: the lower boundary is not axisymmetric, so axisymmetry holds
exactly at t = 0 and is eroded from the first step onward — most visibly for init
2025092000, whose domain contains ~6 % land and terrain up to ~1.9 km.

Geometry
--------
The centre is the domain centre. These are TC-following domains, so the storm is
already there (verified: the analysis MSL minimum sits on the centre grid point for
2025092000 and within ~0.5° for 2025091700), and using it keeps the IC centre, the
Gaussian injection centre and the diagnostics' core mask all on the same point.

Azimuthal coverage is complete only out to :data:`R_MAX_DEG` — the domain is
±10° square, so a ring at 12° only intersects it near the corners. Beyond that
radius the profile is held at its :data:`R_MAX_DEG` value rather than averaging a
few corner-direction points and smearing their bias around a full circle.

I/O reuses :mod:`.background` (``save_background`` / ``load_background``), which is
already a generic ``InitialState`` npz reader/writer.
"""
from __future__ import annotations

import numpy as np

from .. import driver, layout

# Azimuthal coverage is 100 % only inside this radius on a ±10° square domain.
R_MAX_DEG = 10.0
DEG_KM = 111.32
# Radial bin width. ``None`` = one grid spacing, measured off the grid itself.
# Finer than that and the inner bins catch zero or one cell, which turns the first
# few hundred km of the profile into interpolation noise instead of a vortex.
DR_KM = None

# Surface channel policy, by index into ``layout.surface_vars()`` + the appended
# lat/lon. Kept as slices so it reads against the model layout at a glance.
SFC_VECTOR = slice(0, 2)           # u10, v10  — cylindrical decomposition
SFC_SCALAR = slice(2, 9)           # t2m, d2m, msl, sp, tcwv, tp, mtnlwrf
SFC_KEEP = slice(9, None)          # sst_filled, f, solar, hgt, landmask, time, lon/lat


def radial_frame(lat2d, lon2d, *, dr_km: float | None = DR_KM):
    """(r2d [km], θ2d, bin edges [km], bin centres [km]) about the domain centre."""
    lat2d = np.asarray(lat2d, dtype=float)
    lon2d = np.asarray(lon2d, dtype=float)
    cy, cx = lat2d.shape[0] // 2, lat2d.shape[1] // 2
    coslat = np.cos(np.deg2rad(lat2d[cy, cx]))
    dy = (lat2d - lat2d[cy, cx]) * DEG_KM
    dx = (lon2d - lon2d[cy, cx]) * DEG_KM * coslat
    r2d = np.hypot(dy, dx)
    theta = np.arctan2(dy, dx)
    if dr_km is None:
        dr_km = float(abs(lon2d[cy, cx + 1] - lon2d[cy, cx]) * DEG_KM * coslat)
    edges = np.arange(0.0, R_MAX_DEG * DEG_KM + dr_km, dr_km)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return r2d, theta, edges, centers


def radial_mean(f2d, r2d, edges):
    """Azimuthal mean of a 2-D field per radial bin; empty bins are interpolated."""
    prof = np.full(edges.size - 1, np.nan)
    idx = np.digitize(r2d.ravel(), edges) - 1
    flat = np.asarray(f2d, dtype=float).ravel()
    for b in range(prof.size):
        sel = idx == b
        if sel.any():
            prof[b] = flat[sel].mean()
    if np.isnan(prof).any():                     # bins too thin to catch a cell
        good = ~np.isnan(prof)
        centers = np.arange(prof.size, dtype=float)
        prof = np.interp(centers, centers[good], prof[good])
    return prof


def expand(prof, centers, r2d):
    """Paint a radial profile back onto the 2-D grid (flat beyond R_MAX_DEG)."""
    # np.interp clamps to the end values outside the range, which is exactly the
    # "hold the last fully sampled ring" rule we want past R_MAX_DEG.
    return np.interp(np.asarray(r2d, dtype=float), centers, prof)


def _axisym_scalar(f2d, r2d, edges, centers):
    return expand(radial_mean(f2d, r2d, edges), centers, r2d)


def _axisym_vector(u2d, v2d, r2d, theta, edges, centers):
    """Azimuthally average (u, v) through the tangential/radial decomposition."""
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    vt = -u2d * sin_t + v2d * cos_t
    vr = u2d * cos_t + v2d * sin_t
    vt_a = _axisym_scalar(vt, r2d, edges, centers)
    vr_a = _axisym_scalar(vr, r2d, edges, centers)
    return vr_a * cos_t - vt_a * sin_t, vr_a * sin_t + vt_a * cos_t


def axisymmetrize(state: "driver.InitialState", *, dr_km: float = DR_KM
                  ) -> "driver.InitialState":
    """Return a new InitialState whose atmosphere is axisymmetric about the centre.

    See the module docstring for the channel policy — in particular that
    ``sfc[9:]`` (SST, Coriolis, terrain, land mask, time encodings, lat/lon) is
    copied through untouched.
    """
    up = np.array(state.upper, dtype=float, copy=True)
    sfc = np.array(state.surface, dtype=float, copy=True)
    lon2d, lat2d = sfc[:, :, -2], sfc[:, :, -1]
    r2d, theta, edges, centers = radial_frame(lat2d, lon2d, dr_km=dr_km)

    ui, vi = layout.upper_index("u"), layout.upper_index("v")
    scalars = [layout.upper_index(n) for n in ("t", "q", "z", "w")]
    for k in range(up.shape[0]):
        up[k, :, :, ui], up[k, :, :, vi] = _axisym_vector(
            up[k, :, :, ui], up[k, :, :, vi], r2d, theta, edges, centers)
        for i in scalars:
            up[k, :, :, i] = _axisym_scalar(up[k, :, :, i], r2d, edges, centers)

    sfc[:, :, 0], sfc[:, :, 1] = _axisym_vector(
        sfc[:, :, 0], sfc[:, :, 1], r2d, theta, edges, centers)
    for i in range(SFC_SCALAR.start, SFC_SCALAR.stop):
        sfc[:, :, i] = _axisym_scalar(sfc[:, :, i], r2d, edges, centers)
    # sfc[SFC_KEEP] is already the untouched copy.

    return driver.InitialState(up.astype(state.upper.dtype),
                               sfc.astype(state.surface.dtype),
                               state.dlam_lats, state.dlam_lons,
                               state.initial_time)


def transplant_environment(state: "driver.InitialState",
                           env: "driver.InitialState") -> "driver.InitialState":
    """Put an axisymmetric vortex into a *reference* environment, holding the
    environment fixed as a controlled variable across a set of vortices.

    A batch of vortices from different days of one storm sit at different places,
    over different SST, at different f — so comparing their heating response would
    confound "the vortex intensified" with "the environment changed underneath it".
    This copies the reference IC's environment onto ``state`` so that only the
    vortex differs from one member to the next:

    * ``sfc[9:]`` (SST, Coriolis f, terrain, land mask, radiation, time encodings,
      lat/lon) ← taken from ``env``.  The model re-derives f / hgt / landmask /
      solar / diurnal / doy from lat-lon and the valid time every step, so fixing
      lat/lon + ``initial_time`` fixes all of those; ``sst_filled`` is *not*
      re-derived, so it has to come across from ``env`` here.
    * grid metadata and ``initial_time`` ← taken from ``env``, so the per-step
      recomputation runs on the reference day's clock and coordinates.

    The atmosphere (``upper`` + ``sfc[0:9]``) is left exactly as ``state`` has it —
    the vortex array is not re-gridded, only relabelled onto the reference frame,
    so it stays bit-identical to what :func:`axisymmetrize` produced. Call *after*
    :func:`axisymmetrize`. The small cost is that a vortex extracted at one latitude
    now feels the reference latitude's f; over this storm's few-degree track that is
    a percent-level imbalance the run adjusts in the first steps, and it hits the
    control and perturbed members identically so it cancels in ``δ``.
    """
    sfc = np.array(state.surface, copy=True)
    sfc[:, :, SFC_KEEP] = env.surface[:, :, SFC_KEEP]
    return driver.InitialState(state.upper, sfc.astype(state.surface.dtype),
                               np.array(env.dlam_lats), np.array(env.dlam_lons),
                               env.initial_time)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def asymmetry(field2d, lats, lons, *, nr: int = 64, ntheta: int = 72) -> float:
    """Azimuthal variability of a field, as a fraction of its total spread.

    An axisymmetric field is invariant under rotation about the centre, so the
    honest test is to resample onto a polar grid and ask how much the field varies
    *along* θ at fixed r. Returns the ring-averaged θ-standard-deviation divided by
    the field's overall standard deviation: 0 = perfectly axisymmetric, ~1 = the
    azimuthal structure is as large as the radial structure.

    Note this must never be applied to a Cartesian wind component: for a purely
    tangential vortex, u = −v_t·sinθ varies fully with azimuth *by definition*.
    Decompose to v_t / v_r first (:func:`tangential_profile` does).
    """
    from .azimuthal import polar_sample     # reuse the polar resampler

    _r, _th, f_rt = polar_sample(np.asarray(field2d, dtype=float), lats, lons,
                                 nr=nr, ntheta=ntheta, rmax_deg=R_MAX_DEG)
    spread = float(f_rt.std())
    if spread <= 0.0:
        return 0.0
    return float(f_rt.std(axis=1).mean() / spread)


def _sfc_grid(sfc):
    """(lats descending, lons ascending) 1-D vectors from a surface bundle."""
    return sfc[:, 0, -1], sfc[0, :, -2]


def tangential_profile(upper, sfc, level_hpa: int = 850, *,
                       dr_km: float | None = DR_KM):
    """(radius [km], azimuthal-mean V_t [m/s]) at one level — RMW / peak-wind check."""
    lon2d, lat2d = sfc[:, :, -2], sfc[:, :, -1]
    r2d, theta, edges, centers = radial_frame(lat2d, lon2d, dr_km=dr_km)
    k = layout.pressure_levels().index(level_hpa)
    u = upper[k, :, :, layout.upper_index("u")]
    v = upper[k, :, :, layout.upper_index("v")]
    vt = -u * np.sin(theta) + v * np.cos(theta)
    return centers, radial_mean(vt, r2d, edges)


def vt_field(upper, sfc, level_hpa: int = 850):
    """Tangential wind as a 2-D field — the wind quantity :func:`asymmetry` accepts."""
    lon2d, lat2d = sfc[:, :, -2], sfc[:, :, -1]
    _r2d, theta, _e, _c = radial_frame(lat2d, lon2d)
    k = layout.pressure_levels().index(level_hpa)
    return (-upper[k, :, :, layout.upper_index("u")] * np.sin(theta)
            + upper[k, :, :, layout.upper_index("v")] * np.cos(theta))


def rz_sections(upper, sfc, *, dr_km: float | None = DR_KM):
    """Azimuthal-mean (radius, pressure) sections of ζ, V_t and θ.

    Returns ``(r_km, p_hPa, zeta, vt, theta)``. The three fields a reader needs to
    judge whether an IC is still a typhoon: a vorticity core, a wind maximum at a
    plausible RMW, and a warm core in θ.
    """
    from .azimuthal import relative_vorticity

    lon2d, lat2d = sfc[:, :, -2], sfc[:, :, -1]
    lats, lons = _sfc_grid(sfc)
    r2d, theta, edges, centers = radial_frame(lat2d, lon2d, dr_km=dr_km)
    p = np.asarray(layout.pressure_levels(), dtype=float)[: upper.shape[0]]
    ui, vi, ti = (layout.upper_index(k) for k in ("u", "v", "t"))

    zeta, vt, th = [], [], []
    for k in range(upper.shape[0]):
        u2, v2 = upper[k, :, :, ui], upper[k, :, :, vi]
        zeta.append(radial_mean(relative_vorticity(u2, v2, lats, lons), r2d, edges))
        vt.append(radial_mean(-u2 * np.sin(theta) + v2 * np.cos(theta), r2d, edges))
        th.append(radial_mean(upper[k, :, :, ti] * (1000.0 / p[k]) ** (287.05 / 1004.0),
                              r2d, edges))
    return centers, p, np.stack(zeta), np.stack(vt), np.stack(th)


def summary(before: "driver.InitialState", after: "driver.InitialState") -> dict:
    """Numbers worth printing after building an IC: symmetry, RMW, peak wind, MSL."""
    ti, qi = layout.upper_index("t"), layout.upper_index("q")
    k850 = layout.pressure_levels().index(850)
    out = {}
    for tag, st in (("before", before), ("after", after)):
        lats, lons = _sfc_grid(st.surface)
        r, vt = tangential_profile(st.upper, st.surface)
        i = int(np.nanargmax(vt))
        cy, cx = st.surface.shape[0] // 2, st.surface.shape[1] // 2
        out[tag] = {
            "vt_max_ms": float(vt[i]),
            "rmw_km": float(r[i]),
            "msl_center_hPa": float(st.surface[cy, cx, layout.surface_index("msl")]) / 100.0,
            "asym_t850": asymmetry(st.upper[k850, :, :, ti], lats, lons),
            "asym_q850": asymmetry(st.upper[k850, :, :, qi], lats, lons),
            "asym_vt850": asymmetry(vt_field(st.upper, st.surface, 850), lats, lons),
        }
    # channels that must be bit-identical
    out["sfc_keep_identical"] = bool(
        np.array_equal(before.surface[:, :, SFC_KEEP], after.surface[:, :, SFC_KEEP]))
    return out
