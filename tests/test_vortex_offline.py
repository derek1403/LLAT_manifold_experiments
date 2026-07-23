"""Offline tests for the ring-vortex experiment (no DLAMPty model).

Covers the balanced-ring construction (profile shape, gradient-wind residual,
warm core), the RingVortexPerturbation IC injection, the azimuthal-wavenumber
diagnostic on a synthetic m=4 field, quiescent-background flattening, and the
driver_vortex bookkeeping (background npz + frame relaxation + parity of the
snapshot math with the shared driver). Run with pytest, or directly:
``python tests/test_vortex_offline.py``.
"""
from __future__ import annotations

import datetime as _dt
import sys
import tempfile
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llat_manifold import driver, io, layout                        # noqa: E402
from llat_manifold.driver import InitialState                       # noqa: E402
from llat_manifold.idealized_vortex import (azimuthal, background as bg,  # noqa: E402
                                            balance, driver_vortex)
from llat_manifold.idealized_vortex.ring import RingVortexPerturbation  # noqa: E402
from llat_manifold.perturbations.base import StepContext            # noqa: E402

NZ, NY, NX, NUP = 13, 81, 81, 6
NSFC = len(layout.surface_vars()) + 2
P_HPA = np.asarray(layout.pressure_levels(), dtype=float)


class _IdentityDLAMPty:
    def predict_one_step(self, up, sfc):
        return up.copy(), sfc.copy()

    def changing_additional_information(self, up, sfc, t):
        return up, sfc


def _quiet_state():
    """A quiescent-like background: calm winds, uniform thermo, real lat/lon/f."""
    up = np.zeros((NZ, NY, NX, NUP), dtype=np.float32)
    up[:, :, :, layout.upper_index("t")] = 280.0
    up[:, :, :, layout.upper_index("z")] = 5.0e4
    sfc = np.zeros((NY, NX, NSFC), dtype=np.float32)
    sfc[:, :, layout.surface_index("t2m")] = 300.0
    sfc[:, :, layout.surface_index("msl")] = 1.0e5
    sfc[:, :, layout.surface_index("sp")] = 1.0e5
    lats = np.linspace(23.75, 3.75, NY)      # descending, like the real domain
    lons = np.linspace(126.75, 146.75, NX)
    lon2d, lat2d = np.meshgrid(lons, lats)
    sfc[:, :, layout.surface_index("f")] = \
        2 * balance.OMEGA * np.sin(np.radians(lat2d))
    sfc[:, :, -2], sfc[:, :, -1] = lon2d, lat2d
    return InitialState(up, sfc, lats, lons, _dt.datetime(2025, 9, 17, 0))


def _ctx(state):
    return StepContext(fore_i=0, lead_hr=0, initial_time=state.initial_time,
                       dlam_lats=state.dlam_lats, dlam_lons=state.dlam_lons)


# --------------------------------------------------------------------------- #
def test_ring_profile_and_balance():
    """V peaks at ≈vmax near r2; Φ′ satisfies the gradient-wind ODE exactly."""
    ring = balance.build_ring_1d(gamma=0.2, delta=0.7, rmw_deg=2.5, vmax=35.0,
                                 lat_center_deg=13.75, p_hpa=P_HPA)
    r, v = ring["r"], ring["v"]
    assert abs(v.max() - 35.0) < 1e-6
    r_peak = r[np.argmax(v)]
    assert 0.7 * ring["r2"] <= r_peak <= 1.15 * ring["r2"]
    # gradient-wind residual on the 850 hPa level (interior points)
    k = list(P_HPA).index(850.0)
    vl, phi = ring["v_levels"][k], ring["phi_levels"][k]
    dphi_dr = np.gradient(phi, r)
    rhs = vl ** 2 / np.where(r > 0, r, 1.0) + ring["f0"] * vl
    resid = np.abs(dphi_dr - rhs)[5:-5].max() / max(np.abs(rhs).max(), 1e-12)
    assert resid < 5e-3, f"gradient-wind residual {resid}"
    # warm core: T′ > 0 in the eye at mid levels, ~0 far outside
    k500 = list(P_HPA).index(500.0)
    assert ring["t_levels"][k500][0] > 0.05
    assert abs(ring["t_levels"][k500][-1]) < 1e-6
    print("ok test_ring_profile_and_balance")


def test_banded_profile():
    """zeta_bands: alternating-sign ζ(r), balance still exact, |V| peak = vmax."""
    bands = [[0.0, 1.2, 0.1], [1.2, 2.8, 1.0], [2.8, 4.2, -0.35]]
    ring = balance.build_ring_1d(gamma=0.2, delta=0.7, rmw_deg=4.2, vmax=35.0,
                                 lat_center_deg=13.75, p_hpa=P_HPA,
                                 zeta_bands=bands)
    r, zeta = ring["r"], ring["zeta"]
    deg = balance.DEG2M
    assert zeta[np.argmin(np.abs(r - 2.0 * deg))] > 0.9          # positive ring
    assert zeta[np.argmin(np.abs(r - 3.5 * deg))] < -0.25        # negative band
    assert abs(np.abs(ring["v"]).max() - 35.0) < 1e-6
    k = list(P_HPA).index(850.0)
    dphi_dr = np.gradient(ring["phi_levels"][k], r)
    rhs = (ring["v_levels"][k] ** 2 / np.where(r > 0, r, 1.0)
           + ring["f0"] * ring["v_levels"][k])
    resid = np.abs(dphi_dr - rhs)[5:-5].max() / max(np.abs(rhs).max(), 1e-12)
    assert resid < 5e-3, f"gradient-wind residual {resid}"
    print("ok test_banded_profile")


def test_apply_ic_channels():
    """Ring edits u,v,t,z + u10,v10,msl,sp only; q/w/statics untouched; low core."""
    state = _quiet_state()
    pert = RingVortexPerturbation(noise_amp=0.0)
    up, sfc = pert.apply_ic(state.upper.copy(), state.surface.copy(), _ctx(state))
    d_up, d_sfc = up - state.upper, sfc - state.surface
    for name in ("q", "w"):
        assert np.all(d_up[:, :, :, layout.upper_index(name)] == 0.0), name
    for name in ("sst_filled", "hgt", "landmask", "tcwv", "tp"):
        assert np.all(d_sfc[:, :, layout.surface_index(name)] == 0.0), name
    spd850 = np.hypot(d_up[10, :, :, layout.upper_index("u")],
                      d_up[10, :, :, layout.upper_index("v")])
    assert 30.0 < spd850.max() <= 36.0                    # F(850)=1 → ~vmax
    d_msl = d_sfc[:, :, layout.surface_index("msl")]
    assert d_msl[NY // 2, NX // 2] < -100.0               # low at the centre (Pa)
    assert d_msl.min() == d_msl[NY // 2, NX // 2]
    # wind confined by the outer taper: frame ~0 even without boundary relax
    assert spd850[:2, :].max() < 0.2 and spd850[:, :2].max() < 0.2
    print("ok test_apply_ic_channels")


def test_ring_axisymmetric_and_noise_seed():
    """noise_amp=0 → ζ850 has no m≥1 power; noise_amp>0 breaks symmetry slightly."""
    state = _quiet_state()
    k850 = list(P_HPA).index(850.0)
    ui, vi = layout.upper_index("u"), layout.upper_index("v")

    def a_m(noise):
        pert = RingVortexPerturbation(noise_amp=noise, seed=1)
        up, _ = pert.apply_ic(state.upper.copy(), state.surface.copy(), _ctx(state))
        zeta = azimuthal.relative_vorticity(up[k850, :, :, ui], up[k850, :, :, vi],
                                            state.dlam_lats, state.dlam_lons)
        _, _, z_rt = azimuthal.polar_sample(zeta, state.dlam_lats, state.dlam_lons)
        return azimuthal.wavenumber_amplitudes(z_rt).max(axis=0)

    a0, a1 = a_m(0.0), a_m(0.5)
    # a residual few-% m=2/m=4 grid+metric imprint is inherent at 0.25°
    assert a0[1:].max() < 0.05 * a0[0], "IC not axisymmetric"
    assert a1[1:].max() > 2.0 * a0[1:].max(), "noise seed had no effect"
    print("ok test_ring_axisymmetric_and_noise_seed")


def test_azimuthal_recovers_m4():
    """The wavenumber diagnostic recovers a pure synthetic m=4 signal."""
    state = _quiet_state()
    lon2d = state.surface[:, :, -2]
    lat2d = state.surface[:, :, -1]
    r2d, th2d, *_ = balance.domain_polar_geometry(state.dlam_lats, state.dlam_lons)
    field = np.exp(-((r2d / balance.DEG2M - 2.5) / 0.8) ** 2) * np.cos(4 * th2d)
    _, _, f_rt = azimuthal.polar_sample(field, state.dlam_lats, state.dlam_lons)
    amp = azimuthal.wavenumber_amplitudes(f_rt, m_max=8).max(axis=0)
    assert np.argmax(amp[1:]) + 1 == 4
    assert amp[4] > 0.9 and amp[4] > 10 * np.delete(amp[1:], 3).max()
    assert lat2d.shape == lon2d.shape
    print("ok test_azimuthal_recovers_m4")


def test_quiescent_flattening():
    """make_quiescent zeroes winds/terrain, flattens thermo, keeps f + lat/lon."""
    state = _quiet_state()
    rng = np.random.default_rng(3)
    state.upper += rng.standard_normal(state.upper.shape).astype(np.float32)
    state.surface[:, :, :len(layout.surface_vars())] += \
        rng.standard_normal((NY, NX, len(layout.surface_vars()))).astype(np.float32)
    quiet = bg.make_quiescent(state)
    assert np.all(quiet.upper[:, :, :, layout.upper_index("u")] == 0.0)
    t = quiet.upper[:, :, :, layout.upper_index("t")]
    assert np.ptp(t, axis=(1, 2)).max() < 1e-4            # uniform per level
    assert np.all(quiet.surface[:, :, layout.surface_index("landmask")] == 0.0)
    assert np.array_equal(quiet.surface[:, :, layout.surface_index("f")],
                          state.surface[:, :, layout.surface_index("f")])
    assert np.array_equal(quiet.surface[:, :, -1], state.surface[:, :, -1])
    print("ok test_quiescent_flattening")


def test_driver_vortex_bookkeeping():
    """background_npz load, frame relaxation, and snapshot-math parity."""
    state = _quiet_state()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bg_path = bg.save_background(state, d / "quiet.npz")
        cfg = {"mode": "snapshot", "iterations": 2, "background_npz": bg_path,
               "boundary_relax": {"width": 8, "alpha": 1.0},
               "perturbation": {"type": "ring_vortex", "noise_amp": 0.0}}
        saved = driver_vortex.run(cfg, _IdentityDLAMPty(), d / "run")
        assert len(saved) == 2
        d_up, d_sfc = io.load_delta_bundle(saved[-1])
        # frame relaxed to exactly zero
        assert np.all(d_up[:, :8, :, :] == 0.0) and np.all(d_up[:, :, -8:, :] == 0.0)
        assert np.all(d_sfc[-8:, :, :] == 0.0)
        # identity-M parity with the shared snapshot recursion: δ stays the IC delta
        pert = driver_vortex.build_perturbation(cfg["perturbation"])
        ref_up, ref_sfc = driver._ic_delta(state, pert,
                                           layout.active_lock_indices())
        driver_vortex._relax_frame(ref_up, ref_sfc, 8, 1.0)
        assert np.allclose(d_up, ref_up, atol=1e-5)
        assert np.allclose(d_sfc, ref_sfc, atol=1e-5)
        # background bundle saved for the diagnostics' geometry
        assert (d / "run" / "data" / "background.npz").exists()
    print("ok test_driver_vortex_bookkeeping")


if __name__ == "__main__":
    test_ring_profile_and_balance()
    test_banded_profile()
    test_apply_ic_channels()
    test_ring_axisymmetric_and_noise_seed()
    test_azimuthal_recovers_m4()
    test_quiescent_flattening()
    test_driver_vortex_bookkeeping()
    print("\nALL VORTEX OFFLINE TESTS PASSED")
