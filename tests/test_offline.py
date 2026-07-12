"""Offline tests: layout, dynamic masking, heating injection, and driver bookkeeping.

Run WITHOUT the real DLAMPty model by substituting a fake whose step adds a constant
to every surface channel (so static-variable locking is observable). Run with pytest,
or directly: ``python tests/test_offline.py``.
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

from llat_manifold import driver, io, layout                  # noqa: E402
from llat_manifold.driver import InitialState                 # noqa: E402
from llat_manifold.perturbations.heating import (             # noqa: E402
    HeatingPerturbation, vertical_profile, gaussian_centered)
from llat_manifold.perturbations.sst import SSTPerturbation   # noqa: E402

NZ, NY, NX, NUP = 13, 81, 81, 6
NSFC = len(layout.surface_vars()) + 2     # + 2 space-info channels (lat/lon)


class _FakeDLAMPty:
    """M adds +1 to every surface channel; upper unchanged. Time-encodings untouched."""
    def predict_one_step(self, up, sfc):
        return up.copy(), sfc + 1.0

    def changing_additional_information(self, up, sfc, t):
        return up, sfc


class _IdentityDLAMPty:
    """M = identity (a fixed point), so the perturbation recursion is exactly testable."""
    def predict_one_step(self, up, sfc):
        return up.copy(), sfc.copy()

    def changing_additional_information(self, up, sfc, t):
        return up, sfc


def _fake_state():
    rng = np.random.default_rng(0)
    up = rng.standard_normal((NZ, NY, NX, NUP)).astype(np.float32)
    sfc = rng.standard_normal((NY, NX, NSFC)).astype(np.float32)
    lats = np.linspace(30, 10, NY)
    lons = np.linspace(120, 140, NX)
    return InitialState(up, sfc, lats, lons, _dt.datetime(2025, 9, 17, 0))


def test_layout_indices():
    assert layout.surface_index("t2m") == 2
    assert layout.surface_index("sst_filled") == 9
    assert layout.upper_index("t") == 2
    print("ok test_layout_indices")


def test_active_lock_default_and_masked():
    default = layout.active_lock_indices()
    assert sorted(i for i in default if i >= 0) == list(range(9, 18))
    assert set(layout.space_info_indices()).issubset(set(default))
    masked = layout.active_lock_indices(["sst_filled"])
    assert 9 not in masked and 10 in masked
    print("ok test_active_lock_default_and_masked")


def test_heating_ic_matches_onestep():
    """apply_ic must reproduce the LLAT_add_heat_onestep Gaussian heating field."""
    pert = HeatingPerturbation(injection="ic", amp_K=10.0, heat_type="Deep", sigma=5.0)
    upper = np.zeros((NZ, NY, NX, NUP))
    ctx = type("C", (), {})()
    out_up, _ = pert.apply_ic(upper.copy(), np.zeros((NY, NX, NSFC)), ctx)

    v = vertical_profile("Deep", layout.pressure_levels())
    h_xy = gaussian_centered(NX, 5.0)
    ref = np.zeros_like(upper)
    for z in range(NZ):
        ref[z, :, :, 2] += 10.0 * v[z] * h_xy
    assert np.allclose(out_up, ref)
    assert out_up[:, NY // 2, NX // 2, 2].max() > 0
    print("ok test_heating_ic_matches_onestep")


def test_continuous_locking_and_forcing():
    """Static channels zeroed in δ; upper-T forcing enters δ at the expected amplitude."""
    state = _fake_state()
    pert = HeatingPerturbation(injection="per_step", amp_K=4.0, heat_type="Deep",
                               forcing_steps=2, amp_mode="spread", sigma=5.0)
    active = layout.active_lock_indices(pert.claimed_static_vars())
    with tempfile.TemporaryDirectory() as d:
        saved = driver._run_continuous({"total_steps": 1}, _FakeDLAMPty(),
                                       Path(d), state, pert, active)
        d_up, d_sfc = io.load_delta_bundle(saved[0])
        for idx in active:
            assert np.all(d_sfc[:, :, idx] == 0.0), f"static idx {idx} not zeroed"
        # per-step amp = 4/2 = 2; Deep profile peaks at 1 -> δT max ≈ 2 K at the centre.
        t = d_up[:, NY // 2, NX // 2, layout.upper_index("t")]
        assert abs(t.max() - 2.0) < 1e-3
    print("ok test_continuous_locking_and_forcing")


def test_sst_dynamic_mask():
    """A claimed static var (sst_filled) is NOT reset/zeroed; others still are."""
    state = _fake_state()
    pert = SSTPerturbation(delta_K=3.0)
    active = layout.active_lock_indices(pert.claimed_static_vars())
    assert layout.surface_index("sst_filled") not in active
    with tempfile.TemporaryDirectory() as d:
        saved = driver._run_continuous({"total_steps": 1}, _FakeDLAMPty(),
                                       Path(d), state, pert, active)
        _, d_sfc = io.load_delta_bundle(saved[0])
        sst = layout.surface_index("sst_filled")
        assert np.allclose(d_sfc[:, :, sst], 3.0), "claimed SST should persist in δ"
        assert np.all(d_sfc[:, :, layout.surface_index("hgt")] == 0.0)
    print("ok test_sst_dynamic_mask")


def test_snapshot_math():
    """With identity M: ū=u₀ and u'ᵢ=u'ᵢ₋₁+f, so δ grows linearly as i·f."""
    state = _fake_state()
    pert = HeatingPerturbation(injection="per_step", amp_K=1.0, heat_type="Deep",
                               forcing_steps=10, amp_mode="each", sigma=5.0)
    active = layout.active_lock_indices()
    with tempfile.TemporaryDirectory() as d:
        saved = driver._run_snapshot({"iterations": 3}, _IdentityDLAMPty(),
                                     Path(d), state, pert, active)
        assert len(saved) == 3
        d_up, _ = io.load_delta_bundle(saved[-1])     # iteration 3
        t = d_up[:, NY // 2, NX // 2, layout.upper_index("t")]
        # δ_3 = 3·f; Deep profile peaks at 1, amp 1 -> centre column max ≈ 3 K.
        assert abs(t.max() - 3.0) < 1e-3
    print("ok test_snapshot_math")


def test_upper_lock_q():
    """lock_upper_vars: [q] pins δq to 0 each step while the T response is untouched."""
    class _MoistFake:
        """Moisture co-evolves with temperature (q += T), so a T bump grows a δq."""
        def predict_one_step(self, up, sfc):
            out = up.copy()
            out[..., layout.upper_index("q")] += up[..., layout.upper_index("t")]
            return out, sfc.copy()

        def changing_additional_information(self, up, sfc, t):
            return up, sfc

    state = _fake_state()
    pert = HeatingPerturbation(injection="per_step", amp_K=4.0, heat_type="Deep",
                               forcing_steps=2, amp_mode="spread", sigma=5.0)
    active = layout.active_lock_indices(pert.claimed_static_vars())
    q, t = layout.upper_index("q"), layout.upper_index("t")
    with tempfile.TemporaryDirectory() as d:
        for sub in ("free", "lock"):
            (Path(d) / sub).mkdir()
        free = driver._run_continuous({"total_steps": 1}, _MoistFake(),
                                      Path(d) / "free", state, pert, active)
        lock = driver._run_continuous({"total_steps": 1, "lock_upper_vars": ["q"]},
                                      _MoistFake(), Path(d) / "lock", state, pert, active)
        up_f, _ = io.load_delta_bundle(free[0])
        up_l, _ = io.load_delta_bundle(lock[0])
        assert up_f[..., q].max() > 1.0, "moist run should grow a δq"
        assert np.all(up_l[..., q] == 0.0), "locked run must keep δq = 0"
        assert np.allclose(up_l[..., t], up_f[..., t]), "δT must be untouched by the q lock"
    print("ok test_upper_lock_q")


if __name__ == "__main__":
    test_layout_indices()
    test_active_lock_default_and_masked()
    test_heating_ic_matches_onestep()
    test_continuous_locking_and_forcing()
    test_sst_dynamic_mask()
    test_snapshot_math()
    test_upper_lock_q()
    print("\nALL OFFLINE TESTS PASSED")
