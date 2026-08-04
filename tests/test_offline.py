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


class _DriftingDLAMPty:
    """M is deterministic but has **no** fixed point: ``up → 0.9·up + 0.1``.

    The identity fake cannot see the two snapshot bugs fixed on 2026-08-04, because
    with M = I the background ū equals u₀ and the model's own "static" output equals
    the input — exactly the two coincidences that hid them. This fake breaks both:
    ū ≠ u₀ (so basing the iteration on ū is distinguishable) and the surface channels
    are returned +1 (so adopting the model's prescribed channels is distinguishable).
    """
    def predict_one_step(self, up, sfc):
        return (0.9 * up + 0.1).astype(np.float32), (sfc + 1.0).astype(np.float32)

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


def test_snapshot_null_run_is_exactly_zero():
    """f = 0 must give δ ≡ 0 at every iteration, for a model with no fixed point.

    This is the property the old recursion lacked. It based iterations i ≥ 2 on ū and
    pinned the prescribed channels to the model's own output, so a zero-forcing run
    accumulated M(ū) − ū instead of returning ū; on the real model that spurious drift
    reached 0.70 PVU by i = 4 and, being sign-independent, was read as nonlinearity by
    the ±A antisymmetry test. M is deterministic: with no forcing there is nothing for
    δ to be.
    """
    state = _fake_state()
    pert = HeatingPerturbation(injection="per_step", amp_K=0.0, heat_type="Deep",
                               forcing_steps=8, amp_mode="spread", sigma=5.0)
    active = layout.active_lock_indices()
    with tempfile.TemporaryDirectory() as d:
        saved = driver._run_snapshot({"iterations": 4}, _DriftingDLAMPty(),
                                     Path(d), state, pert, active)
        for i, p in enumerate(saved, start=1):
            d_up, d_sfc = io.load_delta_bundle(p)
            assert np.all(d_up == 0.0), f"iteration {i}: |δ_up|max={np.abs(d_up).max()}"
            assert np.all(d_sfc == 0.0), f"iteration {i}: |δ_sfc|max={np.abs(d_sfc).max()}"
    print("ok test_snapshot_null_run_is_exactly_zero")


def test_snapshot_statics_pinned_to_initial_state():
    """ū carries u₀'s prescribed channels, not the model's prediction of them.

    DLAMPty predicts terrain/land mask/lat/lon/time-encodings along with everything
    else and gets them wrong (terrain came back 45 % flatter). ū is what every δ is
    measured against and what diagnostics rebuild ū + δ from, so it has to hold the
    true grid and lower boundary.
    """
    state = _fake_state()
    pert = HeatingPerturbation(injection="per_step", amp_K=2.0, heat_type="Deep",
                               forcing_steps=4, amp_mode="spread", sigma=5.0)
    active = layout.active_lock_indices()
    with tempfile.TemporaryDirectory() as d:
        driver._run_snapshot({"iterations": 2}, _DriftingDLAMPty(),
                             Path(d), state, pert, active)
        _bg_up, bg_sfc = io.load_delta_bundle(Path(d) / "background.npz")
    for idx in active:
        assert np.allclose(bg_sfc[:, :, idx], state.surface[:, :, idx]), \
            f"background static idx {idx} drifted with the model output"
    # the prognostic channels *should* carry the +1 the fake step applies
    assert np.allclose(bg_sfc[:, :, 0], state.surface[:, :, 0] + 1.0)
    print("ok test_snapshot_statics_pinned_to_initial_state")


def test_snapshot_recursion_uses_u0_as_base():
    """δᵢ = M(u₀ + δᵢ₋₁ + f) − M(u₀), checked against the closed form.

    For ``up → 0.9·up + 0.1`` the recursion collapses to δᵢ = 0.9(δᵢ₋₁ + f), i.e.
    δ₃ = 2.439·f. Basing i ≥ 2 on ū instead gives a different, larger number, so this
    pins the base state rather than merely the growth.
    """
    state = _fake_state()
    pert = HeatingPerturbation(injection="per_step", amp_K=1.0, heat_type="Deep",
                               forcing_steps=10, amp_mode="each", sigma=5.0)
    active = layout.active_lock_indices()
    with tempfile.TemporaryDirectory() as d:
        saved = driver._run_snapshot({"iterations": 3}, _DriftingDLAMPty(),
                                     Path(d), state, pert, active)
        expect = 0.0
        for p in saved:                      # δᵢ = 0.9 (δᵢ₋₁ + f), f peaks at 1 K
            expect = 0.9 * (expect + 1.0)
        d_up, _ = io.load_delta_bundle(saved[-1])
        t = d_up[:, NY // 2, NX // 2, layout.upper_index("t")]
        assert abs(t.max() - expect) < 1e-3, f"δ₃ max {t.max():.4f} != {expect:.4f}"
    print("ok test_snapshot_recursion_uses_u0_as_base")


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


def test_moisture_injection():
    """δq-only forcing hits the q channel at the scaled amplitude; T stays at zero."""
    from llat_manifold.perturbations.moisture import MoisturePerturbation, LATENT_GKG_PER_K

    state = _fake_state()
    prof = [0.0] * 13
    prof[9] = 1.0                                   # measured-style: peak at 700 hPa
    pert = MoisturePerturbation(injection="per_step", amp_K=4.0, gkg_per_K=0.5,
                                profile=prof, forcing_steps=2, amp_mode="spread",
                                sigma=5.0)
    assert pert.param_tag == "4K_dqM_2steps"
    active = layout.active_lock_indices(pert.claimed_static_vars())
    q, t = layout.upper_index("q"), layout.upper_index("t")
    with tempfile.TemporaryDirectory() as d:
        saved = driver._run_continuous({"total_steps": 1}, _IdentityDLAMPty(),
                                       Path(d), state, pert, active)
        d_up, _ = io.load_delta_bundle(saved[0])
        # per-step amp = 4/2 = 2 equivalent K × 0.5 g/kg/K = 1 g/kg = 1e-3 kg/kg peak
        # (bundles are float32, so compare at 1e-8).
        centre = d_up[:, NY // 2, NX // 2, q]
        assert abs(centre[9] - 1.0e-3) < 1e-8 and abs(centre.max() - 1.0e-3) < 1e-8
        assert np.all(d_up[..., t] == 0.0), "δq-only forcing must not touch T"
    # Latent-equivalent default: cp/Lv ≈ 0.4016 g/kg per K, Deep profile, tag dqL.
    lat = MoisturePerturbation(injection="per_step", amp_K=5.0, forcing_steps=8,
                               amp_mode="spread", sigma=5.0)
    assert lat.param_tag == "5K_dqL_8steps"
    assert abs(LATENT_GKG_PER_K - 0.4016) < 1e-4
    print("ok test_moisture_injection")


def test_reduce_stats_consistency():
    """max/p95/p90 share a population; only the reduction differs.

    p95/p90 are TAIL MEANS — the average over the top 5 %/10 % of the box, not
    the percentile value. Guards the properties the swap relies on: the tail is
    the right size, the reported value is the tail's mean (so ~63 cells vote, not
    one), the statistics are ordered p90 ≤ p95 ≤ max, and a single-cell selection
    still reports that cell exactly (which is what keeps stat="max" bit-exact
    with the pre-percentile implementation).
    """
    from llat_manifold.diagnostics import response as R

    rng = np.random.default_rng(20260723)
    dpv = rng.normal(size=(13, 81, 81))
    p_hpa = np.asarray(layout.pressure_levels(), dtype=float)
    core = R._core_mask(81, 81, 5.0)
    lev = (p_hpa >= 700) & (p_hpa <= 1000)
    box = core[None, :, :] & lev[:, None, None]
    n_box = int(box.sum())

    v_max, kji, sel = R._reduce(dpv, lev, core, "max", "max")
    assert sel.sum() == 1 and dpv[kji] == v_max
    assert v_max == np.nanmax(np.where(box, dpv, np.nan))
    # the extremum is the 100th percentile of the same population
    assert abs(v_max - np.nanpercentile(np.where(box, dpv, np.nan), 100)) < 1e-12

    seen = {}
    for stat, pct in (("p95", 95), ("p90", 90)):
        v, _kji, sel = R._reduce(dpv, lev, core, "max", stat)
        frac = sel.sum() / n_box
        assert abs(frac - (100 - pct) / 100) < 0.01, (stat, frac)
        # THE contract: the value is the mean of the tail, not the cut that made it
        assert abs(v - dpv[sel].mean()) < 1e-12, f"{stat} must report the tail mean"
        cut = np.nanpercentile(np.where(box, dpv, np.nan), pct)
        assert v > cut, "a tail mean must exceed the percentile that defines it"
        assert v < v_max, "...and still sit below the extremum"
        assert sel.sum() > 50, "the whole point is that many cells vote, not one"
        # min pole takes the mirror tail
        v_min, _k, sel_min = R._reduce(dpv, lev, core, "min", stat)
        assert v_min < 0 < v and abs(sel_min.sum() - sel.sum()) <= 1
        seen[stat] = v
    assert seen["p90"] < seen["p95"] < v_max, "ordering p90 <= p95 <= max"

    # a single-cell selection reports that cell exactly, not a weighted average
    r2d = R._radius_field(np.tile(np.linspace(30, 10, 81)[:, None], (1, 81)),
                          np.tile(np.linspace(120, 140, 81)[None, :], (81, 1)))
    _v, kji, sel = R._reduce(dpv, lev, core, "max", "max")
    p_c, r_c = R._locate(sel, dpv, p_hpa, r2d)
    assert p_c == p_hpa[kji[0]] and r_c == r2d[kji[1], kji[2]]
    print("ok test_reduce_stats_consistency")


if __name__ == "__main__":
    test_layout_indices()
    test_active_lock_default_and_masked()
    test_heating_ic_matches_onestep()
    test_continuous_locking_and_forcing()
    test_sst_dynamic_mask()
    test_snapshot_math()
    test_snapshot_null_run_is_exactly_zero()
    test_snapshot_statics_pinned_to_initial_state()
    test_snapshot_recursion_uses_u0_as_base()
    test_upper_lock_q()
    test_moisture_injection()
    test_reduce_stats_consistency()
    print("\nALL OFFLINE TESTS PASSED")
