#!/usr/bin/env python
"""Build a family of balanced idealized vortices that differ only in intensity.

    python scripts/make_vortex_intensity_ic.py --vmax 0 10 20 30 40 50 --rmw-deg 1.0

Writes ``outputs/vortex_intensity_ic/vortex_v<V>_rmw<R>.npz`` per vortex, in the
:mod:`llat_manifold.idealized_vortex.background` ``InitialState`` format, so any
config picks one up with ``ic_npz:`` and the shared continuous driver takes it from
there — exactly the path ``make_axisymmetric_ic.py`` already uses.

Why this exists
---------------
The heating experiments compare RAGASA at two stages (weak 0917 / strong 0920) and
read the difference as an intensity effect. Those two states differ in far more than
intensity: SST, vertical shear, environmental humidity, translation speed, and for
0920 a domain containing ~6 % land and 1.9 km of terrain. Any claim about the local
Rossby radius L_R = N·H/√(ξη) — that a strong vortex's high inertial stability traps
an injected thermal anomaly near the axis while a weak vortex lets it spread — is
confounded by all of those.

Here intensity is the *only* thing that changes:

  * environment: the quiescent background from ``make_quiescent_background.py``
    (horizontally uniform ocean, no terrain, no land, no ambient flow, real f so β
    survives) — identical for every member;
  * vortex: :class:`~llat_manifold.idealized_vortex.ring.RingVortexPerturbation` in
    gradient-wind + hydrostatic balance, with ``vmax`` swept and everything else fixed.

Vortex shape
------------
``gamma`` (ζ_eye/ζ_ring) is set near 1 so ζ is nearly uniform inside r₂ — a Rankine-like
**monotonic** V(r) rather than the hollow ring the barotropic-instability experiments
use. A hollow ring would satisfy the Rayleigh criterion and break down on its own,
contaminating an intensity test with an instability that has nothing to do with L_R.
For the same reason ``noise_amp`` is 0: no asymmetry seed.

``vmax = 0`` is a legitimate and important member — it is the no-vortex control, whose
L_R is the environmental N·H/f, i.e. the large-L_R end of the sweep.

Each vortex is reported with the numbers that make it usable as a data point: peak
850 hPa V_t, RMW, central MSL drop, and the L_R profile from
:mod:`llat_manifold.diagnostics.waves`. No model is loaded — pure numpy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llat_manifold import config, layout                                # noqa: E402
from llat_manifold.diagnostics import waves                             # noqa: E402
from llat_manifold.idealized_vortex import axisymmetric as ax           # noqa: E402
from llat_manifold.idealized_vortex import background as bg             # noqa: E402
from llat_manifold.idealized_vortex.ring import RingVortexPerturbation  # noqa: E402
from llat_manifold.perturbations.base import StepContext                # noqa: E402

# Near-1 hollowness = uniform ζ inside the ring = monotonic (Rankine-like) V(r).
# The class rejects gamma == 1.0 exactly, so sit just below it.
MONOTONIC_GAMMA = 0.98


def _tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def default_path(vmax: float, rmw_deg: float) -> Path:
    return (config.output_root() / "vortex_intensity_ic"
            / f"vortex_v{_tag(vmax)}_rmw{_tag(rmw_deg)}.npz")


def build_one(base: "object", vmax: float, rmw_deg: float, *, delta: float,
              out_path=None, dry_run: bool = False) -> dict:
    """Add one balanced vortex to a copy of the background; save and report."""
    upper = base.upper.copy()
    sfc = base.surface.copy()

    if vmax > 0.0:
        ctx = StepContext(fore_i=0, lead_hr=0, initial_time=base.initial_time,
                          dlam_lats=base.dlam_lats, dlam_lons=base.dlam_lons)
        pert = RingVortexPerturbation(gamma=MONOTONIC_GAMMA, delta=delta,
                                      rmw_deg=rmw_deg, vmax=vmax,
                                      noise_amp=0.0, seed=0)
        upper, sfc = pert.apply_ic(upper, sfc, ctx)

    from llat_manifold import driver
    state = driver.InitialState(upper, sfc, base.dlam_lats, base.dlam_lons,
                                base.initial_time)

    # upper is (nz, ny, nx, nvar); the axisymmetric helpers want that layout.
    r_km, vt = ax.tangential_profile(upper, sfc, 850)
    i_msl = layout.surface_index("msl")
    cy, cx = sfc.shape[0] // 2, sfc.shape[1] // 2
    msl_drop = float(sfc[cy, cx, i_msl] - base.surface[cy, cx, i_msl]) / 100.0
    summary = waves.rossby_radius_summary(upper, sfc)

    out = str(out_path or default_path(vmax, rmw_deg))
    saved = out if dry_run else bg.save_background(state, out)
    return {
        "vmax_set": vmax, "rmw_deg": rmw_deg, "path": saved,
        "vt_max_ms": float(np.nanmax(vt)),
        "rmw_km": float(r_km[int(np.nanargmax(vt))]),
        "msl_drop_hPa": msl_drop,
        "L_R_core_km": summary["L_R_core_km"],
        "L_R_min_km": summary["L_R_min_km"],
        "I_core_s1": summary["I_core_s1"],
        "N_band_s1": summary["N_band_s1"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", default=None,
                    help="quiescent background npz (default: the 202518W 2025091700 one "
                         "built by scripts/make_quiescent_background.py)")
    ap.add_argument("--vmax", nargs="+", type=float,
                    default=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                    help="peak tangential winds [m/s]; 0 = the no-vortex control")
    ap.add_argument("--rmw-deg", nargs="+", type=float, default=[1.0],
                    help="outer ring radius r2 [deg]; every (vmax, rmw) pair is built")
    ap.add_argument("--delta", type=float, default=0.7, help="ring thickness r1/r2")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the numbers without writing any npz")
    a = ap.parse_args(argv)

    src = a.background or bg.default_background_path("202518W", "2025091700")
    if not Path(config._resolve(str(src))).exists():
        raise SystemExit(
            f"[vortex-ic] background {src} not found — build it first:\n"
            f"  python scripts/make_quiescent_background.py --tc-id 202518W "
            f"--init 2025091700")
    base = bg.load_background(src)
    print(f"[vortex-ic] background: {src}")
    print(f"[vortex-ic] {'v_set':>6s} {'rmw°':>5s} {'Vt_max':>8s} {'RMW':>7s} "
          f"{'ΔMSL':>8s} {'I_core':>10s} {'L_R_core':>9s} {'L_R_min':>8s}")

    rows = []
    for rmw in a.rmw_deg:
        for v in a.vmax:
            out = (Path(a.out_dir) / f"vortex_v{_tag(v)}_rmw{_tag(rmw)}.npz"
                   if a.out_dir else None)
            r = build_one(base, v, rmw, delta=a.delta, out_path=out,
                          dry_run=a.dry_run)
            rows.append(r)
            print(f"[vortex-ic] {r['vmax_set']:6.0f} {rmw:5.1f} "
                  f"{r['vt_max_ms']:7.2f}m {r['rmw_km']:6.0f}km "
                  f"{r['msl_drop_hPa']:7.1f}h {r['I_core_s1']:10.3e} "
                  f"{r['L_R_core_km']:8.0f}k {r['L_R_min_km']:7.0f}k")

    for r in rows:
        if r["vmax_set"] > 0 and abs(r["vt_max_ms"] - r["vmax_set"]) / r["vmax_set"] > 0.05:
            print(f"  [warn] {Path(r['path']).name}: realised Vt_max "
                  f"{r['vt_max_ms']:.2f} is >5% off the requested {r['vmax_set']:.0f} "
                  f"(azimuthal binning on a 0.25° grid; check rmw_deg is resolvable)")
    if not a.dry_run:
        print(f"[vortex-ic] wrote {len(rows)} IC(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
