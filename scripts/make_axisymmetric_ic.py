#!/usr/bin/env python
"""Build the axisymmetric idealized-vortex ICs from the real RAGASA analysis.

    python scripts/make_axisymmetric_ic.py --tc-id 202518W \
        --inits 2025091700 2025092000

Writes ``outputs/axisymmetric_ic/axisym_<tc_id>_<init>.npz`` per init time, in the
same format :mod:`llat_manifold.idealized_vortex.background` reads, so any config
can pick one up with ``ic_npz:`` and the driver takes it from there.

Prints the verification numbers for each IC: the residual asymmetry (should drop to
~0 for the averaged channels), the RMW and peak tangential wind before and after
(the average must not smear the vortex away), the central MSL, and a check that
``sfc[9:]`` came through bit-identical.

No model is loaded — this is pure numpy over the combined NetCDF IC. Sister script
to ``make_quiescent_background.py``, which flattens an analysis instead of
averaging it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llat_manifold import config, driver, operators                      # noqa: E402
from llat_manifold.idealized_vortex import axisymmetric as ax            # noqa: E402
from llat_manifold.idealized_vortex import background as bg              # noqa: E402


OMEGA = 7.292115e-5           # Earth rotation rate, for f = 2Ω sin(lat)


def default_path(tc_id: str, init_str: str, env_init: str | None = None) -> Path:
    """Output path. A transplanted IC (vortex ``init`` in ``env_init``'s
    environment) gets an ``_env<init>`` suffix so it never clobbers the plain
    per-day IC (whose environment is its own)."""
    stem = f"axisym_{tc_id}_{init_str}"
    if env_init and env_init != init_str:
        stem += f"_env{env_init}"
    return config.output_root() / "axisymmetric_ic" / f"{stem}.npz"


def _center_lat_f(state):
    """Storm-centre latitude [°] and Coriolis f [s⁻¹] of an IC's own grid."""
    sfc = state.surface
    cy, cx = sfc.shape[0] // 2, sfc.shape[1] // 2
    lat = float(sfc[cy, cx, -1])
    return lat, 2.0 * OMEGA * np.sin(np.deg2rad(lat))


def build(tc_id: str, init_str: str, dlampty, out_path=None, env=None,
          env_init=None) -> dict:
    """Load one init's analysis, axisymmetrize it, optionally drop it into a
    reference environment, save, and report. Returns the numbers ``main`` needs
    for the cross-member checks."""
    state = driver.load_initial_state(tc_id, init_str, dlampty)
    ideal = ax.axisymmetrize(state)
    native_lat, native_f = _center_lat_f(state)
    if env is not None:
        ideal = ax.transplant_environment(ideal, env)
    out = str(out_path or default_path(tc_id, init_str, env_init))
    saved = bg.save_background(ideal, out)

    s = ax.summary(state, ideal)          # symmetry/RMW/Vt measured on the vortex
    print(f"\n[axisym] {tc_id} {init_str}"
          f"{'  (env=' + env_init + ')' if env is not None else ''} -> {saved}")
    print(f"{'':10s} {'Vt_max':>9s} {'RMW':>8s} {'MSL(c)':>9s} "
          f"{'asym T850':>10s} {'asym q850':>10s} {'asym Vt850':>11s}")
    for tag in ("before", "after"):
        r = s[tag]
        print(f"  {tag:8s} {r['vt_max_ms']:8.2f}m {r['rmw_km']:7.0f}km "
              f"{r['msl_center_hPa']:8.1f}h {r['asym_t850']:10.4f} "
              f"{r['asym_q850']:10.4f} {r['asym_vt850']:11.4f}")
    if max(s["after"]["asym_t850"], s["after"]["asym_q850"]) > 1e-6:
        print("  [warn] residual asymmetry above 1e-6 — check the radial binning")

    # environment channels: with a transplant, sfc[9:] is the reference's, not the
    # storm's own, so the "unchanged vs its own analysis" check no longer applies.
    if env is None:
        print(f"  sfc[9:] bit-identical to own analysis: {s['sfc_keep_identical']}")
        if not s["sfc_keep_identical"]:
            raise SystemExit("[axisym] ABORT: the keep-as-is surface channels changed")
    else:
        env_lat, env_f = _center_lat_f(env)
        print(f"  transplant: vortex sat at {native_lat:.2f}°N (f={native_f:.3e}) "
              f"-> reference {env_lat:.2f}°N (f={env_f:.3e}), "
              f"Δf/f = {(env_f - native_f) / native_f * 100:+.1f}%")

    return {"init": init_str, "saved": saved, "vt_after": s["after"]["vt_max_ms"],
            "rmw": s["after"]["rmw_km"], "native_lat": native_lat,
            "state": ideal}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tc-id", default="202518W")
    ap.add_argument("--inits", nargs="+", default=["2025091700", "2025092000"])
    ap.add_argument("--env-from", default=None,
                    help="init time whose environment (SST/f/radiation/lat-lon/clock) "
                         "every built IC is placed into, so the environment is a "
                         "controlled constant across the vortices and only the vortex "
                         "differs. Omit to keep each date on its own real environment.")
    ap.add_argument("--out-dir", default=None,
                    help="override the default outputs/axisymmetric_ic/ location")
    args = ap.parse_args(argv)

    # load_initial_state only uses the model object to reshape the NetCDF, but it
    # does need it, so pay the load once for the whole batch.
    models = operators.load_models()

    env = None
    if args.env_from:
        env = driver.load_initial_state(args.tc_id, str(args.env_from), models)
        print(f"[axisym] reference environment = {args.tc_id} {args.env_from} "
              f"(all built ICs share its SST/f/radiation/lat-lon/clock)")

    results = []
    for init in args.inits:
        out = (Path(args.out_dir) / default_path(args.tc_id, init, args.env_from).name
               if args.out_dir else None)
        results.append(build(args.tc_id, str(init), models, out,
                             env=env, env_init=args.env_from))

    # Cross-member checks: the whole point of --env-from is that the environment is
    # identical across members, so verify it byte-for-byte, and print the intensity
    # ladder the transplant is meant to isolate.
    if len(results) > 1:
        print("\n[axisym] === cross-member summary ===")
        print(f"  intensity ladder (Vt_max after axisym):")
        for r in sorted(results, key=lambda r: r["vt_after"]):
            print(f"    {r['init']}: {r['vt_after']:6.2f} m/s  (RMW {r['rmw']:.0f} km, "
                  f"vortex from {r['native_lat']:.2f}°N)")
        if env is not None:
            ref = results[0]["state"].surface[:, :, ax.SFC_KEEP]
            ref_t = results[0]["state"].initial_time
            all_env = all(
                np.array_equal(r["state"].surface[:, :, ax.SFC_KEEP], ref) and
                r["state"].initial_time == ref_t for r in results[1:])
            print(f"  environment (sfc[9:] + initial_time) byte-identical across all "
                  f"{len(results)} members: {all_env}")
            if not all_env:
                raise SystemExit("[axisym] ABORT: transplanted environments differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
