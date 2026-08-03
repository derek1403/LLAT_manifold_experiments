#!/usr/bin/env python
"""Measure the δq scaling the moist heating runs grow, for the δq-only reverse probe.

    python scripts/measure_dq_scaling.py --ic axisym          # -> dq_measured_axisym.yaml
    python scripts/measure_dq_scaling.py --ic ragasa --check  # reproduce the committed table

The "measured" δq-only family answers "what if the model got *only* the moisture the
heating would have produced, and no heating at all?" — so its injection has to be
read off the moist runs themselves rather than chosen. For each init time this takes
the reference moist run (``sweep_5K_24h_init<init>``) at hour 24 and, inside the
heated core (r ≤ 2σ), records per pressure level the **signed extremum** of δq. The
profile is normalised so the peak-|δq| level is 1, and

    gkg_per_K = peak |δq| [g/kg] / 5 K

carries the amplitude, which the sweep then scales linearly with ``amp_K``.

Because the measurement is taken off the moist runs, **each IC needs its own table**:
the axisymmetric vortex grows a different amount of moisture from the same heating,
so reusing the real-IC numbers would inject something that run never produced. Run
this after the axisymmetric ``heating_moist`` sweep and before any ``dq_*`` family.

The procedure was previously documented only in prose in
``experiments/diabatic_heating/README.md``; ``--check`` verifies this implementation
reproduces the committed ``dq_measured.yaml`` from the real-IC runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for p in (str(_SRC), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llat_manifold import io, layout                                    # noqa: E402
from llat_manifold.diagnostics.response import _core_mask               # noqa: E402

_CFG_DIR = _HERE.parent / "experiments" / "diabatic_heating" / "configs"
_CATEGORY = {"ragasa": "diabatic_heating", "axisym": "diabatic_heating_axisym"}
_OUT = {"ragasa": _CFG_DIR / "dq_measured.yaml",
        "axisym": _CFG_DIR / "dq_measured_axisym.yaml"}

REF_AMP_K = 5.0                     # the reference moist run the profile is read off
REF_LEAD_HR = 24


def measure(run_dir: Path, *, sigma: float = 5.0, lead_hr: int = REF_LEAD_HR,
            amp_K: float = REF_AMP_K) -> dict:
    """{gkg_per_K, profile} read off one moist run's δq at ``lead_hr``."""
    hits = sorted((run_dir / "data").glob(f"delta_continuous_*lead{lead_hr:03d}hr.npz"))
    if not hits:
        raise FileNotFoundError(f"no lead-{lead_hr:03d}h delta bundle under {run_dir}")
    up, _sfc = io.load_delta_bundle(hits[0])
    dq = up[..., layout.upper_index("q")] * 1e3          # kg/kg -> g/kg
    core = _core_mask(dq.shape[1], dq.shape[2], sigma)

    # Signed extremum per level inside the heated core: the largest departure,
    # keeping its sign (drying levels are part of the structure).
    prof = np.empty(dq.shape[0])
    for k in range(dq.shape[0]):
        vals = dq[k][core]
        prof[k] = vals[int(np.argmax(np.abs(vals)))]
    peak = float(np.max(np.abs(prof)))
    return {"gkg_per_K": round(peak / amp_K, 4),
            "profile": [round(float(v), 4) for v in prof / peak]}


def build_table(ic: str, inits, *, tag: str | None = None) -> dict:
    root = _HERE.parent / "outputs" / _CATEGORY[ic] / "heating_moist"
    table = {}
    for init in inits:
        name = tag.format(init=init) if tag else f"sweep_5K_24h_init{init}"
        run = root / name
        if not run.exists():
            raise FileNotFoundError(
                f"{run} not found — the {ic} heating_moist sweep must run first")
        table[str(init)] = measure(run)
        e = table[str(init)]
        print(f"[dq] {ic} {init}: gkg_per_K={e['gkg_per_K']}  "
              f"peak level={layout.pressure_levels()[int(np.argmax(np.abs(e['profile'])))]} hPa")
    return table


_HEADER = """\
# Measured δq scaling for MoisturePerturbation (scaling: measured).
# Source: {ic} moist heating runs sweep_5K_24h_init<init>, δq at hour 24,
# signed extremum per level within the heated core (r ≤ 2σ), normalized so the
# peak-|value| level = 1. gkg_per_K = peak |δq| [g/kg] / 5 K.
# Regenerate with: python scripts/measure_dq_scaling.py --ic {ic}
# pressure_levels_hPa: {levels}
"""


def write_yaml(table: dict, path: Path, ic: str) -> None:
    lines = [_HEADER.format(ic=ic, levels=layout.pressure_levels())]
    for init, e in table.items():
        prof = ", ".join(f"{v:.4f}" for v in e["profile"])
        lines.append(f'"{init}":\n  gkg_per_K: {e["gkg_per_K"]}\n  profile: [{prof}]\n')
    path.write_text("".join(lines))
    print(f"[dq] wrote {path}")


def check_against_committed(table: dict) -> int:
    """Compare a freshly measured real-IC table with the committed one."""
    import yaml
    ref = yaml.safe_load(_OUT["ragasa"].read_text())
    bad = 0
    for init, e in table.items():
        r = ref[init]
        if abs(e["gkg_per_K"] - r["gkg_per_K"]) > 1e-4:
            print(f"  MISMATCH {init} gkg_per_K: {e['gkg_per_K']} vs {r['gkg_per_K']}")
            bad += 1
        d = np.max(np.abs(np.array(e["profile"]) - np.array(r["profile"])))
        if d > 1e-4:
            print(f"  MISMATCH {init} profile: max |Δ| = {d:.5f}")
            bad += 1
        else:
            print(f"  {init}: profile matches (max |Δ| = {d:.2e})")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ic", choices=sorted(_CATEGORY), default="axisym")
    ap.add_argument("--inits", nargs="+", default=["2025091700", "2025092000"])
    ap.add_argument("--out", default=None, help="override the output YAML path")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed dq_measured.yaml instead of "
                         "writing (only meaningful with --ic ragasa)")
    a = ap.parse_args(argv)

    table = build_table(a.ic, a.inits)
    if a.check:
        bad = check_against_committed(table)
        print(f"[dq] check: {bad} mismatch(es)")
        return 1 if bad else 0
    write_yaml(table, Path(a.out) if a.out else _OUT[a.ic], a.ic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
