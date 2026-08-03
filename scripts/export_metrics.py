#!/usr/bin/env python
"""Export every ΔPV response scalar to one tidy CSV — the write-up's source of truth.

    python scripts/export_metrics.py --ic axisym ragasa --out outputs/metrics.csv

One row per (ic, family, init, amp_K, lead_hr, stat, pole):

    ic        ragasa | axisym                 which initial condition
    family    heating_moist, dq_measured, …   which intervention
    init      2025091700 | 2025092000         weak | strong stage
    amp_K     0.5 … 10.0                      nominal injection amplitude
    lead_hr   24 (sweeps) or 3…120 (series)   iteration n x 3 h
    stat      max | p95 | p90                 how ΔPV was reduced
    pole      lowlevel | upperlevel           generation below | destruction above
    value     [PVU]                           the scalar itself
    p_hPa     level of the reduction          extremum cell, or exceedance centroid
    r_km      radius of the reduction

Written so a claim in the report can be traced to a row instead of to a figure read
by eye, and so the extremum-vs-percentile comparison is a groupby rather than a
re-run. All three statistics come out of one pass over the bundles.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llat_manifold.diagnostics import load_stamp                        # noqa: E402
from llat_manifold.diagnostics.response import (                        # noqa: E402
    _POLE_SPEC, STATS, _continuous_pairs, _find_runs, _pert_params,
    _stat_key, analyze_pair,
)

_CATEGORY = {"ragasa": "diabatic_heating", "axisym": "diabatic_heating_axisym"}
FIELDS = ["ic", "family", "run", "init", "amp_K", "lead_hr", "stat", "pole",
          "value", "p_hPa", "r_km"]


def _rows(ic: str, run: Path, family: str, sweeps_only: bool):
    """All (stat x pole) rows for one run — every lead for a series, 24 h for a sweep."""
    cfg = load_stamp(run).get("resolved_config", {})
    init = str(cfg.get("init_time", "?"))
    pert = _pert_params(run)
    amp, sigma = float(pert["amp_K"]), float(pert["sigma"])
    pairs = _continuous_pairs(run)
    if sweeps_only:
        pairs = [p for p in pairs if p[0] == 24] or pairs[-1:]
    for lead, d, c in pairs:
        m, _aux = analyze_pair(d, c, sigma=sigma)
        for pole, mode, _box in _POLE_SPEC:
            for s in STATS:
                k = _stat_key(pole, mode, s)
                yield {"ic": ic, "family": family, "run": run.name, "init": init,
                       "amp_K": amp, "lead_hr": lead, "stat": s, "pole": pole,
                       "value": round(m[k], 6), "p_hPa": round(m[f"{k}_p"], 2),
                       "r_km": round(m[f"{k}_r"], 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ic", nargs="+", choices=sorted(_CATEGORY),
                    default=["axisym", "ragasa"])
    ap.add_argument("--out", default="outputs/response_metrics.csv")
    ap.add_argument("--all-leads", action="store_true",
                    help="also export every lead of the 5-day series runs "
                         "(default: series contribute their full time axis, sweeps "
                         "only hour 24)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for ic in a.ic:
            root = _HERE.parent / "outputs" / _CATEGORY[ic]
            if not root.exists():
                print(f"[metrics] {ic}: {root} missing, skipped")
                continue
            for fam_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                if fam_dir.name == "figures":
                    continue
                for run in sorted(_find_runs(root, "*", fam_dir.name)):
                    if not run.is_dir():
                        continue
                    try:
                        # A 24 h sweep run has one interesting lead; a 120 h series
                        # is the time axis itself, so keep all of its leads.
                        series = "120h" in run.name
                        for row in _rows(ic, run, fam_dir.name,
                                         sweeps_only=not (series or a.all_leads)):
                            w.writerow(row)
                            n += 1
                    except (FileNotFoundError, ValueError) as e:
                        print(f"[metrics] skip {fam_dir.name}/{run.name}: {e}")
                print(f"[metrics] {ic}/{fam_dir.name}: {n} rows so far")
    print(f"[metrics] wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
