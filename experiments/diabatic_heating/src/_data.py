"""Thin access layer over the shipped diagnostics — no physics is re-implemented here.

Every scalar these figures plot is computed by ``llat_manifold.diagnostics.response``
(``analyze_pair``, ``pv_response_series``, ``energy_series``, ``_sweep_points``).
This module exists only to (a) put the repo's ``src/`` on the path when a figure
script is run directly, and (b) give the private, underscore-prefixed helpers a
stable name so the figure scripts don't reach into them.

Run-family map (what was injected, what was locked). Referenced by the figure
titles so a reader always knows which intervention produced a curve:

    heating_moist   sweep_*      inject ΔT (Deep profile), q free      — baseline
    heating_qlock   sweepq_*     inject ΔT, δq pinned to control       — H0
    dq_measured     sweepdq_*    inject δq only, measured scaling      — H2
    dq_latent       sweepdql_*   inject δq only, latent-equivalent     — H3
    dq_tlock        sweepdqtl_*  inject δq only (measured), δT pinned  — H5
    heating_wlock   sweepw_*     inject ΔT, δw pinned to control       — H1
"""
from __future__ import annotations

import sys
from pathlib import Path

# experiments/diabatic_heating/src/_data.py -> repo root is three levels up.
REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from llat_manifold.diagnostics import load_stamp                      # noqa: E402
from llat_manifold.diagnostics.response import (                      # noqa: E402
    _pert_params,
    _sweep_points,
    analyze_pair,
    energy_series,
    pv_response_series,
)

CATEGORY = REPO / "outputs" / "diabatic_heating"
FIGS = REPO / "experiments" / "diabatic_heating" / "figs"

# family -> (tag glob, what it does). Single source for figure labelling.
FAMILY = {
    "heating_moist": ("sweep_*", "inject ΔT (heating) — q free"),
    "heating_qlock": ("sweepq_*", "inject ΔT — δq locked to 0"),
    "dq_measured": ("sweepdq_*", "inject δq only — measured scaling"),
    "dq_latent": ("sweepdql_*", "inject δq only — latent-equivalent (equal energy)"),
    "dq_tlock": ("sweepdqtl_*", "inject δq only — δT locked to 0"),
    "dq_uvlock": ("sweepdquv_*", "inject δq only — δu,δv locked to 0"),
    "heating_wlock": ("sweepw_*", "inject ΔT — δw locked to 0"),
    "heating_zlock": ("sweepz_*", "inject ΔT — δz locked to 0"),
    "dq_bl": ("sweepdqbl_*", "inject δq only — boundary layer (850–1000 hPa)"),
    "dq_ft": ("sweepdqft_*", "inject δq only — free troposphere (400–700 hPa)"),
    "dq_offcore": ("sweepdqoff_*", "inject δq only — quiescent side (off-vortex)"),
}

# The dipole scalars every figure keys off, with their display names.
POLES = (
    ("lowlevel_max", "low-level max ΔPV (700–1000 hPa)"),
    ("upperlevel_min", "upper-level min ΔPV (200–500 hPa)"),
)

WEAK, STRONG = "2025091700", "2025092000"
CASE = {WEAK: "weak (0917)", STRONG: "strong (0920)"}


def sweep(pattern, lead_hr: int = 24, category=CATEGORY) -> dict:
    """{init_time: [(amp_K, metrics), ...]} at one lead, over a run-tag glob."""
    return _sweep_points(category, lead_hr, pattern)


def forcing_end_hours(run_dir) -> int:
    """Nominal hour at which the injection stops (8 steps × 3 h = 24 h)."""
    return int(_pert_params(run_dir)["forcing_steps"]) * 3


def run(family: str, tag: str) -> Path:
    """Locate one run dir, e.g. run('dq_tlock', 'tseriesdqtl_5K_120h_init2025092000')."""
    p = CATEGORY / family / tag
    if not p.exists():
        raise FileNotFoundError(f"no run {p}")
    return p


__all__ = [
    "CATEGORY", "FIGS", "FAMILY", "POLES", "WEAK", "STRONG", "CASE",
    "analyze_pair", "energy_series", "pv_response_series", "load_stamp",
    "sweep", "forcing_end_hours", "run",
]
