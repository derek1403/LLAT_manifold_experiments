"""Thin access layer over the shipped diagnostics — no physics is re-implemented here.

Every scalar these figures plot is computed by ``llat_manifold.diagnostics.response``
(``analyze_pair``, ``pv_response_series``, ``energy_series``, ``_sweep_points``).
This module exists only to (a) put the repo's ``src/`` on the path when a figure
script is run directly, and (b) give the private, underscore-prefixed helpers a
stable name so the figure scripts don't reach into them.

Two axes select what a figure is made of, and every script takes both:

``--ic``    which initial condition the runs started from —
            ``ragasa`` = the real analysis IC, ``axisym`` = the azimuthally
            averaged idealized vortex built from it. These are separate output
            *categories* holding identically named families and run tags, so
            nothing downstream needs to know which one it is looking at.
``--stat``  how each ΔPV field collapses to one number — ``max`` (single-cell
            extremum), or ``p95`` / ``p90``, the **mean over the top 5 % / 10 %**
            of the same core×layer box. The percentile is only the cut that
            defines the tail; the value reported is the average of every cell
            beyond it, which is what stops one grid point from deciding a curve.
            See :func:`llat_manifold.diagnostics.response._reduce`.

Figures land in ``figs/<ic>/<stat>/``. The loose PNGs at ``figs/`` are the first
round of results (extremum × real IC) and are left exactly where they are.

Run-family map (what was injected, what was constrained). Referenced by the figure
titles so a reader always knows which intervention produced a curve:

    heating_moist   sweep_*      inject ΔT (Deep profile), q free      — baseline
    heating_qlock   sweepq_*     inject ΔT, δq = 0                     — H0
    dq_measured     sweepdq_*    inject δq only, measured scaling      — H2
    dq_latent       sweepdql_*   inject δq only, latent-equivalent     — H3
    dq_tlock        sweepdqtl_*  inject δq only, δT = 0                — H5
    heating_wlock   sweepw_*     inject ΔT, δw = 0                     — H1
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
    DEFAULT_STAT,
    STATS,
    _pert_params,
    _sweep_points,
    _stat_key,
    analyze_pair,
    energy_series,
    pv_response_series,
    stat_label,
)

# init condition -> output category. Same families, same tags, different IC.
IC = {"ragasa": "diabatic_heating", "axisym": "diabatic_heating_axisym"}
IC_LABEL = {"ragasa": "real RAGASA analysis IC",
            "axisym": "axisymmetric idealized vortex IC"}
DEFAULT_IC = "axisym"

FIGS_ROOT = REPO / "experiments" / "diabatic_heating" / "figs"

# family -> (tag glob, what it does). Single source for figure labelling.
FAMILY = {
    "heating_moist": ("sweep_*", "inject ΔT (heating) — q free"),
    "heating_qlock": ("sweepq_*", "inject ΔT — δq = 0"),
    "dq_measured": ("sweepdq_*", "inject δq only — measured scaling"),
    "dq_latent": ("sweepdql_*", "inject δq only — latent-equivalent (equal energy)"),
    "dq_tlock": ("sweepdqtl_*", "inject δq only — δT = 0"),
    "dq_uvlock": ("sweepdquv_*", "inject δq only — δu = δv = 0"),
    "heating_wlock": ("sweepw_*", "inject ΔT — δw = 0"),
    "heating_zlock": ("sweepz_*", "inject ΔT — δz = 0"),
    "dq_bl": ("sweepdqbl_*", "inject δq only — boundary layer (850–1000 hPa)"),
    "dq_ft": ("sweepdqft_*", "inject δq only — free troposphere (400–700 hPa)"),
    "dq_offcore": ("sweepdqoff_*", "inject δq only — quiescent side (off-vortex)"),
}

WEAK, STRONG = "2025091700", "2025092000"
# Intensity-ladder members: the 0921/0922 vortices dropped into 0920's fixed
# environment (axisym_<tc>_<init>_env2025092000.npz). Only the vortex differs from
# STRONG; SST / f / radiation / lat-lon are 0920's, held constant across the ladder.
SEV, VSEV = "2025092100", "2025092200"
LADDER_ENV = "2025092000"                # reference environment for SEV/VSEV
CASE = {WEAK: "weak (0917)", STRONG: "strong (0920)",
        SEV: "severe (0921)", VSEV: "very severe (0922)"}


def category(ic: str = DEFAULT_IC) -> Path:
    """Output category directory holding one IC's run families."""
    if ic not in IC:
        raise KeyError(f"unknown ic {ic!r} (expected one of {sorted(IC)})")
    return REPO / "outputs" / IC[ic]


def axisym_ic_path(init: str, tc_id: str = "202518W") -> Path:
    """The axisymmetric IC npz for an init. Ladder members (SEV/VSEV) carry the
    ``_env<LADDER_ENV>`` suffix — their vortex is their own day's but the
    environment is 0920's; WEAK/STRONG keep their own environment (no suffix)."""
    stem = f"axisym_{tc_id}_{init}"
    if init in (SEV, VSEV):
        stem += f"_env{LADDER_ENV}"
    return REPO / "outputs" / "axisymmetric_ic" / f"{stem}.npz"


def figs_dir(ic: str = DEFAULT_IC, stat: str = DEFAULT_STAT) -> Path:
    """Where a figure for this (IC, statistic) belongs; created on demand."""
    if stat not in STATS:
        raise KeyError(f"unknown stat {stat!r} (expected one of {STATS})")
    d = FIGS_ROOT / ic / stat
    d.mkdir(parents=True, exist_ok=True)
    return d


def poles(stat: str = DEFAULT_STAT):
    """The two dipole scalars every figure keys off, with their display names."""
    return (("lowlevel", stat_label(stat, "lowlevel")),
            ("upperlevel", stat_label(stat, "upperlevel")))


def sweep(pattern, lead_hr: int = 24, ic: str = DEFAULT_IC,
          stat: str = DEFAULT_STAT) -> dict:
    """{init_time: [(amp_K, metrics), ...]} at one lead, over a run-tag glob.

    Each ``metrics`` dict carries every statistic, so a figure that compares
    ``max``/``p95``/``p90`` reads the sweep once.
    """
    return _sweep_points(category(ic), lead_hr, pattern, stat=stat)


def forcing_end_hours(run_dir) -> int:
    """Hour at which the injection stops (8 steps × 3 h = 24 h)."""
    return int(_pert_params(run_dir)["forcing_steps"]) * 3


def run(family: str, tag: str, ic: str = DEFAULT_IC) -> Path:
    """Locate one run dir, e.g. run('dq_tlock', 'tseriesdqtl_5K_120h_init2025092000')."""
    p = category(ic) / family / tag
    if not p.exists():
        raise FileNotFoundError(f"no run {p}")
    return p


def add_data_args(ap):
    """The two data-selection flags every figure script shares."""
    ap.add_argument("--ic", choices=sorted(IC), default=DEFAULT_IC,
                    help="initial condition the runs used (default: %(default)s)")
    ap.add_argument("--stat", choices=STATS, default=DEFAULT_STAT,
                    help="ΔPV reduction: max = single-cell extremum; p95/p90 = "
                         "the MEAN over the top 5%%/10%% of the same box "
                         "(default: %(default)s)")
    return ap


def ic_note(ic: str, stat: str) -> str:
    """One-line provenance stamp for a figure caption."""
    return f"{IC_LABEL[ic]} · ΔPV reduced by {stat}"


__all__ = [
    "IC", "IC_LABEL", "DEFAULT_IC", "DEFAULT_STAT", "STATS", "FIGS_ROOT", "FAMILY",
    "WEAK", "STRONG", "CASE",
    "analyze_pair", "energy_series", "pv_response_series", "load_stamp",
    "category", "figs_dir", "poles", "sweep", "forcing_end_hours", "run",
    "add_data_args", "ic_note", "stat_label", "_stat_key",
]
