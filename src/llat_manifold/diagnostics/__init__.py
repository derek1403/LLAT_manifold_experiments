"""Advanced dynamical/thermodynamic diagnostics for idealized experiments.

These are the PV / divergence / wave tools deliberately left out of the operational
repo (Part 1). Each plotter is config-aware: it reads the ``config_used.yaml`` stamp
written next to the outputs and annotates figures with the experiment parameters.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_stamp(out_dir) -> dict:
    """Load the resolved experiment config stamped at the run root.

    Accepts either the run root or its ``data/`` subdir (checks the dir and its
    parent), so diagnostics work whether handed the run dir or the bundle dir.
    """
    out_dir = Path(out_dir)
    for cand in (out_dir / "config_used.yaml", out_dir.parent / "config_used.yaml"):
        if cand.exists():
            with open(cand) as f:
                return yaml.safe_load(f) or {}
    return {}


def experiment_title(out_dir, prefix: str = "") -> str:
    """A compact two-line figure title built from the config stamp.

    Line 1: the diagnostic prefix + category/tag. Line 2: mode + perturbation params.
    Two lines keep long titles from overrunning the figure width (and overlapping the
    heating-profile panel title).
    """
    stamp = load_stamp(out_dir)
    cfg = stamp.get("resolved_config", stamp)
    cat = cfg.get("category", "?")
    tag = cfg.get("tag", "?")
    mode = cfg.get("mode", "?")
    pert = cfg.get("perturbation", {}) or {}
    # Keep the title short: only the most informative perturbation params.
    keys = ("type", "amp_K", "heat_type", "delta_K", "vortex_name")
    parts = [f"{k}={pert[k]}" for k in keys if k in pert]
    init = cfg.get("init_time")
    if init:
        parts.insert(0, f"init={init}")
    ptxt = ", ".join(parts)
    line1 = f"{prefix} {cat}/{tag}".strip()
    line2 = f"[{mode}]" + (f"  {ptxt}" if ptxt else "")
    return f"{line1}\n{line2}"
