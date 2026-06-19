"""Central configuration access for the idealized-experiment repo.

Mirrors Part 1's ``regional_couple.config`` pattern: loads ``config/paths.yaml``
and ``config/model_layout.yaml`` and resolves repo-relative paths to absolute
ones. This is the single place that knows where things live.
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml


def repo_root() -> Path:
    """Absolute path to the repository root (parent of ``src/``)."""
    # config.py lives at <repo>/src/llat_manifold/config.py
    return Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> Path:
    """Resolve a config path: absolute stays absolute, relative joins repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root() / p)


@functools.lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict:
    with open(repo_root() / "config" / name) as f:
        return yaml.safe_load(f)


def paths() -> dict:
    """Raw parsed ``config/paths.yaml``."""
    return _load_yaml("paths.yaml")


def layout() -> dict:
    """Raw parsed ``config/model_layout.yaml``."""
    return _load_yaml("model_layout.yaml")


# --- convenience resolvers ---------------------------------------------------

def ic_root(key: str) -> Path:
    """Absolute path for an ``ic.<key>`` entry (e.g. 'fcnv2_ic_root')."""
    return _resolve(paths()["ic"][key])


def output_root() -> Path:
    """Absolute path for the experiment ``output.root``."""
    return _resolve(paths()["output"]["root"])


def fcnv2_ic_path(tc_id: str, init_str: str) -> Path:
    """Global FCNv2 analysis IC for a TC at an init time (``YYYYMMDDHH``)."""
    return ic_root("fcnv2_ic_root") / tc_id / "for_FCNV2" / f"analysis_{init_str}.npz"


def dlampty_ic_path(tc_id: str, init_str: str) -> Path:
    """DLAMPty combined-NetCDF IC for a TC at an init time (``YYYYMMDDHH``)."""
    return ic_root("dlampty_ic_root") / tc_id[:4] / tc_id / f"{tc_id}_{init_str}_combined.nc"


def jma_track_path(tc_id: str) -> Path:
    """JMA best-track CSV for a TC."""
    return ic_root("jma_track_root") / tc_id[:4] / f"{tc_id}.csv"
