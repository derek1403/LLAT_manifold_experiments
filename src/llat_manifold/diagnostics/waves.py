"""Outward-propagation diagnostics: gravity-wave radiation & Rossby-radius adjustment.

The LLAT grid is too coarse to resolve polygonal-eyewall wavenumbers directly, so the
physical focus is the *consequences* that do reach our scales:

1. **Gravity-wave radiation** — a radius–time (Hovmöller) plot of an axisymmetric-mean
   perturbation field (default: DLAMPty surface MSL anomaly). An outward-tilting
   phase line gives the apparent radial phase speed of the radiating waves.
2. **Rossby-radius adjustment** — overlay the local Rossby radius of deformation
   L_R = N H / f (a single representative value here) so the perturbation's horizontal
   spreading can be compared against the adjustment scale that separates a
   gravity-wave-dominated (r < L_R) from a balanced (r > L_R) response.

First cut; reads the per-lead delta bundles produced by the continuous driver.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import io, layout
from . import experiment_title


def _radial_mean(field2d: np.ndarray, nbins: int | None = None) -> np.ndarray:
    """Azimuthal mean about the domain centre -> profile vs radius (grid units)."""
    ny, nx = field2d.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    nbins = nbins or int(min(cy, cx))
    rbin = np.clip(r.astype(int), 0, nbins - 1)
    prof = np.zeros(nbins)
    cnt = np.zeros(nbins)
    np.add.at(prof, rbin.ravel(), field2d.ravel())
    np.add.at(cnt, rbin.ravel(), 1.0)
    return prof / np.maximum(cnt, 1.0)


def _lead_of(path: Path) -> int:
    m = re.search(r"lead(\d+)hr", path.name)
    return int(m.group(1)) if m else -1


def rossby_radius_km(N=1e-2, H=1.0e4, f=5e-5) -> float:
    """First-baroclinic Rossby radius L_R = N H / f, in km (rough representative)."""
    return (N * H / f) / 1000.0


def _nice_vmax(value: float) -> float:
    """Round a positive magnitude up to a clean 1/2/5 × 10^k for fixed colorbars."""
    if value <= 0 or not np.isfinite(value):
        return 1.0
    exp = np.floor(np.log10(value))
    frac = value / 10 ** exp
    nice = 1.0 if frac <= 1 else 2.0 if frac <= 2 else 5.0 if frac <= 5 else 10.0
    return nice * 10 ** exp


def hovmoller(out_dir, surface_var="msl", out_png=None):
    """Radius–time Hovmöller of an axisymmetric-mean surface anomaly.

    Scans the delta bundles in ``out_dir`` (one per lead), builds the radial profile
    of the chosen DLAMPty surface variable's perturbation, and stacks them by lead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    files = sorted([p for p in out_dir.glob("delta_*lead*hr.npz")], key=_lead_of)
    if not files:
        raise FileNotFoundError(f"No 'delta_*lead*hr.npz' bundles in {out_dir}")

    s_idx = layout.surface_index(surface_var)
    res_km = float(layout.config.layout()["grid"]["resolution_deg"]) * 111.0

    leads, profiles = [], []
    for p in files:
        _, dsfc = io.load_delta_bundle(p)
        profiles.append(_radial_mean(dsfc[:, :, s_idx]))
        leads.append(_lead_of(p))
    H = np.array(profiles)                      # (ntime, nradius)
    radius_km = np.arange(H.shape[1]) * res_km
    leads = np.array(leads)

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = _nice_vmax(np.nanmax(np.abs(H)))
    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(radius_km, leads, np.clip(H, -vmax, vmax), levels=levels,
                     cmap="RdBu_r", extend="both")
    ax.axvline(rossby_radius_km(), color="k", ls="--", lw=1.2,
               label=f"$L_R$ ≈ {rossby_radius_km():.0f} km")
    ax.set_xlabel("Radius (km)", fontsize=12, weight="bold")
    ax.set_ylabel("Lead time (h)", fontsize=12, weight="bold")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))
    cbar.set_label(f"Δ{surface_var} (axisym. mean)", fontsize=11, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(experiment_title(out_dir, prefix="radius–time:"), fontsize=11, weight="bold")

    if out_png is not None:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"[waves] wrote {out_png}")
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="radius–time Hovmöller of a surface anomaly")
    ap.add_argument("out_dir", help="experiment output dir with delta bundles")
    ap.add_argument("--var", default="msl")
    ap.add_argument("--out", default="hovmoller.png")
    a = ap.parse_args()
    hovmoller(a.out_dir, a.var, a.out)
