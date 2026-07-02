"""Perturbation Integrated Kinetic Energy (ΔIKE) vs iteration — one figure per run.

The operational dashboard (Part 1) tracks **total / inner / outer IKE** of a TC (the
right column of ``claude_RegionalCouple_AI/docs/images/dashboard.png``). Here we ask the
perturbation question instead: for each integration step *k*, how much does the storm's
IKE change because of the injected anomaly,

    ΔIKE_k = IKE(ū + δ_k) − IKE(ū),

split into **total**, **inner** (r < 200 km) and **outer** (r ≥ 200 km). Unlike the
cross-section diagnostics this is **not** an evolution movie: it is a single summary curve
per run, x = iteration (or lead), y = ΔIKE [TJ].

The IKE itself is computed by the *verbatim* Part-1 scientific core
(:mod:`regional_couple.diagnostics.ike`): the 10 m wind is transformed to storm-centred
polar coordinates, azimuthally averaged to a tangential profile, and integrated over
5 km annuli (``0.5·ρ·V_t²·2πr·H·dr``, ρ=H=1, → TJ). We reuse it directly so our ΔIKE is
on exactly the same definition as the dashboard.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import io, layout
from . import experiment_title, load_stamp

# Part-1 IKE core (polar transform + tangential profile + IKE integral), reused verbatim.
from regional_couple.diagnostics import ike as _rc


def _key(p: Path) -> int:
    """Iteration/lead index from a bundle name (mirrors suite._key; kept local to avoid
    a circular import, since suite imports this module)."""
    m = re.search(r"(?:lead|iter)(\d+)", p.name)
    return int(m.group(1)) if m else -1

_INNER_KM = 200.0          # dashboard split: inner r<200 km, outer r≥200 km
_THETA_SMOOTH = 3          # azimuthal smoothing half-width (matches ike_for_run default)


def _ike_profile(sfc, theta_smooth_num=_THETA_SMOOTH):
    """IKE radial profile [TJ per 5 km bin] from a DLAMPty surface field (81,81,20)."""
    u10 = sfc[:, :, layout.surface_index("u10")]
    v10 = sfc[:, :, layout.surface_index("v10")]
    lat2d = sfc[:, :, -1]
    lon2d = sfc[:, :, -2]
    lat0 = float(np.mean(lat2d))
    lon0 = float(np.mean(lon2d))
    # Use the data's *own* 1D axes (lat constant down a column, lon along a row). This is
    # exact and avoids np.arange float-overshoot in Part-1's _centered_grid, which can
    # yield an 82-point axis for some storm centres and break the polar transform.
    lat_1d = lat2d[:, 0]
    lon_1d = lon2d[0, :]

    r, theta, u_rt = _rc.latlon_to_polar(u10, lat_1d, lon_1d, lat0, lon0)
    _, _, v_rt = _rc.latlon_to_polar(v10, lat_1d, lon_1d, lat0, lon0)
    u_p, v_p = np.zeros_like(u_rt), np.zeros_like(v_rt)
    for shift in range(-theta_smooth_num, theta_smooth_num + 1):
        u_p += np.roll(u_rt, shift, axis=0)
        v_p += np.roll(v_rt, shift, axis=0)
    u_p /= (2 * theta_smooth_num + 1)
    v_p /= (2 * theta_smooth_num + 1)
    tan_bar, _ = _rc.tangential_and_radial(u_p, v_p, theta)
    return r, _rc.IKE(tan_bar, r)


def _components(r_km, profile, inner_km=_INNER_KM):
    """(total, inner, outer) IKE [TJ] from a radial profile; inner = r<inner_km."""
    inner_mask = r_km < inner_km
    total = float(np.nansum(profile))
    inner = float(np.nansum(profile[inner_mask]))
    outer = float(np.nansum(profile[~inner_mask]))
    return total, inner, outer


def _baseline_path(bundle: Path, data: Path, mode: str):
    """Control/background bundle to subtract for ΔIKE (or None for forward mode)."""
    if mode == "snapshot":
        p = data / "background.npz"
    elif mode == "continuous":
        p = data / bundle.name.replace("delta_", "control_")
    else:
        return None
    return p if p.exists() else None


def plot_ike_perturbation(run_dir, out_png=None, *, inner_km=_INNER_KM,
                          theta_smooth_num=_THETA_SMOOTH):
    """ΔIKE (total/inner/outer) vs iteration for one run — a single summary figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    data = run_dir / "data"
    bundles = sorted(data.glob("delta_*.npz"), key=_key)
    if not bundles:
        raise FileNotFoundError(f"no δ bundles in {data}")
    mode = load_stamp(run_dir).get("resolved_config", {}).get("mode", "")

    # Background IKE is fixed in snapshot mode, so cache it by baseline path.
    bg_cache: dict[str, tuple] = {}

    def _bg_components(bpath: Path):
        key = str(bpath)
        if key not in bg_cache:
            _, bsfc = io.load_delta_bundle(bpath)
            r, prof = _ike_profile(bsfc, theta_smooth_num)
            bg_cache[key] = _components(r, prof, inner_km)
        return bg_cache[key]

    iters, d_tot, d_inn, d_out = [], [], [], []
    for b in bundles:
        bpath = _baseline_path(b, data, mode)
        if bpath is None:
            print(f"[ike] {b.name}: no baseline ({mode}) — skipped")
            continue
        up_d, sfc_d = io.load_delta_bundle(b)
        _, bsfc = io.load_delta_bundle(bpath)
        r, prof_p = _ike_profile(sfc_d + bsfc, theta_smooth_num)   # absolute ū+δ
        tot_p, inn_p, out_p = _components(r, prof_p, inner_km)
        tot_b, inn_b, out_b = _bg_components(bpath)
        iters.append(_key(b))
        d_tot.append(tot_p - tot_b)
        d_inn.append(inn_p - inn_b)
        d_out.append(out_p - out_b)

    if not iters:
        raise RuntimeError(f"no ΔIKE points (mode={mode!r}); need a control/background")

    x = np.array(iters)
    xlabel = "Iteration" if mode == "snapshot" else "Lead step"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(x, d_tot, color="black", lw=2.4, marker="o", ms=4, label="ΔIKE total")
    ax.plot(x, d_inn, color="tab:blue", lw=2.0, marker="^", ms=4,
            label=f"ΔIKE inner (r<{inner_km:.0f} km)")
    ax.plot(x, d_out, color="tab:red", lw=2.0, marker="s", ms=4,
            label=f"ΔIKE outer (r≥{inner_km:.0f} km)")
    ax.set_xlabel(xlabel, fontsize=13, weight="bold")
    ax.set_ylabel("ΔIKE  (perturbed − background)  [TJ]", fontsize=13, weight="bold")
    ax.set_xlim(x.min(), x.max())
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=11)
    ax.set_title(experiment_title(run_dir, prefix="Perturbation IKE response:"),
                 fontsize=13, weight="bold")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[ike] wrote {out_png}")
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ΔIKE (total/inner/outer) vs iteration")
    ap.add_argument("run_dir", help="experiment run dir (contains data/ and plots/)")
    ap.add_argument("--out", default="ike_perturbation.png")
    ap.add_argument("--inner-km", type=float, default=_INNER_KM)
    a = ap.parse_args()
    plot_ike_perturbation(a.run_dir, a.out, inner_km=a.inner_km)
