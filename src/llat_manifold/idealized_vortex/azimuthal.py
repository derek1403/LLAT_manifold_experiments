"""Azimuthal-wavenumber diagnostics for the ring-vortex instability runs.

For each snapshot iteration bundle the low-level relative vorticity ζ′ is
sampled on a polar (r, θ) grid about the domain/ring centre and decomposed by
FFT in θ into azimuthal-wavenumber amplitudes A_m(r). The instability signal is
the growth of A_m (m ≥ 1) at the ring radius; the theoretical expectation for
the dominant m as a function of (γ, δ) is the Hendricks et al. (2009) phase
space (thick rings → m = 2–4, thin rings → higher m).

Fills the analysis role that ``diagnostics/modal.py`` reserves, without
touching it. Geometry (lat/lon) is read from the run's ``data/background.npz``
(surface channels −2/−1 = lon/lat).

CLI::

    python -m llat_manifold.idealized_vortex.azimuthal <run_dir> [--level 850]
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .. import io, layout
from ..diagnostics import load_stamp
from .balance import DEG2M

_ITER_RE = re.compile(r"iter(\d+)")


# --------------------------------------------------------------------------- #
# Field building blocks
# --------------------------------------------------------------------------- #
def relative_vorticity(u: np.ndarray, v: np.ndarray, lats, lons) -> np.ndarray:
    """ζ = ∂v/∂x − ∂u/∂y on the lat/lon grid (metric-aware, 2-D input)."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    y = lats * DEG2M                                   # descending is fine
    coslat = np.cos(np.radians(lats))[:, None]
    x = lons * DEG2M                                   # scaled per row below
    dv_dx = np.gradient(v, x, axis=1) / coslat
    du_dy = np.gradient(u, y, axis=0)
    return dv_dx - du_dy


def polar_sample(field: np.ndarray, lats, lons, center=None, *,
                 nr: int = 96, ntheta: int = 128, rmax_deg: float = 8.0):
    """Bilinear sample of a 2-D field on a polar (r, θ) grid about ``center``.

    Returns ``(r_1d [m], theta_1d, field_rt (nr, ntheta))``.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if center is None:
        lat_c, lon_c = lats[len(lats) // 2], lons[len(lons) // 2]
    else:
        lat_c, lon_c = center
    res = abs(lats[1] - lats[0])
    r_deg = np.linspace(0.0, rmax_deg, nr)
    theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
    rr, tt = np.meshgrid(r_deg, theta, indexing="ij")
    dlat = rr * np.sin(tt)
    dlon = rr * np.cos(tt) / np.cos(np.radians(lat_c))
    iy = (lats[0] - (lat_c + dlat)) / res              # lats descending
    ix = ((lon_c + dlon) - lons[0]) / res
    f_rt = map_coordinates(field, [iy, ix], order=1, mode="nearest")
    return r_deg * DEG2M, theta, f_rt


def wavenumber_amplitudes(field_rt: np.ndarray, m_max: int = 8) -> np.ndarray:
    """A_m(r) for m = 0..m_max from the θ-FFT of ``field_rt`` (nr, ntheta)."""
    spec = np.fft.rfft(field_rt, axis=1) / field_rt.shape[1]
    amp = np.abs(spec[:, :m_max + 1])
    amp[:, 1:] *= 2.0                                  # one-sided → physical
    return amp


# --------------------------------------------------------------------------- #
# Run-level analysis
# --------------------------------------------------------------------------- #
def _iter_bundles(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted((run_dir / "data").glob("delta_snapshot_*iter*.npz")):
        m = _ITER_RE.search(p.stem)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def _geometry(run_dir: Path):
    up, sfc = io.load_delta_bundle(run_dir / "data" / "background.npz")
    return sfc[:, 0, -1], sfc[0, :, -2]                # lats (col), lons (row)


def panel_iters(bundles, dense_until: int = 24, dense_step: int = 2,
                sparse_step: int = 16) -> set[int]:
    """Iterations to show in panel grids: dense over day 0–3, sparse after.

    The ring's whole visible evolution (deform → merge → monopole) happens in
    the first ~3 days; later panels only document the saturated monopole.
    """
    avail = [it for it, _ in bundles]
    show = {it for it in avail if it <= dense_until and (it - 1) % dense_step == 0}
    show |= {it for it in avail if it > dense_until
             and (it - dense_until) % sparse_step == 0}
    show.add(avail[-1])
    return show


def _fit_growth(iters: np.ndarray, amp: np.ndarray):
    """Exponential-fit σ over the pre-saturation rise; returns (sigma_per_day, window)."""
    if amp.max() <= 0:
        return np.nan, (0, 0)
    i_pk = int(np.argmax(amp))
    lo_amp, hi_amp = 3.0 * max(amp[0], 1e-12), 0.5 * amp[i_pk]
    sel = np.where((amp[:i_pk + 1] > lo_amp) & (amp[:i_pk + 1] < hi_amp))[0]
    if len(sel) < 4:
        return np.nan, (0, 0)
    slope = np.polyfit(iters[sel], np.log(amp[sel]), 1)[0]   # per iteration (3 h)
    return slope * 8.0, (int(sel[0]), int(sel[-1]))          # per day


def analyze_run(run_dir, *, level_hpa: int = 850, m_max: int = 8,
                rmax_deg: float = 8.0, make_plots: bool = True) -> dict:
    """A_m(r, iter) for one run; writes plots + ``data/azimuthal_Am.npz``."""
    run_dir = Path(run_dir)
    lats, lons = _geometry(run_dir)
    k_lev = layout.pressure_levels().index(level_hpa)
    ui, vi = layout.upper_index("u"), layout.upper_index("v")

    stamp = load_stamp(run_dir).get("resolved_config", {})
    pert = stamp.get("perturbation", {}) or {}
    rmw_deg = float(pert.get("rmw_deg", 2.5))

    bundles = _iter_bundles(run_dir)
    if not bundles:
        raise FileNotFoundError(f"no snapshot iteration bundles under {run_dir}/data")

    iters, amps, zeta_maps = [], [], {}
    show = panel_iters(bundles)
    for it, path in bundles:
        dup, _ = io.load_delta_bundle(path)
        zeta = relative_vorticity(dup[k_lev, :, :, ui], dup[k_lev, :, :, vi],
                                  lats, lons)
        r_1d, _, z_rt = polar_sample(zeta, lats, lons, rmax_deg=rmax_deg)
        amps.append(wavenumber_amplitudes(z_rt, m_max))
        iters.append(it)
        if it in show:
            zeta_maps[it] = zeta
    iters = np.asarray(iters)
    amps = np.stack(amps)                              # (nit, nr, m_max+1)

    # amplitude at the ring: max over 0.5–1.5 × rmw
    band = (r_1d >= 0.5 * rmw_deg * DEG2M) & (r_1d <= 1.5 * rmw_deg * DEG2M)
    a_ring = amps[:, band, :].max(axis=1)              # (nit, m_max+1)

    sigma = {m: _fit_growth(iters, a_ring[:, m])[0] for m in range(1, m_max + 1)}
    m_dom = int(np.argmax(a_ring[-1, 1:]) + 1)

    out_npz = run_dir / "data" / "azimuthal_Am.npz"
    np.savez_compressed(out_npz, iters=iters, r=r_1d, amps=amps, a_ring=a_ring,
                        sigma_per_day=np.array([sigma[m] for m in
                                                range(1, m_max + 1)]),
                        m_dominant=m_dom, level_hpa=level_hpa)
    result = {"iters": iters, "r": r_1d, "amps": amps, "a_ring": a_ring,
              "sigma_per_day": sigma, "m_dominant": m_dom, "npz": str(out_npz)}
    if make_plots:
        result["plots"] = _plots(run_dir, iters, a_ring, sigma, m_dom,
                                 zeta_maps, lats, lons, level_hpa, rmw_deg)
    return result


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plots(run_dir, iters, a_ring, sigma, m_dom, zeta_maps, lats, lons,
           level_hpa, rmw_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..diagnostics import experiment_title

    plots_dir = io.plots_dir(run_dir)
    saved = []

    # (a) ζ′ map panels
    keys = sorted(zeta_maps)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow),
                             constrained_layout=True)
    vmax = max(abs(zeta_maps[keys[-1]]).max(), 1e-8)
    for ax, it in zip(np.ravel(axes), keys):
        pm = ax.pcolormesh(lons, lats, zeta_maps[it], cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"iter {it} (day {it * 0.125:.1f})", fontsize=10)
        ax.set_aspect("equal")
    for ax in np.ravel(axes)[len(keys):]:
        ax.axis("off")
    fig.colorbar(pm, ax=np.ravel(axes).tolist(), shrink=0.7,
                 label=rf"$\zeta'$ {level_hpa} hPa (s$^{{-1}}$)")
    fig.suptitle(experiment_title(run_dir, prefix=f"ζ′ {level_hpa} hPa:"),
                 fontsize=12)
    p = plots_dir / f"zeta{level_hpa}_maps.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(str(p))

    # (b) growth curves
    fig, ax = plt.subplots(figsize=(9, 6))
    days = iters * 0.125
    for m in range(1, a_ring.shape[1]):
        lab = f"m={m}" + (f"  σ≈{sigma[m]:.2f}/d" if np.isfinite(sigma[m]) else "")
        lw = 2.5 if m == m_dom else 1.2
        ax.semilogy(days, np.maximum(a_ring[:, m], 1e-12), lw=lw, label=lab)
    ax.semilogy(days, np.maximum(a_ring[:, 0], 1e-12), "k--", lw=1.5,
                label="m=0 (ring)")
    ax.set_xlabel("equivalent days (iter × 3 h)")
    ax.set_ylabel(rf"$A_m(\zeta')$ at ring radius (s$^{{-1}}$)")
    ax.set_title(experiment_title(
        run_dir, prefix=f"Azimuthal growth (dominant m={m_dom}):"), fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    p = plots_dir / "growth_Am.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(str(p))

    # (c) final spectrum
    fig, ax = plt.subplots(figsize=(6, 4))
    ms = np.arange(1, a_ring.shape[1])
    ax.bar(ms, a_ring[-1, 1:], color="steelblue")
    ax.set_xlabel("azimuthal wavenumber m")
    ax.set_ylabel(rf"$A_m$ final iter")
    ax.set_title(f"final m-spectrum at ring radius (rmw={rmw_deg}°)", fontsize=11)
    fig.tight_layout()
    p = plots_dir / "spectrum_final.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(str(p))
    return saved


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--level", type=int, default=850)
    ap.add_argument("--mmax", type=int, default=8)
    ap.add_argument("--rmax-deg", type=float, default=8.0)
    a = ap.parse_args(argv)
    res = analyze_run(a.run_dir, level_hpa=a.level, m_max=a.mmax,
                      rmax_deg=a.rmax_deg)
    print(f"[azimuthal] dominant m = {res['m_dominant']}; "
          f"σ/day = { {m: round(s, 3) for m, s in res['sigma_per_day'].items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
