"""Elliptical-stage (m=2) dynamics diagnostics for the ring-vortex runs.

When the ring deforms into an ellipse before collapsing, the classical vortex
dynamics to verify are:

  * low-level ascent (ω < 0) concentrated at the **long-axis tips**;
  * flow (and upper-level tangential wind) fastest along the **short-axis**
    flanks (streamline squeezing);
  * geopotential contours tracing the same ellipse.

Products (all offline, from the saved δ bundles + the quiescent background):

  * ``w850_maps.png``    — panel grid over iterations: shading = −ω(850 hPa)
    (>0 = ascent; DLAMPty ``w`` is ω in Pa s⁻¹), black contours = total
    geopotential z(850), green dashed = ζ(850) marking the ellipse.
  * ``spd_upper_maps.png`` — same grid; shading = wind speed at an upper level
    (default 300 hPa), green dashed ζ(850) contours for the ellipse axes.
  * ``longaxis_profile.png`` — at the most-elliptical iteration (max A₂ at the
    ring radius, pre-saturation; override with ``--iter``): the m=2 phase of
    ζ(850) defines the major axis; **radial-wind V_r, tangential-wind V_t and
    −ω at 850 hPa plus V_t at the upper level** are sampled along the major
    axis (solid) and minor axis (dashed), r = 0 → 6°, averaging the two
    opposite rays — the reading of Kuo et al. (2016, JGR-A, Fig. 6): BL
    convergence/updraft at the major-axis tips, V_t max on the minor-axis
    flanks.

CLI::

    python -m llat_manifold.idealized_vortex.ellipse <run_dir> [--upper 300] [--iter N]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .. import io, layout
from ..diagnostics import experiment_title
from .azimuthal import (_geometry, _iter_bundles, panel_iters, polar_sample,
                        relative_vorticity)
from .balance import DEG2M

_OMEGA_NOTE = r"$-\omega$ (Pa s$^{-1}$, >0 = ascent)"


# --------------------------------------------------------------------------- #
# field access
# --------------------------------------------------------------------------- #
def _fields(dup, bup, level_hpa: int):
    """(u, v, -omega, z_total) at one level; δ winds ≈ absolute (quiet bg)."""
    k = layout.pressure_levels().index(level_hpa)
    ui, vi = layout.upper_index("u"), layout.upper_index("v")
    wi, zi = layout.upper_index("w"), layout.upper_index("z")
    u = dup[k, :, :, ui] + bup[k, :, :, ui]
    v = dup[k, :, :, vi] + bup[k, :, :, vi]
    w = -(dup[k, :, :, wi] + bup[k, :, :, wi])
    z = dup[k, :, :, zi] + bup[k, :, :, zi]
    return u, v, w, z


def _m2_phase_and_amp(zeta, lats, lons, rmw_deg: float):
    """(A2/A0, long-axis angle φ₂) from the θ-FFT of ζ at the ring radius."""
    r_1d, theta, z_rt = polar_sample(zeta, lats, lons)
    band = (r_1d >= 0.4 * rmw_deg * DEG2M) & (r_1d <= 1.4 * rmw_deg * DEG2M)
    spec = np.fft.rfft(z_rt[band], axis=1) / z_rt.shape[1]
    i_r = int(np.argmax(np.abs(spec[:, 2])))
    c2 = spec[i_r, 2]
    a0 = max(np.abs(spec[i_r, 0]), 1e-15)
    return float(np.abs(c2) * 2 / a0), float(np.angle(c2) / 2.0)


def _ray_sample(field, lats, lons, phi: float, r_max_deg: float = 6.0, nr: int = 61):
    """Sample a 2-D field along the outward ray at angle ``phi`` (r = 0..r_max)."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lat_c, lon_c = lats[len(lats) // 2], lons[len(lons) // 2]
    res = abs(lats[1] - lats[0])
    s = np.linspace(0.0, r_max_deg, nr)
    dlat = s * np.sin(phi)
    dlon = s * np.cos(phi) / np.cos(np.radians(lat_c))
    iy = (lats[0] - (lat_c + dlat)) / res
    ix = ((lon_c + dlon) - lons[0]) / res
    return s, map_coordinates(field, [iy, ix], order=1, mode="nearest")


def _axis_profiles(u, v, w, u_up, v_up, lats, lons, phi: float):
    """Kuo (2016, Fig. 6)-style radial profiles along one axis.

    Samples the two opposite rays (φ, φ+π), decomposes winds into the local
    **radial** (V_r, outward > 0) and **tangential** (V_t, cyclonic > 0)
    components, and averages the two rays (an m=2 structure is symmetric under
    r → −r). Returns ``(r_deg, V_r850, V_t850, −ω850, V_t_upper)``.
    """
    acc = None
    for ang in (phi, phi + np.pi):
        r_deg, us = _ray_sample(u, lats, lons, ang)
        _, vs = _ray_sample(v, lats, lons, ang)
        _, ws = _ray_sample(w, lats, lons, ang)
        _, uus = _ray_sample(u_up, lats, lons, ang)
        _, vus = _ray_sample(v_up, lats, lons, ang)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        prof = (us * cos_a + vs * sin_a,          # V_r 850
                -us * sin_a + vs * cos_a,         # V_t 850
                ws,                               # −ω 850
                -uus * sin_a + vus * cos_a)       # V_t upper
        acc = prof if acc is None else tuple(a + p for a, p in zip(acc, prof))
    return (r_deg,) + tuple(0.5 * a for a in acc)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def maps(run_dir, *, level_hpa: int = 850, upper_hpa: int = 300):
    """The two panel-grid figures (−ω + z contours; upper wind speed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    lats, lons = _geometry(run_dir)
    bup, _ = io.load_delta_bundle(run_dir / "data" / "background.npz")
    bundles = _iter_bundles(run_dir)
    show = panel_iters(bundles)
    ui, vi = layout.upper_index("u"), layout.upper_index("v")
    k_low = layout.pressure_levels().index(level_hpa)

    frames = []
    for it, path in bundles:
        if it not in show:
            continue
        dup, _ = io.load_delta_bundle(path)
        u, v, w, z = _fields(dup, bup, level_hpa)
        uu, vu, _, _ = _fields(dup, bup, upper_hpa)
        zeta = relative_vorticity(dup[k_low, :, :, ui], dup[k_low, :, :, vi],
                                  lats, lons)
        frames.append((it, w, z, np.hypot(uu, vu), zeta))

    saved = []
    plots_dir = io.plots_dir(run_dir)
    for which in ("w", "spd"):
        ncol = 4
        nrow = int(np.ceil(len(frames) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow),
                                 constrained_layout=True)
        if which == "w":
            vmax = max(max(abs(f[1]).max() for f in frames), 1e-6)
            kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            label = _OMEGA_NOTE + f" @{level_hpa} hPa"
        else:
            vmax = max(max(f[3].max() for f in frames), 1e-6)
            kw = dict(cmap="viridis", vmin=0, vmax=vmax)
            label = f"wind speed @{upper_hpa} hPa (m/s)"
        for ax, (it, w, z, spd, zeta) in zip(np.ravel(axes), frames):
            pm = ax.pcolormesh(lons, lats, w if which == "w" else spd,
                               shading="auto", **kw)
            if which == "w":
                cs = ax.contour(lons, lats, z, levels=10, colors="k",
                                linewidths=0.6)
            zl = 0.35 * max(abs(zeta).max(), 1e-9)
            ax.contour(lons, lats, zeta, levels=[zl], colors="lime",
                       linestyles="--", linewidths=1.2)
            ax.set_title(f"iter {it} (day {it * 0.125:.1f})", fontsize=10)
            ax.set_aspect("equal")
        for ax in np.ravel(axes)[len(frames):]:
            ax.axis("off")
        fig.colorbar(pm, ax=np.ravel(axes).tolist(), shrink=0.7, label=label)
        extra = " | black: z contours" if which == "w" else ""
        fig.suptitle(experiment_title(
            run_dir, prefix=("−ω" if which == "w" else "upper |V|")
            + f" maps (green dashed: ζ{level_hpa} ellipse){extra}:"), fontsize=12)
        p = plots_dir / (f"w{level_hpa}_maps.png" if which == "w"
                         else f"spd{upper_hpa}_maps.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(str(p))
        print(f"[ellipse] wrote {p}")
    return saved


def longaxis_profile(run_dir, *, level_hpa: int = 850, upper_hpa: int = 300,
                     it_pick: int | None = None, max_iter_search: int = 32):
    """Radial profiles along the long/short axes at the most-elliptical iter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    run_dir = Path(run_dir)
    lats, lons = _geometry(run_dir)
    bup, _ = io.load_delta_bundle(run_dir / "data" / "background.npz")
    bundles = _iter_bundles(run_dir)
    ui, vi = layout.upper_index("u"), layout.upper_index("v")
    k_low = layout.pressure_levels().index(level_hpa)

    with open(run_dir / "config_used.yaml") as f:
        cfg = (yaml.safe_load(f) or {}).get("resolved_config", {})
    rmw_deg = float((cfg.get("perturbation") or {}).get("rmw_deg", 2.5))

    # pick the most-elliptical pre-saturation iteration (or honor --iter)
    best = None
    for it, path in bundles:
        if it_pick is not None and it != it_pick:
            continue
        if it_pick is None and it > max_iter_search:
            break
        dup, _ = io.load_delta_bundle(path)
        zeta = relative_vorticity(dup[k_low, :, :, ui], dup[k_low, :, :, vi],
                                  lats, lons)
        ratio, phi2 = _m2_phase_and_amp(zeta, lats, lons, rmw_deg)
        if best is None or ratio > best[0]:
            best = (ratio, phi2, it, path, zeta)
    ratio, phi2, it, path, zeta = best
    print(f"[ellipse] most-elliptical iter {it} (day {it * 0.125:.1f}), "
          f"A2/A0={ratio:.2f}, long axis {np.degrees(phi2):.0f}° from east")

    dup, _ = io.load_delta_bundle(path)
    u, v, w, _ = _fields(dup, bup, level_hpa)
    uu, vu2d, _, _ = _fields(dup, bup, upper_hpa)

    prof_long = _axis_profiles(u, v, w, uu, vu2d, lats, lons, phi2)
    prof_short = _axis_profiles(u, v, w, uu, vu2d, lats, lons, phi2 + np.pi / 2)

    labels = [(rf"$V_r$ {level_hpa} hPa (m/s, outward>0)"),
              (rf"$V_t$ {level_hpa} hPa (m/s, cyclonic>0)"),
              (_OMEGA_NOTE + f" @{level_hpa} hPa"),
              (rf"$V_t$ {upper_hpa} hPa (m/s)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    r_deg = prof_long[0]
    for k, (ax, label) in enumerate(zip(np.ravel(axes), labels)):
        ax.plot(r_deg, prof_long[k + 1], "-", lw=2, color="crimson",
                label="major axis (mean of both tips)")
        ax.plot(r_deg, prof_short[k + 1], "--", lw=2, color="steelblue",
                label="minor axis (mean of both flanks)")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(rmw_deg, color="gray", ls=":", lw=1)
        ax.set_title(label, fontsize=11)
        ax.grid(ls=":", alpha=0.5)
    axes[0, 0].legend(fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("radius along axis (deg)")
    fig.suptitle(experiment_title(
        run_dir, prefix=f"Axis profiles (cf. Kuo et al. 2016 Fig. 6) @iter {it} "
        f"(day {it * 0.125:.1f}, A2/A0={ratio:.2f}):"), fontsize=12)
    fig.tight_layout()
    p = io.plots_dir(run_dir) / "longaxis_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[ellipse] wrote {p}")
    return str(p)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--level", type=int, default=850)
    ap.add_argument("--upper", type=int, default=300)
    ap.add_argument("--iter", type=int, default=None,
                    help="force the profile iteration (default: max A2/A0)")
    a = ap.parse_args(argv)
    maps(a.run_dir, level_hpa=a.level, upper_hpa=a.upper)
    longaxis_profile(a.run_dir, level_hpa=a.level, upper_hpa=a.upper,
                     it_pick=a.iter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
