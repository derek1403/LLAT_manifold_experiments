"""Non-hydrostatic (靜力平衡崩潰) diagnostics — two complementary balance checks.

Companion to :mod:`wind_profile` but on the **vertical** axis: every figure here puts
pressure on the y-axis. Two independent strategies quantify how far the model output
departs from hydrostatic balance ``∂P/∂z ≈ −ρg``:

Strategy 1 — vertical-momentum residual (the genuine dynamical test).
    ``Dw/Dt = −(1/ρ)∂P/∂z − g``; the non-dimensional non-hydrostatic parameter is
    ``ε = |Dw/Dt| / g``. We form a single-snapshot, steady-state estimate of the material
    derivative in **spherical** coordinates, including the curvature/metric term::

        Dw/Dt ≈ u·∂w/∂x + v·∂w/∂y + w·∂w/∂z − (u²+v²)/A_EARTH

    ``∂w/∂t`` is neglected because ``snapshot`` mode freezes valid time. The metric
    (centripetal) term is small globally but contributes ~10–20% of ε in a strong eyewall,
    so the plots expose advective-only vs advective+metric explicitly.

Strategy 2 — isobaric thermodynamic / z↔T consistency check.
    In pressure coordinates the hydrostatic equation reads ``∂Φ/∂P = −α = −RT/P`` with
    ``Φ = gz``. Because AI models inherit ERA5 isobaric heights that were *built*
    hydrostatically (hypsometric integration of T), a tiny residual here mainly confirms
    the model kept its z and T fields vertically consistent — it is a coordinate/thermo
    consistency check rather than a dynamics test.

Both operate on a single DLAMPty bundle (absolute state ū+δ). A ``baseline`` may be added
to reconstruct the absolute state from a δ bundle. ``w`` is assumed to be vertical velocity
in m/s (the ε band it produces — 10⁻⁶–10⁻⁴ — confirms this; a runtime print of w's range
makes a unit surprise visible).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import io
from . import experiment_title
from . import _idealized as _id

G = _id.G            # 9.80665, same constant fields_from_bundle uses for geopotential→height
RD = _id.RD          # 287.0
A_EARTH = _id.A_EARTH
OMEGA = 7.2921e-5

# Azimuthal binning: 10 km rings, matching wind_profile's radial resolution.
_DR_M = 10000.0
# Eyewall annulus half-width about the radius of maximum wind.
_EYEWALL_HALFWIDTH_M = 50000.0
# Floor on the RMW search: the innermost few bins are coarsely resolved and can spuriously
# peak the azimuthal-mean tangential wind, so only look for the eyewall at r ≥ this.
_RMW_MIN_M = 50000.0


# --------------------------------------------------------------------------- #
# Shared geometry / azimuthal helpers
# --------------------------------------------------------------------------- #
def _absolute_state(bundle_path, baseline_path):
    """Load (upper, sfc); if a baseline δ-reference is given, add it to get ū+δ."""
    up, sfc = io.load_delta_bundle(bundle_path)
    if baseline_path is not None:
        bup, bsfc = io.load_delta_bundle(baseline_path)
        up = up + bup
        sfc = sfc + bsfc
    return up, sfc


def _geometry(sfc):
    """Radius/azimuth grids about the domain centre (port of wind_profile 82–93)."""
    ny, nx = sfc.shape[0], sfc.shape[1]
    ic, jc = ny // 2, nx // 2
    from .. import layout
    f_center = sfc[:, :, layout.surface_index("f")][ic, jc]
    center_lat = np.degrees(np.arcsin(np.clip(f_center / (2 * OMEGA), -1, 1)))
    dx = 0.25 * (np.pi / 180.0) * A_EARTH * np.cos(np.radians(center_lat))
    dy = 0.25 * (np.pi / 180.0) * A_EARTH
    y_idx, x_idx = np.indices((ny, nx))
    x_dist = (x_idx - jc) * dx
    y_dist = (ic - y_idx) * dy
    r_2d = np.sqrt(x_dist ** 2 + y_dist ** 2)
    theta_2d = np.arctan2(y_dist, x_dist)
    return r_2d, theta_2d, center_lat, f_center


def _radial_bins(r_2d):
    bins = np.arange(0, r_2d.max() + _DR_M, _DR_M)
    r_1d = 0.5 * (bins[:-1] + bins[1:])
    return bins, r_1d


def _radial_mean_levels(field3d, r_2d, bins):
    """Azimuthal mean per level → (nlev, nbin). NaN where a ring is empty."""
    idx = np.digitize(r_2d, bins)
    nlev = field3d.shape[0]
    nbin = len(bins) - 1
    out = np.full((nlev, nbin), np.nan)
    for i in range(1, len(bins)):
        m = (idx == i)
        if np.any(m):
            out[:, i - 1] = np.nanmean(field3d[:, m], axis=1)
    return out


def _eyewall_mask(vt850_2d, r_2d, bins, r_1d):
    """(annulus_mask_2d, rmw_km) from the radius of max azimuthal-mean 850 hPa Vt."""
    prof = _radial_mean_levels(vt850_2d[None], r_2d, bins)[0]
    # Restrict the RMW search to r ≥ _RMW_MIN_M; the inner bins are coarsely resolved.
    search = np.where(r_1d >= _RMW_MIN_M, prof, np.nan)
    if np.all(np.isnan(search)):
        return np.zeros_like(r_2d, dtype=bool), np.nan
    rmw = r_1d[int(np.nanargmax(search))]
    mask = np.abs(r_2d - rmw) <= _EYEWALL_HALFWIDTH_M
    return mask, rmw / 1000.0


def _pressure_yaxis(ax, p_hpa):
    """Log pressure y-axis, 1000 hPa at the bottom, labelled at the model levels."""
    ax.set_yscale("log")
    ax.set_ylim(1000, _id.P_TOP_HPA)
    ax.set_yticks(p_hpa)
    ax.set_yticklabels([f"{int(p)}" for p in p_hpa])
    ax.minorticks_off()
    ax.set_ylabel("Pressure [hPa]", fontsize=13, weight="bold")


def _xsection_path(out_png):
    p = Path(out_png)
    return str(p.with_name(p.stem.replace("_profile", "") + "_xsection.png"))


# --------------------------------------------------------------------------- #
# Strategy 1 — vertical-momentum residual ε = |Dw/Dt| / g
# --------------------------------------------------------------------------- #
def _dwdt_fields(up, sfc):
    """Steady-state spherical Dw/Dt → (eps_full, eps_adv, w, z, p_hpa)."""
    u, v, t, z, w, f, lat2d, lon2d, p = _id.fields_from_bundle(up, sfc)
    lat_r = np.deg2rad(lat2d)
    lon_r = np.deg2rad(lon2d)
    # Spherical horizontal spacing (axis 1 = lat, axis 2 = lon for this domain).
    dy = A_EARTH * np.gradient(lat_r, axis=0)
    dx = A_EARTH * np.cos(lat_r) * np.gradient(lon_r, axis=1)
    dy3 = np.where(dy[None] == 0, 1e-10, dy[None])
    dx3 = np.where(dx[None] == 0, 1e-10, dx[None])
    dwdx = np.gradient(w, axis=2) / dx3
    dwdy = np.gradient(w, axis=1) / dy3
    # Pointwise vertical gradient — divide the two pressure-axis gradients so the
    # non-uniform level spacing and the 3-D height field are both handled locally.
    dwdz = np.gradient(w, axis=0) / np.gradient(z, axis=0)
    adv = u * dwdx + v * dwdy + w * dwdz
    metric = -(u ** 2 + v ** 2) / A_EARTH
    eps_adv = np.abs(adv) / G
    eps_full = np.abs(adv + metric) / G
    return eps_full, eps_adv, w, z, np.array(p)


def plot_nonhydrostatic_epsilon(bundle_path, out_png=None, baseline_path=None, *,
                                which="both"):
    """ε = |Dw/Dt|/g profile (+ azimuthal-mean radius–height map).

    ``which`` selects which figure(s) to build/save: ``"both"`` (default; writes the
    profile to ``out_png`` and the map to ``*_xsection.png``), ``"profile"``, or
    ``"xsection"``. Returns the *profile* figure for ``"both"``/``"profile"`` and the
    *map* figure for ``"xsection"`` (so the evolution renderer can stamp+save it).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    bundle_path = Path(bundle_path)
    run_dir = bundle_path.parent.parent
    up, sfc = _absolute_state(bundle_path, baseline_path)
    from .. import layout

    eps_full, eps_adv, w, z, p = _dwdt_fields(up, sfc)
    r_2d, theta_2d, _, _ = _geometry(sfc)
    print(f"[hydrostatic] w range [{np.nanmin(w):+.3f}, {np.nanmax(w):+.3f}] "
          f"mean|w|={np.nanmean(np.abs(w)):.3f}  (assumed m/s)")

    bins, r_1d = _radial_bins(r_2d)
    r_km = r_1d / 1000.0

    # Eyewall annulus from the low-level tangential wind.
    k850 = layout.pressure_levels().index(850)
    u850 = up[k850, :, :, layout.upper_index("u")]
    v850 = up[k850, :, :, layout.upper_index("v")]
    vt850 = -u850 * np.sin(theta_2d) + v850 * np.cos(theta_2d)
    eye_mask, rmw_km = _eyewall_mask(vt850, r_2d, bins, r_1d)
    result = None

    # ---- Profile figure -------------------------------------------------- #
    if which in ("both", "profile"):
        def _prof(field3d, mask):
            return (np.nanmean(field3d[:, mask], axis=1) if np.any(mask)
                    else np.full(field3d.shape[0], np.nan))

        full_dom = np.nanmean(eps_full.reshape(eps_full.shape[0], -1), axis=1)
        full_eye = _prof(eps_full, eye_mask)
        full_max = np.nanmax(eps_full.reshape(eps_full.shape[0], -1), axis=1)
        adv_dom = np.nanmean(eps_adv.reshape(eps_adv.shape[0], -1), axis=1)
        adv_eye = _prof(eps_adv, eye_mask)

        fig, ax = plt.subplots(figsize=(8, 9))
        ax.plot(full_dom, p, color="black", lw=2.2, label="ε domain-mean (adv+metric)")
        ax.plot(full_eye, p, color="crimson", lw=2.2,
                label=f"ε eyewall (RMW≈{rmw_km:.0f} km)")
        ax.plot(full_max, p, color="darkorange", lw=1.6, ls=":", label="ε domain-max")
        ax.plot(adv_dom, p, color="black", lw=1.4, ls="--", alpha=0.7,
                label="ε domain-mean (advective only)")
        ax.plot(adv_eye, p, color="crimson", lw=1.4, ls="--", alpha=0.7,
                label="ε eyewall (advective only)")
        ax.axvspan(1e-4, 1e-3, color="green", alpha=0.10,
                   label="hydrostatic band (10⁻⁴–10⁻³)")
        ax.set_xscale("log")
        ax.set_xlim(1e-8, 1e0)
        _pressure_yaxis(ax, p)
        ax.set_xlabel(r"$\epsilon = |Dw/Dt|\,/\,g$", fontsize=13, weight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(experiment_title(run_dir, prefix="Non-hydrostatic ε:")
                     + "\n(Dw/Dt steady, advective+metric; ∂w/∂t neglected)",
                     fontsize=12, weight="bold")
        fig.tight_layout()
        if out_png is not None:
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            print(f"[hydrostatic] wrote {out_png}")
        result = fig

    # ---- Azimuthal-mean radius–height map -------------------------------- #
    if which in ("both", "xsection"):
        eps_rz = _radial_mean_levels(eps_full, r_2d, bins)
        z_rz = _radial_mean_levels(z, r_2d, bins) / 1000.0
        r_grid = np.tile(r_km, (eps_rz.shape[0], 1))
        log_eps = np.log10(np.clip(eps_rz, 1e-8, None))

        fig2, ax2 = plt.subplots(figsize=(13, 7))
        levels = np.arange(-7, -2.99, 0.25)   # ε is ~10⁻⁷–10⁻³ here; focus the contrast
        cf = ax2.contourf(r_grid, z_rz, log_eps, levels=levels, cmap="inferno", extend="both")
        # pad clears the right-hand pressure twin axis before the colorbar.
        cbar = fig2.colorbar(cf, ax=ax2, pad=0.11, aspect=30)
        cbar.set_label(r"$\log_{10}\,\epsilon$", fontsize=12, weight="bold")
        if np.isfinite(rmw_km):
            ax2.axvline(rmw_km, color="cyan", lw=1.4, ls="--", label=f"RMW≈{rmw_km:.0f} km")
            ax2.legend(loc="upper right", fontsize=10)
        ax2.set_xlim(0, min(750, r_km.max()))
        # Ceiling at 200 hPa (_idealized.P_TOP_HPA): above it the model's own error
        # dominates. z_rz is (level, radius), so its radial mean gives each level's
        # height for the pressure -> altitude conversion.
        ax2.set_ylim(float(np.nanmin(z_rz)),
                     _id.z_top_km(layout.pressure_levels(),
                                  np.nanmean(z_rz, axis=1)))
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.set_xlabel("Radius from Storm Center [km]", fontsize=13, weight="bold")
        ax2.set_ylabel("Altitude [km]", fontsize=13, weight="bold")
        axp = ax2.twinx()
        axp.set_ylim(ax2.get_ylim())
        mean_z_km = np.nanmean(z_rz, axis=1)
        axp.set_yticks(mean_z_km)
        axp.set_yticklabels([f"{int(pp)}" for pp in p])
        axp.set_ylabel("Pressure [hPa]", fontsize=13, weight="bold", color="#333333")
        ax2.set_title(experiment_title(run_dir, prefix="Non-hydrostatic ε (azimuthal mean):"),
                      fontsize=12, weight="bold")
        fig2.tight_layout()
        if out_png is not None:
            xpath = _xsection_path(out_png)
            fig2.savefig(xpath, dpi=200, bbox_inches="tight")
            print(f"[hydrostatic] wrote {xpath}")
        if which == "xsection":
            result = fig2
    return result


# --------------------------------------------------------------------------- #
# Strategy 2 — isobaric thermodynamic / z↔T consistency check
# --------------------------------------------------------------------------- #
def plot_hydrostatic_thermo(bundle_path, out_png=None, baseline_path=None, *,
                            which="both"):
    """∂Φ/∂P vs −RT/P profiles + fractional residual (+ r–z map).

    ``which`` ∈ {``"both"``, ``"profile"``, ``"xsection"``} — see
    :func:`plot_nonhydrostatic_epsilon`. Returns the profile figure for
    ``"both"``/``"profile"`` and the residual-map figure for ``"xsection"``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    bundle_path = Path(bundle_path)
    run_dir = bundle_path.parent.parent
    up, sfc = _absolute_state(bundle_path, baseline_path)

    u, v, t, z, w, f, lat2d, lon2d, p = _id.fields_from_bundle(up, sfc)
    p = np.array(p)
    p_pa = p * 100.0
    phi = G * z
    lhs = np.gradient(phi, p_pa, axis=0)          # ∂Φ/∂P
    rhs = -RD * t / p_pa[:, None, None]           # −RT/P
    resid_frac = (lhs - rhs) / np.abs(rhs)

    r_2d, theta_2d, _, _ = _geometry(sfc)
    bins, r_1d = _radial_bins(r_2d)
    r_km = r_1d / 1000.0
    from .. import layout
    k850 = layout.pressure_levels().index(850)
    u850 = up[k850, :, :, layout.upper_index("u")]
    v850 = up[k850, :, :, layout.upper_index("v")]
    vt850 = -u850 * np.sin(theta_2d) + v850 * np.cos(theta_2d)
    eye_mask, rmw_km = _eyewall_mask(vt850, r_2d, bins, r_1d)
    result = None

    # ---- Profile figure (two panels) ------------------------------------ #
    if which in ("both", "profile"):
        lhs_dom = np.nanmean(lhs.reshape(lhs.shape[0], -1), axis=1)
        rhs_dom = np.nanmean(rhs.reshape(rhs.shape[0], -1), axis=1)
        rf_dom = np.nanmean(resid_frac.reshape(resid_frac.shape[0], -1), axis=1)
        rf_eye = (np.nanmean(resid_frac[:, eye_mask], axis=1) if np.any(eye_mask)
                  else np.full(resid_frac.shape[0], np.nan))

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 9))
        axL.plot(lhs_dom, p, color="black", lw=2.2, label=r"LHS  $\partial\Phi/\partial P$")
        axL.plot(rhs_dom, p, color="red", lw=1.8, ls="--", label=r"RHS  $-RT/P$")
        _pressure_yaxis(axL, p)
        axL.set_xlabel(r"$\partial\Phi/\partial P$  [m$^2$ s$^{-2}$ Pa$^{-1}$]",
                       fontsize=12, weight="bold")
        axL.grid(True, which="both", linestyle="--", alpha=0.4)
        axL.legend(loc="upper left", fontsize=11)
        axL.set_title("Hydrostatic equation (isobaric)", fontsize=12, weight="bold")

        axR.plot(100 * rf_dom, p, color="black", lw=2.2, label="domain-mean")
        axR.plot(100 * rf_eye, p, color="crimson", lw=2.0,
                 label=f"eyewall (RMW≈{rmw_km:.0f} km)")
        axR.axvline(0, color="grey", lw=0.8)
        _pressure_yaxis(axR, p)
        axR.set_xlim(-8, 8)
        axR.set_xlabel("Fractional residual  (LHS−RHS)/|RHS|  [%]",
                       fontsize=12, weight="bold")
        axR.grid(True, which="both", linestyle="--", alpha=0.4)
        axR.legend(loc="upper left", fontsize=11)
        axR.set_title("z↔T hydrostatic consistency", fontsize=12, weight="bold")
        fig.suptitle(experiment_title(run_dir, prefix="Hydrostatic check:"),
                     fontsize=13, weight="bold")
        fig.tight_layout()
        if out_png is not None:
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            print(f"[hydrostatic] wrote {out_png}")
        result = fig

    # ---- Azimuthal-mean radius–height residual map ----------------------- #
    if which in ("both", "xsection"):
        rf_rz = _radial_mean_levels(resid_frac, r_2d, bins) * 100.0
        z_rz = _radial_mean_levels(z, r_2d, bins) / 1000.0
        r_grid = np.tile(r_km, (rf_rz.shape[0], 1))

        fig2, ax2 = plt.subplots(figsize=(13, 7))
        vmax = 5.0
        cf = ax2.contourf(r_grid, z_rz, np.clip(rf_rz, -vmax, vmax),
                          levels=np.linspace(-vmax, vmax, 21), cmap="bwr", extend="both")
        # pad clears the right-hand pressure twin axis before the colorbar.
        cbar = fig2.colorbar(cf, ax=ax2, pad=0.11, aspect=30)
        cbar.set_label("Fractional residual [%]", fontsize=12, weight="bold")
        if np.isfinite(rmw_km):
            ax2.axvline(rmw_km, color="black", lw=1.2, ls="--")
        ax2.set_xlim(0, min(750, r_km.max()))
        # Ceiling at 200 hPa (_idealized.P_TOP_HPA): above it the model's own error
        # dominates. z_rz is (level, radius), so its radial mean gives each level's
        # height for the pressure -> altitude conversion.
        ax2.set_ylim(float(np.nanmin(z_rz)),
                     _id.z_top_km(layout.pressure_levels(),
                                  np.nanmean(z_rz, axis=1)))
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.set_xlabel("Radius from Storm Center [km]", fontsize=13, weight="bold")
        ax2.set_ylabel("Altitude [km]", fontsize=13, weight="bold")
        axp = ax2.twinx()
        axp.set_ylim(ax2.get_ylim())
        mean_z_km = np.nanmean(z_rz, axis=1)
        axp.set_yticks(mean_z_km)
        axp.set_yticklabels([f"{int(pp)}" for pp in p])
        axp.set_ylabel("Pressure [hPa]", fontsize=13, weight="bold", color="#333333")
        ax2.set_title(experiment_title(run_dir, prefix="Hydrostatic residual (azimuthal mean):"),
                      fontsize=12, weight="bold")
        fig2.tight_layout()
        if out_png is not None:
            xpath = _xsection_path(out_png)
            fig2.savefig(xpath, dpi=200, bbox_inches="tight")
            print(f"[hydrostatic] wrote {xpath}")
        if which == "xsection":
            result = fig2
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="non-hydrostatic balance diagnostics")
    ap.add_argument("bundle_npz")
    ap.add_argument("--baseline", default=None, help="δ-reference to add (→ absolute state)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--strategy", choices=["1", "2", "both"], default="both")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.strategy in ("1", "both"):
        plot_nonhydrostatic_epsilon(a.bundle_npz, str(out / "hydrostatic_eps_profile.png"),
                                    baseline_path=a.baseline)
    if a.strategy in ("2", "both"):
        plot_hydrostatic_thermo(a.bundle_npz, str(out / "hydrostatic_thermo_profile.png"),
                                baseline_path=a.baseline)
