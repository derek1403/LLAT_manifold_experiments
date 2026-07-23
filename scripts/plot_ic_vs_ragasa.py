#!/usr/bin/env python
"""Side-by-side IC validation: idealized balanced ring vs. the real RAGASA state.

    python scripts/plot_ic_vs_ragasa.py \
        --background outputs/idealized_vortex/backgrounds/quiescent_202518W_2025091700.npz \
        --config experiments/idealized_vortex/configs/base.yaml

Builds the full idealized IC (quiescent background + balanced ring from the
config's perturbation block) and compares it against the real RAGASA analysis
(same variables, RAGASA recentred on its msl minimum):

  * ``ic_vs_ragasa_rz.png`` — azimuthal-mean radius–pressure sections of
    relative vorticity ζ, tangential wind V_t, and potential temperature θ
    (shading = θ anomaly from the outermost radius → warm core).
  * ``wind_balance_ideal.png`` / ``wind_balance_ragasa.png`` — the existing
    gradient/geostrophic balance diagnostic
    (``diagnostics.wind_profile.plot_wind_balance_profile``) on each state.

Everything lands in ``outputs/idealized_vortex/ic_check/``. No DLAMPty model is
loaded — the RAGASA state is read straight from the combined NetCDF (statics the
balance plots need, e.g. the Coriolis channel, are recomputed analytically).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llat_manifold import config, io, layout                     # noqa: E402
from llat_manifold.diagnostics import wind_profile               # noqa: E402
from llat_manifold.idealized_vortex import azimuthal, background as bg  # noqa: E402
from llat_manifold.idealized_vortex.balance import DEG2M, OMEGA  # noqa: E402
from llat_manifold.idealized_vortex.driver_vortex import build_perturbation  # noqa: E402
from llat_manifold.perturbations.base import StepContext         # noqa: E402
from run_experiment import load_config                           # noqa: E402

RD = 287.05
KAPPA = RD / 1004.0


def load_ragasa_state(tc_id: str, init_str: str):
    """Light-weight RAGASA loader (no ONNX): combined nc → (upper, sfc, lats, lons).

    Statics not present in the nc (f, solar, hgt, ...) are zero-filled except
    the Coriolis channel, recomputed as 2Ω sin(lat) for the balance plot.
    """
    import xarray as xr
    grid = config.layout()["grid"]
    ds = xr.open_dataset(str(config.dlampty_ic_path(tc_id, init_str)))
    ds = ds.isel(latitude=np.arange(*grid["ic_lat_slice"]),
                 longitude=np.arange(*grid["ic_lon_slice"]))
    lats, lons = ds.latitude.values, ds.longitude.values
    ny, nx = len(lats), len(lons)

    upper = np.stack([ds[v].values for v in layout.upper_vars()], axis=-1).squeeze()
    sfc = np.zeros((ny, nx, len(layout.surface_vars()) + 2), dtype=float)
    for name in ("u10", "v10", "t2m", "d2m", "msl", "sp", "tcwv", "tp", "mtnlwrf"):
        sfc[:, :, layout.surface_index(name)] = ds[name].values.squeeze()
    lon2d, lat2d = np.meshgrid(lons, lats)
    sfc[:, :, layout.surface_index("f")] = 2 * OMEGA * np.sin(np.radians(lat2d))
    sfc[:, :, -2], sfc[:, :, -1] = lon2d, lat2d
    return upper, sfc, lats, lons


def rz_sections(upper, lats, lons, center):
    """Azimuthal-mean ζ(r,p), V_t(r,p), θ(r,p) about ``center`` (lat, lon)."""
    p = np.asarray(layout.pressure_levels(), dtype=float)
    ui, vi, ti = (layout.upper_index(k) for k in ("u", "v", "t"))
    nz = upper.shape[0]
    zeta_rp, vt_rp, th_rp = [], [], []
    for k in range(nz):
        u2, v2 = upper[k, :, :, ui], upper[k, :, :, vi]
        zeta = azimuthal.relative_vorticity(u2, v2, lats, lons)
        r1d, theta, z_rt = azimuthal.polar_sample(zeta, lats, lons, center)
        _, _, u_rt = azimuthal.polar_sample(u2, lats, lons, center)
        _, _, v_rt = azimuthal.polar_sample(v2, lats, lons, center)
        vt_rt = -u_rt * np.sin(theta)[None, :] + v_rt * np.cos(theta)[None, :]
        _, _, t_rt = azimuthal.polar_sample(upper[k, :, :, ti], lats, lons, center)
        zeta_rp.append(z_rt.mean(axis=1))
        vt_rp.append(vt_rt.mean(axis=1))
        th_rp.append(t_rt.mean(axis=1) * (1000.0 / p[k]) ** KAPPA)
    return r1d, p, np.stack(zeta_rp), np.stack(vt_rp), np.stack(th_rp)


def _row(axes, r1d, p, zeta, vt, th, label):
    import matplotlib.pyplot as plt  # noqa: F401
    r_km = r1d / 1000.0
    zmax = max(abs(zeta).max(), 1e-8)
    pm0 = axes[0].contourf(r_km, p, zeta, levels=21, cmap="RdBu_r",
                           vmin=-zmax, vmax=zmax)
    pm1 = axes[1].contourf(r_km, p, vt, levels=21, cmap="viridis")
    th_anom = th - th[:, -1:]
    tmax = max(abs(th_anom).max(), 1e-3)
    pm2 = axes[2].contourf(r_km, p, th_anom, levels=21, cmap="RdBu_r",
                           vmin=-tmax, vmax=tmax)
    cs = axes[2].contour(r_km, p, th, levels=np.arange(280, 400, 5),
                         colors="k", linewidths=0.5)
    axes[2].clabel(cs, fontsize=6, fmt="%d")
    titles = (rf"$\zeta$ (s$^{{-1}}$)", r"$V_t$ (m/s)",
              r"$\theta$ anom (shading) + $\theta$ (K)")
    for ax, pm, ti in zip(axes, (pm0, pm1, pm2), titles):
        ax.invert_yaxis()
        ax.set_title(f"{label}: {ti}", fontsize=10)
        ax.set_xlabel("radius (km)")
        ax.figure.colorbar(pm, ax=ax, shrink=0.9)
    axes[0].set_ylabel("pressure (hPa)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--background", required=True, help="quiescent background npz")
    ap.add_argument("--config",
                    default="experiments/idealized_vortex/configs/base.yaml")
    ap.add_argument("--tc-id", default="202518W")
    ap.add_argument("--init", default="2025091700")
    args = ap.parse_args(argv)

    out_dir = config.output_root() / "idealized_vortex" / "ic_check"
    data_dir = io.data_dir(out_dir)

    # ---- idealized IC = quiescent background + balanced ring ---- #
    cfg = load_config(Path(args.config).resolve())
    state = bg.load_background(args.background)
    pert = build_perturbation(cfg.get("perturbation", {}))
    ctx = StepContext(fore_i=0, lead_hr=0, initial_time=state.initial_time,
                      dlam_lats=state.dlam_lats, dlam_lons=state.dlam_lons)
    ideal_up, ideal_sfc = pert.apply_ic(state.upper.copy(), state.surface.copy(), ctx)
    io.save_delta_bundle(data_dir / "ic_ring", ideal_up, ideal_sfc)

    # ---- real RAGASA state, recentred on its msl minimum ---- #
    rag_up, rag_sfc, rlats, rlons = load_ragasa_state(args.tc_id, args.init)
    io.save_delta_bundle(data_dir / f"ragasa_{args.init}", rag_up, rag_sfc)
    msl = rag_sfc[:, :, layout.surface_index("msl")]
    iy, ix = np.unravel_index(np.argmin(msl), msl.shape)
    rag_center = (rlats[iy], rlons[ix])
    print(f"[ic_check] RAGASA msl-min centre: {rag_center[0]:.2f}N "
          f"{rag_center[1]:.2f}E ({msl.min()/100:.1f} hPa)")

    # ---- r–z comparison figure ---- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    _row(axes[0], *rz_sections(ideal_up, state.dlam_lats, state.dlam_lons, None),
         label=f"Idealized ring ({pert.param_tag})")
    _row(axes[1], *rz_sections(rag_up, rlats, rlons, rag_center),
         label=f"RAGASA {args.init}")
    fig.suptitle("Idealized balanced ring vs. real RAGASA analysis "
                 "(azimuthal-mean r–p sections)", fontsize=13)
    p_rz = out_dir / f"ic_vs_ragasa_rz_{args.init}.png"
    fig.savefig(p_rz, dpi=150)
    plt.close(fig)
    print(f"[ic_check] wrote {p_rz}")

    # ---- existing gradient/geostrophic balance diagnostic on both ---- #
    wind_profile.plot_wind_balance_profile(data_dir / "ic_ring.npz",
                                           out_dir / "wind_balance_ideal.png")
    wind_profile.plot_wind_balance_profile(
        data_dir / f"ragasa_{args.init}.npz",
        out_dir / f"wind_balance_ragasa_{args.init}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
