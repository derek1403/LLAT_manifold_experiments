"""B3 (basics) — is the idealized initial condition still a typhoon, and how long?

The runs this study now leads with start from an *axisymmetric* vortex: the real
RAGASA analysis, azimuthally averaged about the domain centre. That removes the
"maybe some asymmetric feature of that particular day did it" objection, but it
raises two of its own, and this figure answers both.

**Rows 1–2 — did the averaging keep the storm?** Azimuthal-mean radius–pressure
sections of relative vorticity ζ, tangential wind V_t and potential temperature θ,
for the idealized IC (top) against the real analysis it came from (bottom). The
averaging is supposed to remove azimuthal structure and nothing else, so these two
rows should look nearly identical — same vorticity core, same RMW, same warm core.
They do; the peak V_t and RMW are printed on the panels so it is checkable, not a
matter of impression.

**Row 3 — how long does the symmetry last?** Only the atmosphere was averaged. SST,
Coriolis, terrain and the land mask were deliberately left as the real fields, so
the lower boundary is *not* axisymmetric and will re-imprint asymmetry from the
first step. This panel measures that directly: the azimuthal variability of the
control trajectory's 850 hPa temperature, per lead, for both ICs. The idealized run
starts near zero and climbs; where it meets the real-IC line is where "idealized
vortex" stops being a meaningful description. Any claim resting on these runs has to
live inside that window, and the panel is what says how wide it is.

Sections stop at 200 hPa.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _data as D
import style as S
from llat_manifold import io, layout, operators
from llat_manifold.diagnostics import _idealized as _id
from llat_manifold.diagnostics.response import _continuous_pairs
from llat_manifold.idealized_vortex import axisymmetric as ax
from llat_manifold.idealized_vortex import background as bg

TC_ID = "202518W"


def _ic_states(init: str):
    """(idealized InitialState, real InitialState) for one init time."""
    from llat_manifold import config, driver
    ideal = bg.load_background(
        config.output_root() / "axisymmetric_ic" / f"axisym_{TC_ID}_{init}.npz")
    real = driver.load_initial_state(TC_ID, init, operators.load_models())
    return ideal, real


def _draw_row(axes, state, label):
    r_km, p, zeta, vt, th = ax.rz_sections(state.upper, state.surface)
    i = int(np.nanargmax(vt[layout.pressure_levels().index(850)]))
    th_an = th - th[:, -1:]                     # warm core = θ relative to the rim

    zlim = float(np.nanpercentile(np.abs(zeta), 99)) or 1e-6
    cf0 = axes[0].contourf(r_km, p, zeta, levels=np.linspace(-zlim, zlim, 21),
                           cmap=S.CMAP_PV, extend="both")
    cf1 = axes[1].contourf(r_km, p, vt, levels=21, cmap="viridis")
    tlim = float(np.nanpercentile(np.abs(th_an), 99)) or 1e-3
    cf2 = axes[2].contourf(r_km, p, th_an, levels=np.linspace(-tlim, tlim, 21),
                           cmap=S.CMAP_PV, extend="both")
    axes[1].annotate(f"peak V$_t$ = {vt[layout.pressure_levels().index(850)][i]:.1f} m/s\n"
                     f"RMW = {r_km[i]:.0f} km", (0.97, 0.06),
                     xycoords="axes fraction", ha="right", va="bottom",
                     fontsize=S.FS_ANNOT, color="white", weight="bold")
    for a, cf, name, sci in zip(axes, (cf0, cf1, cf2),
                                (r"$\zeta$  [s$^{-1}$]", r"$V_t$  [m s$^{-1}$]",
                                 r"$\theta - \theta_{\rm rim}$  [K]"),
                                (True, False, False)):
        a.set_ylim(1000, _id.P_TOP_HPA)
        a.set_xlim(0, 800)
        cb = a.figure.colorbar(cf, ax=a, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=S.FS_TICK - 3)
        if sci:      # zeta is ~1e-4; plain ticks are wide enough to collide
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
        a.set_title(f"{label}: {name}", fontsize=S.FS_TICK + 0.5)
    axes[0].set_ylabel("pressure  [hPa]")
    return r_km, vt


def _asym_series(ic: str, init: str, level_hpa: int = 850):
    """(hours, azimuthal variability of control T) along a run's control trajectory."""
    run = D.run("heating_moist", f"tseries_5K_120h_init{init}", ic=ic)
    k = layout.pressure_levels().index(level_hpa)
    ti = layout.upper_index("t")
    hours, vals = [], []
    for lead, _d, c in _continuous_pairs(run):
        up, sfc = io.load_delta_bundle(c)
        lats, lons = sfc[:, 0, -1], sfc[0, :, -2]
        hours.append(lead)
        vals.append(ax.asymmetry(up[k, :, :, ti], lats, lons))
    return np.array(hours), np.array(vals)


def plot(init: str = D.STRONG, *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT, style: str = "note"):
    S.apply()
    ideal, real = _ic_states(init)

    fig = plt.figure(figsize=(16.5, 13.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85], hspace=0.34, wspace=0.42)
    top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    mid = [fig.add_subplot(gs[1, j]) for j in range(3)]
    _draw_row(top, ideal, "idealized (azimuthally averaged)")
    _draw_row(mid, real, "real RAGASA analysis")
    for a in mid:
        a.set_xlabel("radius  [km]")

    ax3 = fig.add_subplot(gs[2, :])
    for tag, colour, lab in (("axisym", S.C_WEAK, "axisymmetric IC"),
                             ("ragasa", S.C_STRONG, "real RAGASA IC")):
        try:
            h, v = _asym_series(tag, init)
        except FileNotFoundError as e:
            print(f"[B3] no {tag} 5-day run yet ({e}); panel (c) will be partial")
            continue
        ax3.plot(h, v, color=colour, lw=2.4, marker="o", ms=3.5, label=lab)
    S.forcing_span(ax3, 24)
    ax3.set_xlabel("Iteration n  (hour)")
    ax3.set_ylabel("azimuthal variability of T at 850 hPa\n(σ$_θ$ / σ$_{\\rm total}$)")
    ax3.set_ylim(bottom=0)
    ax3.set_title("how long the idealization lasts — the real lower boundary "
                  "re-imprints asymmetry from step one")
    ax3.legend(loc="best")

    fig.suptitle(f"B3 — the initial condition: averaging kept the storm, and the "
                 f"symmetry has a shelf life  ({D.CASE.get(init, init)})")
    S.caption(fig, "only the atmosphere was averaged; SST, Coriolis, terrain and the "
                   "land mask are the real fields by design, so panel (c) is the honest "
                   "window inside which these runs are an 'idealized vortex' at all",
              style)
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/b3_ic_check.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "b3_ic_check.png"
    S.save(plot(a.init, ic=a.ic, stat=a.stat, style=a.style), out,
           style=a.style, pdf=not a.no_pdf)
