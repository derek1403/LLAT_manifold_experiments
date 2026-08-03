"""MP1 — the vortices themselves: azimuthal-mean *absolute* PV at n = 0.

Every figure in the intensity-ladder report shows ΔPV — perturbed minus control,
the *response* to an injected heating. None of them shows what the vortices being
perturbed actually look like. This one does: the absolute Ertel PV of the
axisymmetric initial conditions, before anything is injected.

Radius–pressure sections, one panel per ladder member, on a shared colour scale so
the members are comparable. Shading is PV, with cyan level lines every 1 PVU over it
so the tower's shape can be read off the section and not just its colour; black
contours are isentropes (the warm core); the dashed line is the 850 hPa RMW, so the
wind maximum and the PV core can be read against each other.

Read across the panels: the PV tower does not merely get stronger from 0920 to
0921 — its maximum also *lifts*, from ~500 hPa to ~300–400 hPa, while the RMW
contracts. That is the vertical structure behind the single V_t(r) curve in the
report's L1(a).

Three things about the numbers, all of which belong in any write-up using this figure:

* this is absolute PV (units of PVU, values of several PVU), *not* the ΔPV of the
  response figures (tenths of a PVU) — the two are an order of magnitude apart and
  must never be compared;
* the ladder members 0921/0922 carry 0920's environment by construction, so their
  Coriolis f is 0920's (16.5°N). In the core ζ ~ 7e-4 ≫ f ~ 4e-5 and the tower is
  unaffected, but beyond r ≈ 500 km, where ζ → 0, the PV inherits f's ~10 % offset;
* axisymmetry is exact only at n = 0 — which is exactly what this figure plots, so
  this is the one moment where these sections are the whole field, not a summary.

Sections stop at 200 hPa (``_idealized.P_TOP_HPA``), where the model's levels thin
out and its own error takes over.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This script lives one level below src/, so src/ has to go on the path before the
# shared _data / style modules can be imported the way every other figure does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.patheffects as pe                                   # noqa: E402
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402

import _data as D                                                     # noqa: E402
import style as S                                                     # noqa: E402
from llat_manifold.diagnostics import _idealized as _id               # noqa: E402
from llat_manifold.idealized_vortex import axisymmetric as ax         # noqa: E402
from llat_manifold.idealized_vortex import background as bg           # noqa: E402

# The two members the figure is about. Pass more via --inits and the whole ladder
# comes out in one row — nothing in the drawing code is hard-wired to two panels.
_INITS = (D.STRONG, D.SEV)

PV_LEVELS = np.linspace(0.0, 8.0, 33)      # diagnostics/pv.py's absolute range
PV_TICKS = np.arange(0, 9, 2)
PV_CONTOURS = np.arange(1.0, 8.1, 1.0)     # level lines over the shading, PVU
PV_LABELLED = 2.0                          # label every 2 PVU — every 1 crowds the core
# θ stops at 350 K: the next isentrope up exists only in the top-left corner of the
# section, where its label has nowhere to sit.
THETA_LEVELS = np.arange(290, 351, 4)      # K
# Isentrope labels all sit on this radius (as a fraction of the x limit) instead of
# wherever matplotlib decides to put them, so they line up into a column — and the
# same column in every panel. Move it left/right by changing this one number.
THETA_LABEL_RFRAC = 0.87

# Observed best-track category at each analysis time. Deliberately *not* derived from
# the IC's peak V_t: azimuthal averaging on a 0.25° grid smooths the eyewall, so these
# vortices carry far less wind than the storm did (0921 is a Category 4 storm whose
# axisymmetric IC peaks at 34 m/s). Members with no entry fall back to _data.CASE.
CATEGORY = {D.STRONG: "Category 1", D.SEV: "Category 4"}

FIGS = D.FIGS_ROOT / "myplot"


def _title(init: str) -> str:
    """Panel title: the storm's stage plus the analysis time, e.g. 'Category 4 (0921 00Z)'."""
    stamp = f"{init[4:8]} {init[8:10]}Z"
    return f"{CATEGORY.get(init, D.CASE.get(init, init))} ({stamp})"


def _sections(init: str):
    """(r_km, p_hPa, PV, θ, peak V_t, RMW) for one axisymmetric IC.

    PV comes from the shipped Ertel calculation on the full 3-D state and is then
    azimuthally averaged with :mod:`idealized_vortex.axisymmetric`'s own operator —
    the same one that *built* these ICs, so the section is exact for them and lands
    on the identical radial grid as ``rz_sections``' θ and V_t.
    """
    state = bg.load_background(D.axisym_ic_path(init))
    u, v, t, _z, _w, f, lat2d, lon2d, p_hpa = _id.fields_from_bundle(
        state.upper, state.surface)
    pv = _id.calculate_pv_spherical(u, v, t, f, p_hpa, lat2d, lon2d)

    r2d, _theta2d, edges, centers = ax.radial_frame(lat2d, lon2d)
    pv_rp = np.stack([ax.radial_mean(pv[k], r2d, edges) for k in range(pv.shape[0])])

    r_km, p, _zeta, _vt, th = ax.rz_sections(state.upper, state.surface)
    assert np.allclose(r_km, centers), "PV and θ landed on different radial grids"

    r850, vt850 = ax.tangential_profile(state.upper, state.surface, 850)
    i = int(np.nanargmax(vt850))
    return r_km, p, pv_rp, th, float(vt850[i]), float(r850[i])


def _theta_label_points(r_km, p, th, x_km):
    """One (x, y) per isentrope, all on the radius ``x_km`` — for ``clabel(manual=…)``.

    ``clabel`` labels the contour nearest each point it is handed, so pinning the
    labels to a column is just a matter of finding, for every level, the pressure at
    which that isentrope crosses this radius: read θ(p) off the section at ``x_km``
    and invert it. Levels whose crossing falls outside the plotted pressure range are
    dropped rather than dragged to the frame edge.
    """
    prof = np.array([np.interp(x_km, r_km, th[k]) for k in range(th.shape[0])])
    order = np.argsort(prof)                        # np.interp needs ascending x
    p_sorted = np.asarray(p, dtype=float)[order]
    pts = []
    for lev in THETA_LEVELS:
        if not prof.min() <= lev <= prof.max():
            continue
        y = float(np.interp(lev, prof[order], p_sorted))
        if _id.P_TOP_HPA <= y <= 1000.0:
            pts.append((x_km, y))
    return pts


def _draw(ax_, sec, *, rmax: float, label: str, letter: str):
    r_km, p, pv_rp, th, _vmax, rmw = sec
    cf = ax_.contourf(r_km, p, pv_rp, levels=PV_LEVELS, cmap=S.CMAP_PV_ABS,
                      extend="max")
    cs = ax_.contour(r_km, p, th, levels=THETA_LEVELS, colors=S.C_THETA,
                     linewidths=1.0, alpha=0.85)
    ax_.clabel(cs, manual=_theta_label_points(r_km, p, th, THETA_LABEL_RFRAC * rmax)[::2],
               inline=True, fontsize=S.FS_ANNOT+2, fmt="%1.0f K")


    # Level lines of the shaded field itself. The white stroke is what keeps them
    # legible over both the pale outer field and the dark core (pv.py's trick).
    cp = ax_.contour(r_km, p, pv_rp, levels=PV_CONTOURS, colors=S.C_PV_LINE,
                     linewidths=1.6)
    #cp.set(path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax_.clabel(cp, levels=[l for l in cp.levels if l % PV_LABELLED == 0],
               inline=True, fontsize=S.FS_ANNOT+2, fmt="%1.0f")

    ax_.axvline(rmw, color=S.C_GUIDE, lw=1.2, ls="--", zorder=3)

    ax_.set_ylim(1000, _id.P_TOP_HPA)
    ax_.set_xlim(0, rmax)
    ax_.set_xlabel("radius  (km)")
    ax_.set_title(label)
    ax_.set_title(f"({letter})", loc="left")     # panel letter beside the title
    return cf


def plot(inits=_INITS, *, rmax: float = 800.0, style: str = "note"):
    S.apply()
    secs = []
    for init in inits:
        path = D.axisym_ic_path(init)
        if not Path(path).exists():
            print(f"[MP1] skipping {init}: no axisymmetric IC at {path}")
            continue
        sec = _sections(init)
        secs.append((init, sec))
        r_km, p, pv_rp, _th, vmax, rmw = sec
        win = np.asarray(p) >= _id.P_TOP_HPA
        sub = pv_rp[win][:, r_km <= rmax]
        k, _j = np.unravel_index(int(np.nanargmax(sub)), sub.shape)
        print(f"[MP1] {init} ({D.CASE.get(init, init)}): peak V_t = {vmax:.1f} m/s, "
              f"RMW = {rmw:.0f} km, max PV = {np.nanmax(sub):.2f} PVU "
              f"@ {np.asarray(p)[win][k]:.0f} hPa")
    if not secs:
        raise FileNotFoundError("no axisymmetric ICs found for the requested inits")

    fig, axes = plt.subplots(1, len(secs), figsize=(6.1 * len(secs) + 1.5, 5.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for k, (ax_, (init, sec)) in enumerate(zip(axes, secs)):
        cf = _draw(ax_, sec, rmax=rmax, label=_title(init), letter=chr(97 + k))
    axes[0].set_ylabel("pressure  (hPa)")

    cb = fig.colorbar(cf, ax=axes.tolist(), pad=0.015, fraction=0.024)
    cb.set_ticks(PV_TICKS)
    cb.set_label("azimuthal-mean PV  (PVU)", fontsize=S.FS_LABEL, weight="bold")

    fig.suptitle("MP1 — the vortices themselves: azimuthal-mean absolute PV of the "
                 "axisymmetric ICs (n = 0)")
    S.caption(fig, "shading = absolute Ertel PV (NOT the ΔPV of the response figures); "
                   "cyan = PV every 1 PVU, black = isentropes, dashed = 850 hPa "
                   "RMW; both members sit in 0920's environment, so f is 0920's and the "
                   "outer PV (beyond r ~ 500 km, where ζ → 0) carries that ~10 % offset; "
                   "sections stop at 200 hPa", style)
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--inits", nargs="+", default=list(_INITS),
                    help="ladder members to draw, left to right (default: %(default)s)")
    ap.add_argument("--rmax", type=float, default=800.0, help="radius limit [km]")
    ap.add_argument("--out", default=None,
                    help="default: figs/myplot/mp1_axisym_pv.png")
    a = ap.parse_args()
    out = a.out or FIGS / "mp1_axisym_pv.png"
    S.save(plot(tuple(a.inits), rmax=a.rmax, style=a.style), out,
           style=a.style, pdf=not a.no_pdf)
