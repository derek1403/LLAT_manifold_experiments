"""MP5 — the advection hypothesis, made quantitative: secondary circulation and a ΔPV budget.

MP3 showed the isentropic prediction $Dq/Dt \\approx q\\,\\partial\\dot\\theta/\\partial\\theta$
works at n = 1 and then stops describing the field. That equation is *Lagrangian*;
MP2 plots an *Eulerian* section. The obvious reconciliation is that the heating grows a
secondary circulation which then carries the generated PV somewhere else. This figure
tests that directly, in two parts.

**Part 1 (``--part circulation``) — is there a secondary circulation at all?**
Radius–pressure sections of the *perturbation* vertical velocity δω and radial wind
δu_r, i.e. the circulation the heating itself created (control is subtracted, so this
is not the storm's own overturning). Expected if the hypothesis holds: δω < 0 (ascent,
plotted so that red = ascent) in a deep core column, and δu_r < 0 (inflow) low down
with a strong positive (outflow) maximum aloft.

**Part 2 (``--part budget``) — does advection quantitatively account for ΔPV?**
Each term of the linearised ΔPV equation, accumulated over the iterations so that
every row is in PVU and directly comparable with the measured ΔPV:

    row 1   ∫ q̄ ∂θ̇/∂θ dt          diabatic generation (MP3's term)
    row 2   ∫ −δω ∂q̄/∂p dt        vertical advection of background PV by the induced ω
    row 3   ∫ −δu_r ∂q̄/∂r dt      radial advection of background PV by the induced u_r
    row 4   sum of rows 1–3        what the hypothesis predicts
    row 5   measured ΔPV           what the model actually did

All five rows share one colour scale. Row 4 against row 5 is the whole test; the gap
between them is what these three terms do not explain.

**What this budget is and is not.** It is the *linearised* budget: the perturbation
velocity acting on the **background** PV gradient, plus diabatic generation. It leaves
out the background flow advecting the perturbation PV (−v̄·∇δq) and the perturbation
advecting itself (−δv·∇δq). For azimuthally averaged fields the mean *tangential*
advection drops out identically (∂/∂λ = 0), which is why an axisymmetric budget is
worth doing at all — but the radial and vertical parts of those neglected terms are
real, and they are why rows 4 and 5 are not expected to match once δq is no longer
small. Everything is computed from azimuthal means, so eddy correlations
(⟨δω ∂q/∂p⟩ ≠ ⟨δω⟩⟨∂q/∂p⟩) are dropped too.

θ̇ is the injected heating, known analytically — see :mod:`fig_MP3_pv_tendency`.

**These are snapshot runs.** n counts applications of the model operator at frozen
valid time, not forecast hours. The background ū is computed once and never moves, so
∂q̄/∂p and ∂q̄/∂r are the same at every n — which is what makes an accumulated budget
meaningful at all. Each application is still a 3 h DLAMPty step internally, so the
advection it performs is 3 h worth; that is the only sense in which ``STEP_S`` is a time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt                                       # noqa: E402
import matplotlib.ticker as ticker                                    # noqa: E402
import numpy as np                                                    # noqa: E402

import _data as D                                                     # noqa: E402
import style as S                                                     # noqa: E402
from fig_MP1_axisym_pv import _title as _member_title                 # noqa: E402
import _snap as SNAP                                                  # noqa: E402
from llat_manifold import layout                                      # noqa: E402
from llat_manifold.diagnostics import _idealized as _id               # noqa: E402
from llat_manifold.idealized_vortex import axisymmetric as ax         # noqa: E402
from llat_manifold.perturbations import heating                       # noqa: E402

_MEMBERS = (D.STRONG, D.SEV)
FIGS = D.FIGS_ROOT / "myplot"

R_MAX_KM = SNAP.R_MAX_KM
_ITERS = (1, 2, 4, 8, 16)            # columns; the budget accumulates every iteration
# The operator is a 3 h DLAMPty step even though the valid time is frozen, so the
# advection one application performs is 3 h worth. That is what makes the tendency
# terms integrable into PVU. It does NOT make n a forecast clock.
STEP_S = 3 * 3600.0
FORCING_STEPS = SNAP.FORCING_ITERS
FORCING_END = SNAP.FORCING_ITERS
SIGMA = 5.0
HEAT_TYPE = "Deep"
KAPPA = _id.RD / _id.CP


def _azim(field3d, r2d, edges):
    return np.stack([ax.radial_mean(field3d[k], r2d, edges)
                     for k in range(field3d.shape[0])])


def _radial_wind(u, v, theta):
    """u_r = u cosθ + v sinθ, the outward component (positive = outflow)."""
    return u * np.cos(theta) + v * np.sin(theta)


def _step_fields(init, amp, n, run=None):
    """Azimuthal-mean pieces of the budget at one iteration.

    In snapshot mode the background ū is frozen, so the reference PV, its gradients and
    the radial frame are the *same* at every n — computed once by :mod:`_snap` and
    reused. That is what makes this budget well posed: in the continuous runs the
    "background" was a marching trajectory on a moving domain, and differentiating it
    mixed the vortex's response with the domain sliding underneath.

    Returns ``(r_km, p, dpv, dw, dur, gen, vadv, radv)`` — ΔPV in PVU, the induced
    circulation in Pa/s and m/s, and the three tendencies in PVU/s.
    """
    run = run if run is not None else SNAP.run_dir(amp, init)
    b_up, b_sfc = SNAP._background(run)
    d_up, d_sfc = SNAP.delta(run, n)
    # δ is the forcing response and nothing else: with the base pinned to u₀ the
    # iteration has no forcing-independent drift, so the circulation read off δ is the
    # circulation the heating induced. (Before 2026-08-04 it was not, and the amp = 0
    # twin had to be subtracted here — see _snap.null_run.)
    p = SNAP.pressure()
    p_col = p[:, None, None]

    r2d, theta, edges, centers = SNAP._frame(run)

    cu, cv, ct, _z, _w, cf, clat, clon, ph = _id.fields_from_bundle(b_up, b_sfc)
    q_ctrl = _id.calculate_pv_spherical(cu, cv, ct, cf, ph, clat, clon)

    qbar = _azim(q_ctrl, r2d, edges)                       # PVU (r, p)
    dpv = _azim(SNAP.dpv_field(run, n), r2d, edges)        # PVU

    ui, vi, wi = (layout.upper_index(k) for k in ("u", "v", "w"))
    dw = _azim(d_up[:, :, :, wi], r2d, edges)              # Pa/s
    dur = _azim(_radial_wind(d_up[:, :, :, ui], d_up[:, :, :, vi], theta),
                r2d, edges)                                # m/s

    # Gradients of the background PV on the azimuthal-mean grid.
    dq_dp = np.gradient(qbar, p * 100.0, axis=0)           # PVU/Pa
    dq_dr = np.gradient(qbar, centers * 1000.0, axis=1)    # PVU/m

    vadv = -dw * dq_dp                                     # PVU/s
    radv = -dur * dq_dr                                    # PVU/s

    # Diabatic generation, from the injection itself (zero once forcing has stopped).
    if n <= FORCING_END:
        V = heating.vertical_profile(HEAT_TYPE, ph)[: ct.shape[0]]
        G = heating.gaussian_centered(b_sfc.shape[0], SIGMA)
        dtheta_step = (amp / FORCING_STEPS) * V[:, None, None] * G[None] \
            * (1000.0 / p_col) ** KAPPA
        theta_bg = ct * (1000.0 / p_col) ** KAPPA
        dth_dp = np.gradient(theta_bg, p * 100.0, axis=0)
        dinj_dp = np.gradient(dtheta_step, p * 100.0, axis=0)
        ratio = np.divide(dinj_dp, dth_dp, out=np.zeros_like(dinj_dp),
                          where=np.abs(dth_dp) > 1e-12)
        gen = _azim(q_ctrl * ratio, r2d, edges) / STEP_S   # PVU/s
    else:
        gen = np.zeros_like(dpv)
    return centers, p, dpv, dw, dur, gen, vadv, radv


def collect(init, amp, ns):
    """{n: dict of fields}; tendencies are accumulated over every iteration."""
    need = sorted(set(ns) | set(range(1, max(ns) + 1)))
    run = SNAP.run_dir(amp, init)
    acc = None
    out = {}
    for h in need:
        r_km, p, dpv, dw, dur, gen, vadv, radv = _step_fields(init, amp, h, run)
        if acc is None:
            acc = {k: np.zeros_like(dpv) for k in ("gen", "vadv", "radv")}
        for k, term in (("gen", gen), ("vadv", vadv), ("radv", radv)):
            acc[k] = acc[k] + term * STEP_S                # PVU
        if h in ns:
            out[h] = dict(r_km=r_km, p=p, dpv=dpv, dw=dw, dur=dur,
                          gen=acc["gen"].copy(), vadv=acc["vadv"].copy(),
                          radv=acc["radv"].copy())
            out[h]["sum"] = out[h]["gen"] + out[h]["vadv"] + out[h]["radv"]
    return out


_CIRC_ROWS = [("dw", "induced $\\delta\\omega$\n(Pa s$^{-1}$, red = ascent)", -1.0),
              ("dur", "induced $\\delta u_r$\n(m s$^{-1}$, red = outflow)", 1.0)]
_BUDGET_ROWS = [
    ("gen", "diabatic generation\n$\\int q\\,\\partial\\dot\\theta/\\partial\\theta\\,dt$"),
    ("vadv", "vertical advection\n$\\int -\\delta\\omega\\,\\partial\\bar q/\\partial p\\,dt$"),
    ("radv", "radial advection\n$\\int -\\delta u_r\\,\\partial\\bar q/\\partial r\\,dt$"),
    ("sum", "sum of the three\n(the hypothesis)"),
    ("dpv", "measured $\\Delta$PV\n(what the model did)"),
]


def _strip(panels, ns, rows, *, title, cbar_label, scale, style):
    """Shared drawing: rows x iterations of r–p sections.

    ``scale`` picks what shares a colour range: ``"row"`` (units differ per row),
    or ``"column"`` (all rows at one iteration share it, so the budget rows stay comparable
    with each other while the range is free to grow with time — a single scale for the
    whole figure would be set by the late, diverging columns and blank out the rest).
    """
    S.apply()
    have = [h for h in ns if h in panels]
    fig, axes = plt.subplots(len(rows), len(have), squeeze=False, sharey=True,
                             figsize=(2.05 * len(have) + 1.6, 3.5 * len(rows) + 1.0))

    def lim_of(keys, sel):
        vals = []
        for key, *_ in keys:
            for h in sel:
                d = panels[h]
                win = np.ix_(d["p"] >= _id.P_TOP_HPA, d["r_km"] <= R_MAX_KM)
                vals.append(np.abs(d[key][win]).ravel())
        return float(np.nanpercentile(np.concatenate(vals), 99.5)) or 1.0

    col_lim = {h: lim_of(rows, [h]) for h in have} if scale == "column" else {}
    col_mesh = {}
    meshes = []
    for i, row in enumerate(rows):
        key, label = row[0], row[1]
        sign = row[2] if len(row) > 2 else 1.0
        if scale == "row":
            levels = np.linspace(-lim_of([row], have), lim_of([row], have), 25)
        for j, h in enumerate(have):
            axx = axes[i, j]
            d = panels[h]
            if scale == "column":
                levels = np.linspace(-col_lim[h], col_lim[h], 25)
            cf = axx.contourf(d["r_km"], d["p"], sign * d[key], levels=levels,
                              cmap=S.CMAP_PV, extend="both")
            axx.contour(d["r_km"], d["p"], sign * d[key], levels=[0], colors="k",
                        linewidths=0.5)
            col_mesh[h] = cf
            axx.set_ylim(1000, _id.P_TOP_HPA)
            axx.set_xlim(0, R_MAX_KM)
            axx.tick_params(labelsize=S.FS_TICK - 2)
            if i == 0:
                forced = " (forcing on)" if SNAP.forcing_on(h) else ""
                axx.set_title(f"{SNAP.label(h)}{forced}", fontsize=S.FS_TICK,
                              color=(S.C_SENS if h <= FORCING_END else "k"))
            if i == len(rows) - 1:
                axx.set_xlabel("radius  (km)", fontsize=S.FS_TICK - 1)
        axes[i, 0].set_ylabel(f"{label}\n\npressure  (hPa)", fontsize=S.FS_TICK - 1)
        meshes.append(cf)
        if scale == "row":
            cb = fig.colorbar(cf, ax=axes[i, :].tolist(), pad=0.012, fraction=0.02)
            cb.ax.tick_params(labelsize=S.FS_TICK - 3)
    if scale == "column":
        # One bar under each column, built from *that column's* mappable — every row
        # in a column shares its levels, so any of them carries the right range.
        for j, h in enumerate(have):
            cb = fig.colorbar(col_mesh[h], ax=axes[:, j].tolist(),
                              orientation="horizontal", pad=0.06, fraction=0.03,
                              aspect=16)
            cb.locator = ticker.MaxNLocator(nbins=4)
            cb.update_ticks()
            cb.ax.tick_params(labelsize=S.FS_TICK - 3)
            if j == 0:
                cb.set_label(cbar_label, fontsize=S.FS_TICK - 2, weight="bold")
    fig.suptitle(title, wrap=True)
    return fig, axes


def plot_circulation(init, amp, panels, ns, *, style="note"):
    fig, _axes = _strip(
        panels, ns, _CIRC_ROWS, scale="row", cbar_label="",
        title=f"MP5 — the secondary circulation the {amp} K heating created  "
              f"({_member_title(init).replace(chr(10), '  ')})", style=style)
    S.caption(fig, "perturbation fields (control subtracted), azimuthally averaged, so "
                   "this is the circulation the heating added and not the storm's own "
                   "overturning; δω is plotted sign-flipped so red = ascent, matching "
                   "the usual reading of a secondary circulation; each row has its own "
                   "colour scale because Pa/s and m/s are not comparable; columns up to "
                   "n <= 8 still have forcing going in; n counts operator iterations, "
                   "not forecast hours", style, wrap=True)
    return fig


def plot_budget(init, amp, panels, ns, *, style="note"):
    fig, _axes = _strip(
        panels, ns, _BUDGET_ROWS, scale="column",
        cbar_label="azimuthal-mean ΔPV  (PVU)",
        title=f"MP5 — ΔPV budget at {amp} K: does the induced circulation account for "
              f"it?  ({_member_title(init).replace(chr(10), '  ')})", style=style)
    S.caption(fig, "every row is accumulated up to the column's iteration, so all five are "
                   "in PVU; each COLUMN has its own colour range because the linearised "
                   "budget grows by two orders of magnitude across the strip, but all "
                   "five rows within a column share it, which is what the comparison "
                   "needs; row 4 = rows 1+2+3 is what the "
                   "advection hypothesis predicts, row 5 is what the model did, and the "
                   "difference is what these terms do not explain; this is the "
                   "*linearised* budget — the perturbation velocity acting on the "
                   "background PV gradient — so it omits −v̄·∇δq and −δv·∇δq, and it is "
                   "built from azimuthal means, so eddy correlations are dropped too",
              style, wrap=True)
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--amp", type=int, default=5)
    ap.add_argument("--members", nargs="+", default=list(_MEMBERS))
    ap.add_argument("--iters", nargs="+", type=int, default=list(_ITERS),
                    help="iterations to draw as columns (default: %(default)s)")
    ap.add_argument("--part", choices=("circulation", "budget", "both"), default="both")
    ap.add_argument("--outdir", default=None, help="default: figs/myplot/")
    a = ap.parse_args()

    ns = tuple(a.iters)
    outdir = Path(a.outdir) if a.outdir else FIGS
    for init in a.members:
        panels = collect(init, a.amp, ns)
        tag = init[4:8]
        win_of = lambda d: np.ix_(d["p"] >= _id.P_TOP_HPA, d["r_km"] <= R_MAX_KM)
        for h in ns:
            d = panels[h]
            w = win_of(d)
            print(f"[MP5] {tag} n={h:3d}  sum {d['sum'][w].min():+.2f}/"
                  f"{d['sum'][w].max():+.2f}  measured {d['dpv'][w].min():+.2f}/"
                  f"{d['dpv'][w].max():+.2f}  "
                  f"corr {np.corrcoef(d['sum'][w].ravel(), d['dpv'][w].ravel())[0, 1]:+.2f}")
        if a.part in ("circulation", "both"):
            S.save(plot_circulation(init, a.amp, panels, ns, style=a.style),
                   outdir / f"mp5-1_{tag}_secondary_circulation.png",
                   style=a.style, pdf=not a.no_pdf)
            plt.close("all")
        if a.part in ("budget", "both"):
            S.save(plot_budget(init, a.amp, panels, ns, style=a.style),
                   outdir / f"mp5-2_{tag}_pv_budget.png",
                   style=a.style, pdf=not a.no_pdf)
            plt.close("all")
