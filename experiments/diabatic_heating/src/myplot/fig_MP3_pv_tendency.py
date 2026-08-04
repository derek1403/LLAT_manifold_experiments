"""MP3 — the textbook prediction, computed and laid next to what the model did.

Theory says a Deep heating bump should generate PV below its maximum and destroy it
above. In isentropic coordinates the leading diabatic term of the PV equation is

    Dq/Dt  ≈  q · ∂θ̇/∂θ                                                    (1)

so the sign of the PV tendency is the sign of ∂θ̇/∂θ alone. θ̇ is a *bump* — zero at
1000 hPa, maximum at 600 hPa, zero at 200 hPa — so its derivative is positive below
the maximum and negative above it. That is the whole content of "positive PV low,
negative PV aloft": it is the derivative of a hump, nothing subtler. Physically, in
θ-space: below the maximum the cross-isentropic mass flux converges, the isentropic
layer thins (σ = −g⁻¹∂p/∂θ falls), and q = ζ_a/σ goes up. Above it, the opposite.

**How θ̇ is computed here — the part that is easy to over-think.** It does not have to
be diagnosed from the model at all: this experiment *injects* the heating, so it is
known analytically. ``perturbations/heating.py`` adds, to the temperature channel,

    ΔT(p, r) = (amp_K / forcing_steps) · V(p) · exp(−r²/2σ²)   per step,
    V(p) = sin(π (p−200)/800)  on 200–1000 hPa   ("Deep"), σ = 5 grid points,

for the first ``forcing_steps`` steps. Over the whole 24 h forcing window the injected
total is ``amp_K · V(p) · G(r)`` in temperature, i.e.

    Δθ_inj(p, r) = amp_K · V(p) · G(r) · (1000/p)^κ                        (2)

in potential temperature (κ = R/c_p). Working with the *window total* rather than a
per-second rate keeps the units honest: ∂Δθ_inj/∂θ is then dimensionless and
q · ∂Δθ_inj/∂θ comes out directly in PVU, comparable cell for cell with the measured
ΔPV at n = 24 h. ∂/∂θ is evaluated as (∂/∂p) / (∂θ/∂p) on the model's own levels,
with θ taken from the control state at n = 0.

Columns, left to right: what goes in (Δθ_inj), the sign factor (∂θ̇/∂θ), the multiplier
(absolute PV q), the prediction (their product), and what the model actually produced.
Rows are the two ladder members, as in MP2.

**What (1) leaves out**, and it matters for reading the last two columns against each
other: the full frictionless tendency also carries a term in ∇_θθ̇ × ∂v/∂θ — the
*horizontal* gradient of the heating acting on the vertical shear. The bump here is
narrow (σ = 5 grid points ≈ 139 km) and sits in a strongly sheared vortex, so that
term is not obviously small. (1) is the leading term, not the whole story, and the
prediction column should be read as an order-of-magnitude and a sign map.
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
import _snap as SNAP                                                 # noqa: E402
from llat_manifold.diagnostics import _idealized as _id               # noqa: E402
from llat_manifold.idealized_vortex import axisymmetric as ax         # noqa: E402
from llat_manifold.idealized_vortex import background as bg           # noqa: E402
from llat_manifold.perturbations import heating                       # noqa: E402

_MEMBERS = (D.STRONG, D.SEV)
FIGS = D.FIGS_ROOT / "myplot"

R_MAX_KM = SNAP.R_MAX_KM
LEAD_HR = 1                      # default iteration: the near-linear regime (see _theory)
FORCING_HR = SNAP.FORCING_ITERS  # amp_K is spread over the first 8 iterations
SIGMA = 5.0                      # injection width [grid points], from the run configs
HEAT_TYPE = "Deep"
KAPPA = _id.RD / _id.CP


def _radial(field3d, lat2d, lon2d):
    """Azimuthal mean of a (level, y, x) field on the IC's own radial frame."""
    r2d, _th, edges, centers = ax.radial_frame(lat2d, lon2d)
    return centers, np.stack([ax.radial_mean(field3d[k], r2d, edges)
                              for k in range(field3d.shape[0])])


def _measured_dpv(init, amp, n):
    """Azimuthal-mean ΔPV at one iteration, from the same runs MP2 plots."""
    run = SNAP.run_dir(amp, init)
    return SNAP.azimuthal(SNAP.dpv_field(run, n), run)[1]


def _theory(init, amp, lead_hr=LEAD_HR):
    """(r_km, p_hPa, Δθ_inj, ∂θ̇/∂θ, q, predicted ΔPV, measured ΔPV) for one member.

    ``lead_hr`` is the **iteration** n, and it scales the injected total: the forcing is
    spread evenly over the first 8 iterations, so by iteration n (n ≤ 8) only n/8 of it
    has gone in. Comparing the linear prediction against the end of the forcing window
    is the wrong test — the ±A antisymmetry check (MP4) says the response is only a few
    per cent nonlinear at n = 1 but tens of per cent by n = 8, so the early iterations
    are where a linear prediction is entitled to work.
    """
    # The frozen background ū is what every iteration is measured against, so it is
    # also the state the linear prediction must be built on — not the n = 0 IC.
    b_up, b_sfc = SNAP._background(SNAP.run_dir(amp, init))
    u, v, t, _z, _w, f, lat2d, lon2d, p_hpa = _id.fields_from_bundle(b_up, b_sfc)
    p = np.asarray(p_hpa, dtype=float)
    p_col = p[:, None, None]

    theta = t * (1000.0 / p_col) ** KAPPA                       # K
    q = _id.calculate_pv_spherical(u, v, t, f, p_hpa, lat2d, lon2d)      # PVU

    # (2): the injection accumulated up to this lead, expressed in θ.
    frac = min(lead_hr, FORCING_HR) / FORCING_HR
    V = heating.vertical_profile(HEAT_TYPE, p_hpa)[: t.shape[0]]
    G = heating.gaussian_centered(b_sfc.shape[0], SIGMA)
    dtheta_inj = frac * amp * V[:, None, None] * G[None] * (1000.0 / p_col) ** KAPPA

    # ∂/∂θ = (∂/∂p) / (∂θ/∂p). ∂θ/∂p < 0 in a stable atmosphere, so the ratio flips
    # the sign of ∂/∂p — which is what puts generation *below* the heating maximum.
    dp = p * 100.0                                              # Pa
    dth_dp = np.gradient(theta, dp, axis=0)
    dinj_dp = np.gradient(dtheta_inj, dp, axis=0)
    ratio = np.divide(dinj_dp, dth_dp, out=np.zeros_like(dinj_dp),
                      where=np.abs(dth_dp) > 1e-12)              # dimensionless
    pred = q * ratio                                             # PVU over the window

    r_km, fields = None, []
    for fld in (dtheta_inj, ratio, q, pred):
        r_km, az = _radial(fld, lat2d, lon2d)
        fields.append(az)
    return (r_km, p, *fields, _measured_dpv(init, amp, lead_hr))


# (label, index into the tuple above, colormap, symmetric?) per column.
_COLS = [
    (r"injected $\Delta\theta$ so far  (K)", 2, S.CMAP_PV, True),
    (r"$\partial\dot\theta/\partial\theta$  (per injection so far)", 3, S.CMAP_PV, True),
    ("absolute PV  (PVU)", 4, S.CMAP_PV_ABS, False),
    (r"predicted $\Delta$PV = $q\,\partial\dot\theta/\partial\theta$  (PVU)", 5, S.CMAP_PV, True),
    (r"measured $\Delta$PV  (PVU)", 6, S.CMAP_PV, True),
]


def plot(amp, secs, lead_hr, *, style: str = "note"):
    S.apply()
    n_rows, n_cols = len(secs), len(_COLS)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharey=True,
                             figsize=(3.0 * n_cols + 1.0, 3.6 * n_rows + 1.4))

    for j, (label, idx, cmap, sym) in enumerate(_COLS):
        # One colour scale per column so the two rows stay comparable; the 99.5th
        # percentile inside the plotted window, as everywhere else in this set.
        vals = []
        for _init, sec in secs:
            r_km, p = sec[0], sec[1]
            win = np.ix_(p >= _id.P_TOP_HPA, r_km <= R_MAX_KM)
            vals.append(np.abs(sec[idx][win]).ravel())
        lim = float(np.nanpercentile(np.concatenate(vals), 99.5)) or 1.0
        levels = np.linspace(-lim, lim, 25) if sym else np.linspace(0, lim, 13)

        for i, (init, sec) in enumerate(secs):
            axx = axes[i, j]
            cf = axx.contourf(sec[0], sec[1], sec[idx], levels=levels, cmap=cmap,
                              extend="both" if sym else "max")
            if sym:
                axx.contour(sec[0], sec[1], sec[idx], levels=[0], colors="k",
                            linewidths=0.6)
            axx.axhline(600, color=S.C_GUIDE, lw=1.0, ls="--", zorder=3)
            axx.set_ylim(1000, _id.P_TOP_HPA)
            axx.set_xlim(0, R_MAX_KM)
            axx.tick_params(labelsize=S.FS_TICK - 2)
            if i == 0:
                axx.set_title(label, fontsize=S.FS_TICK - 0.5)
            if i == n_rows - 1:
                axx.set_xlabel("radius  (km)", fontsize=S.FS_TICK - 1)
        cb = fig.colorbar(cf, ax=axes[:, j].tolist(), orientation="horizontal",
                          pad=0.10, fraction=0.045, aspect=22)
        # One tick per contour level is unreadable on a bar this narrow. nbins counts
        # intervals, so 4 gives 5 labels — 5 would collide at the extend arrows.
        cb.locator = ticker.MaxNLocator(nbins=4)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=S.FS_TICK - 3)

    for i, (init, _sec) in enumerate(secs):
        axes[i, 0].set_ylabel(f"{_member_title(init).replace(' (', chr(10) + '(')}"
                              f"\n\npressure  (hPa)", fontsize=S.FS_TICK)

    fig.suptitle(f"MP3 — the isentropic-coordinate prediction $Dq/Dt \\approx q\\,"
                 f"\\partial\\dot\\theta/\\partial\\theta$ at {amp:g} K, "
                 f"against what the model produced at n = {lead_hr}", wrap=True)
    # No font on this box carries the combining dot, so "θ̇" renders as a bare θ with
    # a gap. Spelled out in the caption; the panel titles use mathtext, which is fine.
    S.caption(fig, "n counts iterations of the model operator, not forecast hours. Dashed "
                   "line = 600 hPa, the Deep heating maximum: the heating rate "
                   "rises below it and falls above it, so its θ-derivative — and with "
                   "it the predicted PV tendency — changes sign there; the heating rate "
                   "is not diagnosed but taken from "
                   "the injection itself, accumulated up to this lead, so column 4 is in "
                   "PVU and directly comparable with column 5; the ±A antisymmetry "
                   "check says the response is ~5 % nonlinear at n = 3 h but 30-50 % by "
                   "n = 24 h, so an early lead is where a linear prediction is entitled "
                   "to work; the "
                   "neglected (horizontal heating gradient × vertical shear) term is "
                   "not obviously small for a bump this narrow in a vortex this "
                   "sheared, so read column 4 as a sign map and an order of magnitude",
              style, wrap=True)
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--amps", nargs="+", type=int, default=[5],
                    help="injected amplitudes in K (default: %(default)s). The "
                         "prediction scales linearly with amplitude, so one is enough "
                         "to read the structure.")
    ap.add_argument("--members", nargs="+", default=list(_MEMBERS))
    ap.add_argument("--lead", type=int, default=LEAD_HR,
                    help="iteration to compare against (default: %(default)s — the "
                         "near-linear regime; 8 is the end of the forcing window)")
    ap.add_argument("--outdir", default=None, help="default: figs/myplot/")
    a = ap.parse_args()

    outdir = Path(a.outdir) if a.outdir else FIGS
    for amp in a.amps:
        secs = []
        for init in a.members:
            try:
                secs.append((init, _theory(init, amp, a.lead)))
            except FileNotFoundError as e:
                print(f"[MP3] {amp} K: skipping {init} ({e})")
        if not secs:
            raise SystemExit(f"[MP3] no data for {amp} K")
        for init, sec in secs:
            r_km, p = sec[0], sec[1]
            win = np.ix_(p >= _id.P_TOP_HPA, r_km <= R_MAX_KM)
            print(f"[MP3] {amp:2d} K {init}: predicted ΔPV "
                  f"[{sec[5][win].min():+.2f}, {sec[5][win].max():+.2f}] PVU, "
                  f"measured [{sec[6][win].min():+.2f}, {sec[6][win].max():+.2f}] PVU")
        S.save(plot(amp, secs, a.lead, style=a.style),
               outdir / f"mp3-{'m' if amp < 0 else ''}{abs(amp):02.0f}_n{a.lead:03d}_pv_tendency.png",
               style=a.style, pdf=not a.no_pdf)
        plt.close("all")
