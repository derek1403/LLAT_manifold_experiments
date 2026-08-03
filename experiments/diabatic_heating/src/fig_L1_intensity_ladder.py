"""L1 — the intensity ladder: same heating, same environment, four vortices.

The heating experiments were first run on two RAGASA stages (weak 0917, strong 0920),
but 0920 is only a 25 m/s storm — the mature stage was still to come. This adds the
0921 (severe, 34 m/s) and 0922 (very severe, 42 m/s) vortices, azimuthally averaged
and dropped into 0920's *fixed* environment (SST / f / radiation / lat-lon / clock),
so the only thing that changes across 0920 → 0921 → 0922 is the vortex itself. 0917
is shown for range but sits in its own (weak) environment, so it is context, not part
of the controlled ladder.

  (a) the ladder itself: azimuthal-mean V_t(r) at 850 hPa, RMW marked — the vortex
      contracts and intensifies monotonically (RMW 284 → 173 → 147 → 120 km);
  (b) the heating→PV response at 5 K vs peak wind: the low-level generation is
      *non-monotonic* — it peaks at the severe stage (0921) and rolls over at the
      very severe stage (0922); the upper-level destruction weakens monotonically;
  (c) the moisture dependence vs peak wind: the δq = 0 survival fraction (qlock/moist)
      and the reverse-probe fraction (dq/moist) — both roughly flat across the ladder,
      i.e. the moisture binding is a property of the mechanism, not of the intensity.

The rollover in (b) carries a caveat the report states plainly: the heating footprint
σ is held fixed (2σ = 278 km) while the RMW contracts to 120 km, so for the very
severe stage the bump increasingly overhangs the vortex — the rollover may be an
intrinsic efficiency turn or a scale mismatch, and this experiment does not separate
them.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _data as D
import style as S


# The ladder in intensity order. 0917 is context (own environment); the controlled
# ladder is 0920 / 0921 / 0922, all in 0920's environment.
_ORDER = [D.WEAK, D.STRONG, D.SEV, D.VSEV]
_CONTROLLED = {D.STRONG, D.SEV, D.VSEV}


def _vmax_profiles():
    """{init: (r_km, V_t, vmax, rmw)} from each member's axisymmetric IC."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(D.REPO / "src"))
    from llat_manifold.idealized_vortex import axisymmetric as ax, background as bg
    out = {}
    for init in _ORDER:
        p = D.axisym_ic_path(init)
        if not Path(p).exists():
            continue
        st = bg.load_background(str(p))
        r, vt = ax.tangential_profile(st.upper, st.surface, 850)
        i = int(np.nanargmax(vt))
        out[init] = (r, vt, float(vt[i]), float(r[i]))
    return out


def _at_amp(pts, amp=5.0):
    """metrics dict at the sweep point closest to ``amp``."""
    if not pts:
        return None
    a, m = min(pts, key=lambda t: abs(t[0] - amp))
    return m


def plot(lead_hr: int = 24, amp: float = 5.0, *, ic: str = "axisym",
         stat: str = D.DEFAULT_STAT, style: str = "note"):
    if ic != "axisym":
        raise SystemExit("L1 is an axisymmetric-ladder figure; use --ic axisym")
    S.apply()
    prof = _vmax_profiles()
    moist = D.sweep(D.FAMILY["heating_moist"][0], lead_hr, ic=ic, stat=stat)
    qlock = D.sweep(D.FAMILY["heating_qlock"][0], lead_hr, ic=ic, stat=stat)
    dqm = D.sweep(D.FAMILY["dq_measured"][0], lead_hr, ic=ic, stat=stat)
    color = S.case_colors(_ORDER)

    inits = [i for i in _ORDER if i in prof and i in moist]
    vmax = {i: prof[i][2] for i in inits}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # (a) the vortex ladder ---------------------------------------------------
    for init in inits:
        r, vt, vm, rmw = prof[init]
        c = color[init]
        axA.plot(r, vt, color=c, lw=2.4, label=f"{D.CASE[init]}  (RMW {rmw:.0f} km)")
        axA.plot(rmw, vm, "o", color=c, ms=6)
    axA.axvspan(0, 278, color=S.C_GUIDE, alpha=0.10, lw=0)
    axA.annotate("heating core\n2σ = 278 km", (278, axA.get_ylim()[1]), xytext=(-4, -6),
                 textcoords="offset points", ha="right", va="top",
                 fontsize=S.FS_ANNOT, color=S.C_NOTE)
    axA.set_xlim(0, 600)
    axA.set_xlabel("radius  [km]")
    axA.set_ylabel("azimuthal-mean $V_t$ at 850 hPa  [m/s]")
    axA.set_title("(a) the intensity ladder")
    axA.legend(loc="upper right", fontsize=S.FS_LEGEND - 0.5)

    # (b) response vs peak wind — the rollover --------------------------------
    lowk, upk = "lowlevel", "upperlevel"
    xs = [vmax[i] for i in inits]
    low = [_at_amp(moist[i], amp)[lowk] for i in inits]
    up = [_at_amp(moist[i], amp)[upk] for i in inits]
    axB.plot(xs, low, "-o", color=S.C_LOW, lw=2.4, ms=7, label="low-level generation")
    axB.plot(xs, up, "-s", color=S.C_UP, lw=2.4, ms=7, label="upper-level destruction")
    for i, x in zip(inits, xs):
        mk = "" if i in _CONTROLLED else "  (own env)"
        axB.annotate(D.CASE[i].split()[0] + mk, (x, _at_amp(moist[i], amp)[lowk]),
                     xytext=(0, 8), textcoords="offset points", ha="center",
                     fontsize=S.FS_ANNOT, color=S.C_NOTE)
    S.zero_line(axB)
    axB.set_xlabel("peak $V_t$  [m/s]")
    axB.set_ylabel(f"ΔPV at {amp:g} K, hour {lead_hr}  [PVU]")
    axB.set_title("(b) response vs intensity — the rollover")
    axB.legend(loc="center left")

    # (c) moisture dependence vs peak wind — controlled ladder only (0917's moist
    # response is ~0.05 PVU, so its ratios are dominated by noise; drop it here).
    cinits = [i for i in inits if i in _CONTROLLED]
    cx = [vmax[i] for i in cinits]
    surv, frac = [], []
    for i in cinits:
        m = _at_amp(moist[i], amp); q = _at_amp(qlock.get(i), amp)
        d = _at_amp(dqm.get(i), amp)
        surv.append(q[lowk] / m[lowk] * 100 if q else np.nan)
        frac.append(d[lowk] / m[lowk] * 100 if d else np.nan)
    axC.plot(cx, surv, "-o", color=S.C_SENS, lw=2.4, ms=7,
             label="δq = 0 survival  (qlock/moist)")
    axC.plot(cx, frac, "-^", color=S.C_LAT, lw=2.4, ms=7,
             label="reverse probe  (dq/moist)")
    axC.axhspan(0, 60, color=S.C_GUIDE, alpha=0.08, lw=0)
    axC.set_ylim(0, 100)
    axC.set_xlim(axB.get_xlim())
    axC.set_xlabel("peak $V_t$  [m/s]")
    axC.set_ylabel("low-level fraction of moist response  [%]")
    axC.set_title("(c) moisture dependence (controlled ladder)")
    axC.legend(loc="upper right")

    S.panel_letters([axA, axB, axC])
    fig.suptitle("L1 — intensity ladder: heating→PV response across four vortices "
                 "in one fixed (0920) environment")
    S.caption(fig, "0920/0921/0922 share 0920's environment (only the vortex differs); "
                   "0917 is context in its own environment. (b) the low-level response "
                   "peaks at the severe stage and rolls over — but σ is fixed while RMW "
                   "contracts to 120 km, so intrinsic efficiency and scale mismatch are "
                   "not separated here", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--amp", type=float, default=5.0)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/l1_intensity_ladder.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "l1_intensity_ladder.png"
    S.save(plot(a.lead, a.amp, ic=a.ic, stat=a.stat, style=a.style),
           out, style=a.style, pdf=not a.no_pdf)
