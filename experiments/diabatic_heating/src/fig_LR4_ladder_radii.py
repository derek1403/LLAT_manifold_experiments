"""LR4 — the Rossby-deformation-radius axis, measured on the intensity ladder itself.

``findings_RossbyDeformation.md`` established the L_R argument on two *real* initial
conditions (0917 weak vs 0920 strong) plus a clean idealized vmax sweep. The seminar
frames its weak/strong contrast on **0920 vs 0921**, both vortices sitting in 0920's
fixed environment. This script puts that contrast on the same axis: it recomputes, for
every ladder member, the vortex-modified Rossby radius

    L_R = N·H / sqrt(xi·eta),   xi = f + 2 V_t / r,   eta = f + zeta

together with the confinement metrics the theory predicts (r50 / r80 of the positive
column sensible-energy anomaly, and the fraction of it retained inside the 2 sigma
heated core). Because the ladder holds SST, f, radiation, land-sea and terrain fixed at
0920's values, N is common to every member by construction and the whole spread in L_R
comes from the inertial stability I = sqrt(xi·eta) — which is exactly the variable the
theory says should matter, and nothing else.

Nothing here is a new physical calculation: L_R comes from
``diagnostics.waves.rossby_radius_summary`` and the confinement metrics from
``diagnostics.response.confinement_radii``, both already used by the LR1-LR3 figures.

Output: a two-panel figure plus a markdown table printed to stdout for pasting into the
seminar notes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _data as D
import style as S

from llat_manifold.diagnostics.response import confinement_radii
from llat_manifold.diagnostics.waves import local_rossby_radius, rossby_radius_summary

_CORE_KM = 278.0          # 2 sigma heating footprint
_LEAD_H = 24              # end of the injection window


def _ic_state(init):
    import sys
    sys.path.insert(0, str(D.REPO / "src"))
    from llat_manifold.idealized_vortex import axisymmetric as ax, background as bg
    p = D.axisym_ic_path(init)
    if not Path(p).exists():
        return None
    st = bg.load_background(str(p))
    r, vt = ax.tangential_profile(st.upper, st.surface, 850)
    i = int(np.nanargmax(vt))
    return st, float(vt[i]), float(r[i])


def collect(inits, ic="axisym"):
    """[{init, vmax, rmw, L_R_*, I_core, N_band, r50, r80, core_frac}] in ladder order."""
    rows = []
    for init in inits:
        got = _ic_state(init)
        if got is None:
            print(f"[lr4] no axisymmetric IC for {init} — skipped")
            continue
        st, vmax, rmw = got
        summ = rossby_radius_summary(st.upper, st.surface, r_core_km=_CORE_KM)
        row = dict(init=init, case=D.CASE[init], vmax=vmax, rmw=rmw,
                   r_km=summ["r_km"], L_R_prof=summ["L_R_km"],
                   L_R_core=summ["L_R_core_km"], L_R_min=summ["L_R_min_km"],
                   I_core=summ["I_core_s1"], N_band=summ["N_band_s1"])
        try:
            rd = D.run("heating_moist", f"tseries_5K_120h_init{init}", ic=ic)
            conf = confinement_radii(rd, _LEAD_H)
            row.update(r50=conf["r50"], r80=conf["r80"],
                       core_frac=conf["core_frac"], e_net=conf["e_net"])
        except (FileNotFoundError, KeyError) as exc:
            print(f"[lr4] no confinement for {init}: {exc}")
        rows.append(row)
    return rows


def markdown_table(rows) -> str:
    head = ("| member | peak $V_t$ [m/s] | RMW [km] | $N$ [s$^{-1}$] | $I_\\mathrm{core}$ "
            "[s$^{-1}$] | $L_R$ core [km] | $L_R$ min [km] | r50 [km] | core frac |")
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    out = [head, sep]
    for r in rows:
        out.append(
            f"| {r['case']} | {r['vmax']:.1f} | {r['rmw']:.0f} | {r['N_band']:.3e} | "
            f"{r['I_core']:.2e} | **{r['L_R_core']:.0f}** | {r['L_R_min']:.0f} | "
            + (f"{r['r50']:.0f} | {r['core_frac']:.3f} |" if "r50" in r else "— | — |"))
    return "\n".join(out)


def plot(rows, *, style: str = "note"):
    S.apply()
    color = S.case_colors([r["init"] for r in rows])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # (a) L_R(r) profiles ------------------------------------------------------
    for r in rows:
        axA.plot(r["r_km"], r["L_R_prof"], color=color[r["init"]], lw=2.4,
                 label=f"{r['case']}  ($L_R^{{core}}$ = {r['L_R_core']:.0f} km)")
        axA.plot(r["rmw"], np.interp(r["rmw"], r["r_km"], r["L_R_prof"]), "o",
                 color=color[r["init"]], ms=6)
    axA.axvspan(0, _CORE_KM, color=S.C_GUIDE, alpha=0.10, lw=0)
    axA.annotate(f"heated core\n2σ = {_CORE_KM:.0f} km", (_CORE_KM, 0.97),
                 xycoords=axA.get_xaxis_transform(), xytext=(-4, 0),
                 textcoords="offset points", ha="right", va="top",
                 fontsize=S.FS_ANNOT, color=S.C_NOTE)
    axA.set_yscale("log")
    axA.set_xlim(0, 600)
    axA.set_xlabel("radius  [km]")
    axA.set_ylabel("$L_R(r) = NH/\\sqrt{\\xi\\eta}$  [km]")
    axA.set_title("(a) adjustment scale across the ladder")
    axA.legend(loc="lower right", fontsize=S.FS_LEGEND - 1)

    # (b) confinement vs L_R ---------------------------------------------------
    have = [r for r in rows if "r50" in r]
    if have:
        x = [r["L_R_core"] for r in have]
        axB.plot(x, [r["r50"] for r in have], "-o", color=S.C_LOW, lw=2.4, ms=8,
                 label="r50 of positive $c_p\\!\\int\\!\\delta T\\,dm$")
        for r in have:
            axB.annotate(r["case"].split()[0], (r["L_R_core"], r["r50"]),
                         xytext=(0, 9), textcoords="offset points", ha="center",
                         fontsize=S.FS_ANNOT, color=S.C_NOTE)
        # Axes anchored at zero on purpose. Autoscaling a 55 km spread over a 310-365 km
        # range turns a ~15 % wobble into a dramatic V, which would misrepresent a null
        # result as a signal. The flatness is the point.
        axB.set_ylim(0, 1.25 * max(r["r50"] for r in have))
        ax2 = axB.twinx()
        ax2.plot(x, [r["core_frac"] for r in have], "--s", color=S.C_UP, lw=2.2, ms=7,
                 label="fraction retained inside 2σ")
        ax2.set_ylabel("core fraction", color=S.C_UP)
        ax2.set_ylim(0, 1)
        ax2.grid(False)
        h1, l1 = axB.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axB.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=S.FS_LEGEND - 1)
    axB.set_xlabel("$L_R$ over the heated core  [km]")
    axB.set_ylabel("r50  [km]", color=S.C_LOW)
    axB.set_title(f"(b) confinement does NOT resolve this $L_R$ range "
                  f"(nominal hour {_LEAD_H})")

    S.panel_letters([axA, axB])
    fig.suptitle("LR4 — Rossby deformation radius on the intensity ladder: "
                 "one environment, only the vortex differs")
    S.caption(fig, "All members share 0920's environment, so N is common by "
                   "construction and the spread in L_R is inertial stability alone. "
                   "L_R core is N·H divided by the mean I over the 2σ disc, not the mean "
                   "of the L_R profile. (b) is a NULL result: across a 1.3x L_R range the "
                   "confinement metrics are flat, so the L_R scaling argument must be made "
                   "on the idealized vmax sweep (9x range), not on this ladder", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--with-0917", action="store_true",
                    help="add the uncontrolled developing-stage member for range")
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/ike/lr4_ladder_radii.png")
    a = ap.parse_args()
    order = ([D.WEAK] if a.with_0917 else []) + [D.STRONG, D.SEV, D.VSEV]
    rows = collect(order, ic=a.ic)
    print("\n" + markdown_table(rows) + "\n")
    out = a.out or (D.FIGS_ROOT / a.ic / "ike" / "lr4_ladder_radii.png")
    S.save(plot(rows, style=a.style), out, style=a.style, pdf=not a.no_pdf)
