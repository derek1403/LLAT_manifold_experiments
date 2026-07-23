"""M1 + M3 — is q→PV a deep-convection integrator with state dependence, or a blind
column-moisture shortcut?

Left (M1, deep-convection integrator): core max ΔPV vertical profile at 5 K when the
same column moisture is injected only in the boundary layer (BL, 850–1000 hPa) or
only the free troposphere (FT, 400–700 hPa), against the full-column profile. The
injection bands are shaded. The response keeps the same shape — a mid-level
(400–500 hPa) PV peak — no matter which layer the moisture enters: the manifold
routes any moisture into one preferred deep-convective structure. Gain is partly
height-dependent (strong FT ~2× BL aloft, an entrainment signature) but the shape
is not.

Right (M3 proxy, state dependence): the same δq placed on the vortex vs on the
quiescent side ~745 km away, low-level max ΔPV vs amplitude. Off-vortex gives only
11 % (strong) / 42 % (weak) — the nonlinear amplifier lives on the vortex, not in
the q channel itself. The global "blind shortcut" is ruled out; a true dry-region
IC (M3 proper) is still needed to close the local version.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S


def _profile(run_dir, sigma=5.0):
    """Core (r ≤ 2σ) max ΔPV per pressure level at lead 24 h."""
    from llat_manifold.diagnostics.response import analyze_pair, _core_mask
    from llat_manifold import layout
    d = sorted(Path(run_dir).glob("data/delta_continuous_*lead024hr.npz"))[0]
    c = d.parent / d.name.replace("delta_", "control_")
    _m, aux = analyze_pair(d, c, sigma=sigma)
    dpv = aux["dpv"]
    core = _core_mask(dpv.shape[1], dpv.shape[2], sigma)
    prof = np.array([np.nanmax(np.where(core, dpv[k], np.nan)) for k in range(dpv.shape[0])])
    return np.asarray(layout.pressure_levels(), dtype=float)[:dpv.shape[0]], prof


def plot(init: str = D.STRONG, style: str = "note"):
    S.apply()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.6))

    # (a) vertical profile, full vs BL vs FT at 5 K.
    layers = [("dq_measured", "sweepdq_5K_24h_init{init}", "full column", "#111111", "-"),
              ("dq_bl", "sweepdqbl_5K_24h_init{init}", "BL only (850–1000)", S.C_LAT, "-"),
              ("dq_ft", "sweepdqft_5K_24h_init{init}", "FT only (400–700)", S.C_STRONG, "--")]
    axL.axhspan(850, 1000, color=S.C_LAT, alpha=0.10, lw=0)
    axL.axhspan(400, 700, color=S.C_STRONG, alpha=0.10, lw=0)
    for fam, tag, lab, c, ls in layers:
        p, prof = _profile(D.run(fam, tag.format(init=init)))
        axL.plot(prof, p, color=c, ls=ls, lw=2.2, marker="o", ms=4, label=lab)
    axL.axvline(0, color=S.C_GUIDE, lw=0.8)
    axL.set_ylim(1000, 100)
    axL.set_yscale("log")
    axL.set_yticks([1000, 850, 700, 500, 400, 300, 200, 100])
    axL.set_yticklabels([1000, 850, 700, 500, 400, 300, 200, 100])
    axL.yaxis.set_minor_formatter(plt.NullFormatter())      # kill "6×10²" minor labels
    axL.set_xlabel("core max ΔPV  [PVU]")
    axL.set_ylabel("pressure  [hPa]")
    axL.set_title(f"M1 — structure is invariant to injection height  ({D.CASE[init]}, 5 K)")
    axL.legend(loc="lower right")

    # (b) state dependence: on-vortex vs off-vortex low-level max ΔPV.
    on = D.sweep(D.FAMILY["dq_measured"][0], 24)
    off = D.sweep(D.FAMILY["dq_offcore"][0], 24)
    color = S.case_colors(set(on) & set(off))
    axR.axhline(0, color=S.C_GUIDE, lw=0.8)
    for init_k in sorted(set(on) & set(off)):
        c = color[init_k]
        a, b = dict(on[init_k]), dict(off[init_k])
        amps = sorted(set(a) & set(b))
        axR.plot(amps, [a[x]["lowlevel_max"] for x in amps], color=c, ls="-", lw=2.4,
                 marker="o", ms=6, label=f"{D.CASE.get(init_k, init_k)} · on vortex")
        axR.plot(amps, [b[x]["lowlevel_max"] for x in amps], color=c, ls="--", lw=1.9,
                 marker="s", ms=6, mfc="none", label=f"{D.CASE.get(init_k, init_k)} · quiescent side")
    axR.set_xlabel("Nominal amplitude amp_K  [K]")
    axR.set_ylabel("low-level max ΔPV  [PVU]")
    axR.set_xlim(left=0)
    axR.set_title("M3 proxy — the amplifier lives on the vortex, not in q")
    axR.legend(loc="upper left")

    S.panel_letters([axL, axR])
    fig.suptitle("M1 + M3 — deep-convection integrator with state dependence, not a blind shortcut")
    S.caption(fig, "left: same mid-level PV peak wherever q is injected (deep-convection "
                   "integrator); right: off-vortex δq gives only 11 % (strong) / 42 % (weak) "
                   "→ the nonlinear amplifier is state-dependent", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--out", default=str(D.FIGS / "m1_m3_layers_offcore.png"))
    a = ap.parse_args()
    S.save(plot(a.init, a.style), a.out, style=a.style, pdf=not a.no_pdf)
