"""H5 appendix — the same term decomposition applied to the LOCKED runs, in PVU.

In the locked runs the percentage view is uninformative by construction: δT ≡ 0
kills the stability term exactly, δu = δv ≡ 0 kills the vorticity term exactly —
whatever survives is 100 % of the other term by definition. The informative
comparison is in ABSOLUTE PVU, each surviving term against the free run's same
term:

* lock T leaves the low-level vorticity term almost exactly at its free value —
  the wind branch does not need the accumulated δT;
* lock u,v GROWS a stability term that is ~0 in the free run — the network
  reroutes the moisture into δT when the wind channel is denied (the amplitude-
  flat 23 % residual of F11, made visible term-by-term).

Main figure: fig_H5_term_decomp.py (free runs, % view). This one goes to the
appendix.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

# (family, tag template, x label) — the three δq-only variants.
_RUNS = [
    ("dq_measured", "sweepdq_5K_24h_init{i}", "free"),
    ("dq_tlock", "sweepdqtl_5K_24h_init{i}", "+ δT = 0"),
    ("dq_uvlock", "sweepdquv_5K_24h_init{i}", "+ δu = δv = 0"),
]
_PANELS = [(D.STRONG, "lowlevel"), (D.WEAK, "lowlevel"),
           (D.STRONG, "upperlevel"), (D.WEAK, "upperlevel")]

C_VORT, C_STAB = "#009E73", S.C_SENS


def _decompose_abs(run_dir, pole, lead_hr=24):
    """(full, vort, stab) ΔPV in PVU at the run's own extremum."""
    from llat_manifold.diagnostics.response import analyze_pair, _pv_of
    from llat_manifold import io, layout
    d = sorted(Path(run_dir).glob(f"data/delta_continuous_*lead{lead_hr:03d}hr.npz"))[0]
    c = d.parent / d.name.replace("delta_", "control_")
    m, aux = analyze_pair(d, c, sigma=5.0)
    kji = m[f"{pole}_kji"]

    d_up, d_sfc = io.load_delta_bundle(d)
    c_up, c_sfc = io.load_delta_bundle(c)
    sfc = c_sfc + d_sfc
    ui, vi, ti = (layout.upper_index(n) for n in ("u", "v", "t"))

    pv_ctrl = _pv_of(c_up, sfc)
    up_vort = c_up.copy()
    up_vort[..., ui] = c_up[..., ui] + d_up[..., ui]
    up_vort[..., vi] = c_up[..., vi] + d_up[..., vi]
    up_stab = c_up.copy()
    up_stab[..., ti] = c_up[..., ti] + d_up[..., ti]
    return (float(aux["dpv"][kji]),
            float((_pv_of(up_vort, sfc) - pv_ctrl)[kji]),
            float((_pv_of(up_stab, sfc) - pv_ctrl)[kji]))


def plot(lead_hr: int = 24, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    # sharey per row: the two inits of one pole sit on the same PVU scale,
    # so the strong/weak columns compare directly.
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharey="row")
    pole_name = dict(D.poles(stat))
    for ax, (init, pole) in zip(axes.ravel(), _PANELS):
        fulls, vorts, stabs = [], [], []
        for fam, tag, _lab in _RUNS:
            f, v, s = _decompose_abs(str(D.run(fam, tag.format(i=init), ic=ic)), pole, lead_hr)
            fulls.append(f); vorts.append(v); stabs.append(s)
            print(f"[decomp-abs] {D.CASE[init]:14s} {pole:10s} {fam:12s} "
                  f"full {f:+.3f}  vort {v:+.3f}  stab {s:+.3f} PVU")
        x = np.arange(len(_RUNS))
        w = 0.32
        S.zero_line(ax)
        ax.bar(x - w / 2, vorts, width=w, color=C_VORT, label="vorticity term  Δζ·∂θ/∂p")
        ax.bar(x + w / 2, stabs, width=w, color=C_STAB, label="stability term  (ζ+f)·Δ(∂θ/∂p)")
        for xi, f in zip(x, fulls):
            ax.hlines(f, xi - 0.46, xi + 0.46, color="#222222", lw=1.6,
                      label="full ΔPV at the extremum" if xi == 0 else None)
        
        ax.set_xticks(x, [lab for _f, _t, lab in _RUNS])
        ax.set_title(f"{D.CASE[init]} · {pole_name[pole]}",
                     fontsize=S.FS_LABEL)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("ΔPV  [PVU]")
    axes[0, 0].legend(loc="upper right")
    S.panel_letters(axes)
    fig.suptitle(f"H5 appendix — term decomposition of the LOCKED runs, absolute PVU  "
                 f"(δq-only, 5 K, hour {lead_hr})")
    S.caption(fig, "in a locked run one term is zero BY CONSTRUCTION (δT≡0 → stability term; "
                   "δu=δv≡0 → vorticity term), so percentages are uninformative there; in PVU "
                   "the surviving term compares against the free run's same term — lock T keeps "
                   "the low-level vorticity term at its free value, lock u,v grows a stability "
                   "term the free run does not have (the F11 intra-step reroute)", style)
    fig.tight_layout()
    return fig



if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/h5_term_decomp_locked.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "h5_term_decomp_locked.png"
    S.save(plot(a.lead, a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
