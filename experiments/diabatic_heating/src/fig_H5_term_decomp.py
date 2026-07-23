"""H5 positive evidence — which term of PV carries the response? (no locks, pure diagnosis)

The T-lock survival figure is negative evidence ("δT not needed"). This is the
positive twin: decompose the *free* runs' ΔPV at the extremum into

    ΔPV ≈ Δζ·∂θ/∂p  (vorticity term)  +  (ζ+f)·Δ(∂θ/∂p)  (stability term)

by counterfactual evaluation — recompute PV with only the wind fields perturbed
(vorticity term) or only the temperature field perturbed (stability term), control
everything else, and read both at the free run's own extremum. Shown as the
percentage of the full ΔPV each term carries (5 K, nominal hour 24).

The punchline pair: the δq-only low-level response is carried ~100 % by the
vorticity term with zero stability contribution, while the *same model's* heating
runs split ~60/40 (the classical dry route) — same pole, same model, different
forcing, different term. Cross-checked against the lock experiments: the more a
pole leans on the vorticity term, the better it survives T-locking and the harder
it dies under wind-locking.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

# (family, tag template, pole key, row label) — order follows finding 8's table.
_CASES = [
    ("dq_measured", "sweepdq_5K_24h_init{i}", D.STRONG, "lowlevel", "δq-only · strong · low-level"),
    ("dq_measured", "sweepdq_5K_24h_init{i}", D.WEAK, "lowlevel", "δq-only · weak · low-level"),
    ("heating_moist", "sweep_5K_24h_init{i}", D.STRONG, "lowlevel", "heating · strong · low-level"),
    ("heating_moist", "sweep_5K_24h_init{i}", D.WEAK, "lowlevel", "heating · weak · low-level"),
    ("dq_measured", "sweepdq_5K_24h_init{i}", D.STRONG, "upperlevel", "δq-only · strong · upper-level"),
    ("dq_measured", "sweepdq_5K_24h_init{i}", D.WEAK, "upperlevel", "δq-only · weak · upper-level"),
]


def _decompose(run_dir, pole, lead_hr=24):
    """(vort %, stab %, full PVU) of the ΔPV at the run's own extremum.

    Each run/pole is normalised by its OWN full ΔPV at its OWN extremum — every
    row of the figure has its own 100 %. The two shares sum to ~100 % up to the
    nonlinear cross term Δζ·Δ(∂θ/∂p) (both fields perturbed together), which is
    a few percent here.
    """
    from llat_manifold.diagnostics.response import analyze_pair, _pv_of
    from llat_manifold import io, layout
    d = sorted(Path(run_dir).glob(f"data/delta_continuous_*lead{lead_hr:03d}hr.npz"))[0]
    c = d.parent / d.name.replace("delta_", "control_")
    m, aux = analyze_pair(d, c, sigma=5.0)
    kji = m[f"{pole}_kji"]
    full = aux["dpv"][kji]

    d_up, d_sfc = io.load_delta_bundle(d)
    c_up, c_sfc = io.load_delta_bundle(c)
    sfc = c_sfc + d_sfc                       # f / lat / lon channels (unchanged by δ)
    ui, vi, ti = (layout.upper_index(n) for n in ("u", "v", "t"))

    pv_ctrl = _pv_of(c_up, sfc)
    up_vort = c_up.copy()                     # only the wind perturbed
    up_vort[..., ui] = c_up[..., ui] + d_up[..., ui]
    up_vort[..., vi] = c_up[..., vi] + d_up[..., vi]
    up_stab = c_up.copy()                     # only the temperature perturbed
    up_stab[..., ti] = c_up[..., ti] + d_up[..., ti]

    vort = (_pv_of(up_vort, sfc) - pv_ctrl)[kji]
    stab = (_pv_of(up_stab, sfc) - pv_ctrl)[kji]
    return 100 * vort / full, 100 * stab / full, full


def plot(lead_hr: int = 24, style: str = "note"):
    S.apply()
    rows = []
    for fam, tag, init, pole, lab in _CASES:
        v, s, full = _decompose(str(D.run(fam, tag.format(i=init))), pole, lead_hr)
        rows.append((f"{lab}\n(100 % = {full:+.2f} PVU)", v, s))
        print(f"[decomp] {lab:34s} vort {v:6.0f} %   stab {s:6.0f} %   full {full:+.3f} PVU")

    y = np.arange(len(rows))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.axvline(0, color=S.C_GUIDE, lw=0.8)
    ax.axvline(100, color="#333333", lw=1.1, ls=":")
    ax.barh(y + h / 2, [r[1] for r in rows], height=h, color="#009E73",
            label="vorticity term  Δζ·∂θ/∂p")
    ax.barh(y - h / 2, [r[2] for r in rows], height=h, color=S.C_SENS,
            label="stability term  (ζ+f)·Δ(∂θ/∂p)")
    for yi, (_l, v, s) in zip(y, rows):
        ax.text(v + (2 if v >= 0 else -2), yi + h / 2, f"{v:.0f} %",
                ha="left" if v >= 0 else "right", va="center",
                fontsize=S.FS_ANNOT + 0.5, weight="bold", color="#00664B")
        ax.text(s + (2 if s >= 0 else -2), yi - h / 2, f"{s:.0f} %",
                ha="left" if s >= 0 else "right", va="center",
                fontsize=S.FS_ANNOT + 0.5, weight="bold", color=S.C_SENS)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=S.FS_TICK)
    ax.set_xlabel("share of the full ΔPV at the extremum  [%]")
    ax.set_xlim(-15, 125)
    ax.set_title(f"H5 positive evidence — ΔPV term decomposition of the free runs  "
                 f"(5 K, nominal hour {lead_hr})")
    ax.legend(loc="lower right")
    ax.text(100, 0.99, " 100 % = carries the whole response ",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=S.FS_ANNOT, style="italic", color="#333333")
    S.caption(fig, "counterfactual evaluation: PV recomputed with only wind (or only "
                   "temperature) perturbed, read at the free extremum; each row is normalised "
                   "by its OWN run's full ΔPV there (denominator on the row label) — the two "
                   "bars sum to ~100 % up to the nonlinear cross term Δζ·Δ(∂θ/∂p), a few % here", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--out", default=str(D.FIGS / "h5_term_decomposition.png"))
    a = ap.parse_args()
    S.save(plot(a.lead, a.style), a.out, style=a.style, pdf=not a.no_pdf)
