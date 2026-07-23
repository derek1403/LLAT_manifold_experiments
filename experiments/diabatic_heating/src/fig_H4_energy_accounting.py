"""H4 mechanism clue — the manifold's condensation efficiency and the T–q attractor.

Two bar panels over the six 5 K runs (inject ΔT / latent-equiv δq / measured δq ×
strong / weak vortex), all at nominal hour 24:

(a) Core max δT. The δq-only runs end up *warmer* than the directly heated runs
    (+1.83–1.85 K vs +1.19–1.49 K): injected ΔT gets advected and mixed away,
    while injected moisture keeps cashing itself in — sensible heat is one-shot,
    latent heat is a redeemable reservoir.

(b) Core-mean column energy partition (sensible cp·∫δT dm vs latent Lv·∫δq dm),
    with the latent/sensible ratio printed above each pair. Whatever is injected —
    heat or moisture, much or little — the state relaxes to nearly the same ratio
    (1.3–1.8): the manifold has one preferred thermo–moisture covariance axis and
    projects every injection onto it. This is the mechanism seed for the
    self-refilling-reservoir story (companion H4 figures).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

_RUNS = [  # (family, tag template, short label)
    ("heating_moist", "tseries_5K_120h_init{init}", "inject ΔT"),
    ("dq_latent", "tseriesdql_5K_120h_init{init}", "inject δq\n(latent-equiv)"),
    ("dq_measured", "tseriesdq_5K_120h_init{init}", "inject δq\n(measured)"),
]
_BAR_INITS = [(D.STRONG, "strong"), (D.WEAK, "weak")]


def _core_max_dT(run_dir, lead_hr=24, sigma=5.0):
    from llat_manifold.diagnostics.response import _core_mask
    from llat_manifold import io, layout
    d = sorted(Path(run_dir).glob(f"data/delta_continuous_*lead{lead_hr:03d}hr.npz"))[0]
    up, _ = io.load_delta_bundle(d)
    dT = up[..., layout.upper_index("t")]
    core = _core_mask(dT.shape[1], dT.shape[2], sigma)
    return float(np.nanmax(np.where(core[None], dT, np.nan)))


def plot(lead_hr: int = 24, style: str = "note"):
    S.apply()
    labels, dT, es, el = [], [], [], []
    for fam, tag, lab in _RUNS:
        for init, who in _BAR_INITS:
            run = str(D.run(fam, tag.format(init=init)))
            s = D.energy_series(run)
            i24 = s["hour"].index(lead_hr)
            labels.append(f"{lab}\n{who}")
            dT.append(_core_max_dT(run, lead_hr))
            es.append(s["e_sens"][i24])
            el.append(s["e_lat"][i24])

    x = np.arange(len(labels))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.2))

    # Colour by injection type, matching panel (b)'s semantics: heat red, moisture blue.
    #colors = [S.C_SENS if "ΔT" in l else S.C_LAT for l in labels]
    colors = ['red' if "ΔT" in l else 'blue' for l in labels]
    axA.bar(x, dT, color=colors, alpha=0.7, width=0.62)
    for xi, v in zip(x, dT):
        axA.text(xi, v + 0.03, f"+{v:.2f} K", ha="center", fontsize=S.FS_ANNOT + 2.5,
                 weight="bold")
    axA.set_xticks(x, labels, fontsize=S.FS_TICK - 1)
    axA.set_ylabel("core max δT at nominal hour 24  [K]")
    axA.set_title("(the δq-only runs end up warmer than the heated ones)")
    axA.set_ylim(0, max(dT) * 1.2)

    w = 0.36
    axB.bar(x - w / 2, es, width=w, color='red', alpha=0.7, label="sensible  cp·∫δT dm") # color=S.C_SENS
    axB.bar(x + w / 2, el, width=w, color='blue', alpha=0.7, label="latent  Lv·∫δq dm") # color=S.C_LAT
    for xi, (a, b) in enumerate(zip(es, el)):
        axB.text(xi, max(a, b) + 0.08, f"L/S = {b / a:.2f}", ha="center",
                 fontsize=S.FS_ANNOT + 2.5, weight="bold")
    axB.set_xticks(x, labels, fontsize=S.FS_TICK - 1)
    axB.set_ylabel("core-mean column energy  [MJ m$^{-2}$]")
    axB.set_title("(all six relax to nearly the same latent/sensible ratio)")
    axB.legend(loc="upper right")

    S.panel_letters([axA, axB])
    fig.suptitle("H4 mechanism clue — condensation efficiency and the preferred T–q axis "
                 f"(5 K runs, nominal hour {lead_hr})")
    S.caption(fig, "whatever is injected — heat or moisture, much or little — the state relaxes "
                   "to L/S ≈ 1.3–1.8: one preferred thermo–moisture covariance axis; and pure-δq "
                   "runs end up warmer than heated ones (latent heat keeps cashing in)", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", default=str(D.FIGS / "h4_energy_accounting.png"))
    a = ap.parse_args()
    S.save(plot(style=a.style), a.out, style=a.style, pdf=not a.no_pdf)
