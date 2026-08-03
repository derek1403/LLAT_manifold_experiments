"""H4 — is the injected δq fuel, or the seed of a self-refilling reservoir?

Core-mean column energy partition per iteration: sensible cp·∫δT dm, latent
Lv·∫δq dm, and their total, against the cumulative injected reference. Six 5-day
runs in a 2×3 grid — rows are the strong / weak vortex, columns are what was
injected:

    heating_moist   inject ΔT, q free           injected energy = 1 (baseline)
    dq_latent       inject δq only, latent-equiv injected energy = 1 (equal-energy)
    dq_measured     inject δq only, measured     injected energy ≈ 0.37–0.49

All six panels share one y-axis, so the curves read straight across the row — the
comparison the research note wanted but the old per-panel scaling could not give.
The injected reference tops out at ~11 MJ m⁻² for the equal-energy columns, which
would flatten the data curves; it is drawn but clipped, with its plateau value
annotated at the top of each panel.

Reading: latent decaying while sensible rises = one-shot condensation (δq spent);
latent persisting and intermittently regrowing while the total outlasts a fixed
budget = the self-refilling reservoir — the circulation keeps harvesting new
environmental moisture.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

# The six runs, in grid order (row-major): strong row, then weak row.
_RUNS = [
    ("heating_moist", "tseries_5K_120h_init{init}", "inject ΔT (heating) — q free"),
    ("dq_latent", "tseriesdql_5K_120h_init{init}", "inject δq only — latent-equivalent"),
    ("dq_measured", "tseriesdq_5K_120h_init{init}", "inject δq only — measured (≈0.4× energy)"),
]


def plot(style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    S.apply()
    inits = [D.STRONG, D.WEAK]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9), sharex=True, sharey=True)

    # One shared y-range from the data (not the injected line, which we clip).
    lo, hi = 0.0, 0.0
    series = {}
    for r, init in enumerate(inits):
        for c, (fam, tag, _lab) in enumerate(_RUNS):
            s = D.energy_series(str(D.run(fam, tag.format(init=init), ic=ic)))
            es, el = np.asarray(s["e_sens"]), np.asarray(s["e_lat"])
            tot = es + el
            series[(r, c)] = (s, tot)
            lo = min(lo, es.min(), el.min(), tot.min())
            hi = max(hi, es.max(), el.max(), tot.max())
    pad = 0.4
    ylim = (lo - pad, hi + pad)

    for r, init in enumerate(inits):
        for c, (fam, tag, lab) in enumerate(_RUNS):
            ax = axes[r, c]
            s, tot = series[(r, c)]
            t_end = int(D.forcing_end_hours(str(D.run(fam, tag.format(init=init), ic=ic))))
            S.zero_line(ax)
            S.forcing_span(ax, t_end, label=(r == 0 and c == 0))

            inj_max = max(s["e_inj"])
            ax.plot(s["hour"], s["e_inj"], color=S.C_INJ, lw=1.5, ls=":",
                    label="injected (cumulative)")
            ax.plot(s["hour"], s["e_sens"], color='red', lw=2.0, marker="o", ms=3, # color=S.C_SENS
                    label="sensible  cp·∫δT dm")
            ax.plot(s["hour"], s["e_lat"], color='blue', lw=2.0, marker="s", ms=3, # color=S.C_LAT
                    label="latent  Lv·∫δq dm")
            ax.plot(s["hour"], tot, color=S.C_TOTAL, lw=1.8, ls="--", label="total")

            # Injected plateau sits above the data ceiling: annotate rather than rescale.
            if inj_max > ylim[1]:
                ax.annotate(f"injected → {inj_max:.0f}", xy=(t_end, ylim[1]),
                            xytext=(0, -2), textcoords="offset points",
                            ha="left", va="top", fontsize=S.FS_ANNOT,
                            style="italic", color=S.C_INJ)
            ax.set_ylim(ylim)
            ax.set_xlim(s["hour"][0], s["hour"][-1])
            ax.set_title(f"{lab}\n{D.CASE[init]}", fontsize=S.FS_TICK + 0.5, weight="bold")

    for ax in axes[-1, :]:
        ax.set_xlabel("Iteration n  (hour)")
    for ax in axes[:, 0]:
        ax.set_ylabel("core-mean column energy\n[MJ m$^{-2}$]")
    axes[0, 0].legend(loc="upper right", fontsize=S.FS_LEGEND - 0.5)
    S.panel_letters(axes)
    fig.suptitle("H4 — energy partition of the perturbation: sensible vs latent vs injected")
    S.caption(fig, "latent decaying while sensible rises = one-shot condensation (δq spent); "
                   "latent persisting while the total outlasts the injected budget = the "
                   "self-refilling reservoir regime", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/h4_energy_partition.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "h4_energy_partition.png"
    S.save(plot(a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
