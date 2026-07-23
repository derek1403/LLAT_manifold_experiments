"""H1 + the self-correction — is the collapse specific to q, or does any lock break it?

Five-day ΔPV series of the strong-vortex heating run (5 K, 24 h injection) under
four treatments: free heating, and heating with δq / δw / δz each locked to the
control. Left panel low-level generation, right upper-level destruction.

This is where the study corrects its own earlier claim. Finding 3 read the q-lock
collapse as "the memory of the heating lives in q". But locking w or z also
collapses the 5-day maintenance — so the memory lives in the whole T–q–w–z loop,
not in q alone. The locks still differ in fingerprint and timing (q dies the
instant forcing stops; w rides the free run to ~45 h then falls; z is damped even
during forcing), and q is the tightest bottleneck (deepest, fastest) — but it is
not the sole carrier. That every lock decays rather than blows up is what the
imbalance diagnostic (companion figure) confirms as physical, not numerical.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import _data as D
import style as S

# (family, tag, label, colour, linestyle)
_TREATMENTS = [
    ("heating_moist", "tseries_5K_120h_init{init}", "free heating", "#111111", "-"),
    ("heating_qlock", "tseriesq_5K_120h_init{init}", "lock δq", S.C_LAT, "-"),
    ("heating_wlock", "tseriesw_5K_120h_init{init}", "lock δw", S.C_STRONG, "--"),
    ("heating_zlock", "tseriesz_5K_120h_init{init}", "lock δz", "#009E73", "-."),
]


def plot(init: str = D.STRONG, style: str = "note"):
    S.apply()
    series = {fam: D.pv_response_series(str(D.run(fam, tag.format(init=init))))
              for fam, tag, *_ in _TREATMENTS}
    t_end = D.forcing_end_hours(str(D.run("heating_moist",
                                          "tseries_5K_120h_init{}".format(init))))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), sharex=True)
    for ax, key, name in ((axL, "lowlevel_max", "low-level max ΔPV (700–1000 hPa)"),
                          (axR, "upperlevel_min", "upper-level min ΔPV (200–500 hPa)")):
        S.zero_line(ax)
        S.forcing_span(ax, t_end, label=(ax is axL))
        for fam, _tag, lab, c, ls in _TREATMENTS:
            s = series[fam]
            ax.plot(s["hour"], s[key], color=c, ls=ls, lw=2.4 if fam == "heating_moist" else 1.9,
                    marker="o", ms=2.5, label=lab)
        ax.set_xlabel("Iteration n  (nominal hour)")
        ax.set_ylabel("ΔPV  [PVU]")
        ax.set_title(name)
        ax.set_xlim(s["hour"][0], s["hour"][-1])
    axL.legend(loc="upper right")
    S.panel_letters([axL, axR])
    fig.suptitle(f"H1 — every channel lock collapses the 5-day maintenance  "
                 f"({D.CASE[init]}, 5 K)")
    S.caption(fig, "q dies the instant forcing stops, w rides to ~45 h then falls, z is damped "
                   "even during forcing — different fingerprints, but all decay: the memory is "
                   "in the whole T–q–w–z loop, q the tightest bottleneck", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--out", default=str(D.FIGS / "h1_locks_tseries.png"))
    a = ap.parse_args()
    S.save(plot(a.init, a.style), a.out, style=a.style, pdf=not a.no_pdf)
