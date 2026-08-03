"""B4 (basics) — the method, as a picture: two trajectories, a difference, a constraint.

This is the piece that is hardest to pick up from the results figures, because by
the time a curve is drawn the whole apparatus has been compressed into "ΔPV". No
model output is used here — it is a diagram.

(a) **Two trajectories.** Each step advances a control state and a perturbed state
    through the *same* model operator M (one 3 h DLAMPty step). The forcing f is
    added to the perturbed branch only, for the first 8 steps:

        control     I_n   = M(I_{n-1})
        perturbed   A_n   = M(I_{n-1} + δ_{n-1} + f_n)
        difference  δ_n   = A_n − I_n

    Everything the study reports is δ, the gap between the branches. Taking the
    difference each step is what removes the model's own drift, the diurnal cycle,
    the storm's own motion: whatever both branches do together cancels.

(b) **A constraint.** A "δq = 0" run is the same loop with one extra line: after
    each step, the moisture component of δ is set back to zero, so the perturbed
    branch re-enters the next step carrying the control's moisture. Nothing is
    removed from the physics — the model still does whatever it does with moisture;
    what is removed is the perturbation's *freedom to differ* in that one channel.
    Whatever response survives is the part that did not need moisture to co-evolve.

(c) **The reduction.** The ΔPV field inside the core is collapsed to one number per
    pole, which is what every sweep and ratio in the study plots.

The point to take away from (a): the horizontal axis is the iteration count n, and a
step is *nominally* 3 h. These are perturbation iterations, not a forecast — n × 3 h
must never be read against an observation clock.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _data as D
import style as S

_CTRL = "#4C4C4C"
_PERT = S.C_SENS
_DELTA = S.C_LAT


def _box(ax, xy, w, h, text, fc, ec, fontsize=9.5, tc="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight="bold", color=tc, zorder=4)


def _arrow(ax, a, b, color, style="-|>", lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, color=color, lw=lw,
                                 linestyle=ls, mutation_scale=13, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def _trajectory(ax, *, constrained: bool):
    """One step-by-step diagram; ``constrained`` adds the δq = 0 reset."""
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    y_c, y_p, y_d = 3.35, 1.95, 0.45
    w, h = 1.45, 0.68

    for i, n in enumerate(("n−1", "n", "n+1")):
        x = 0.55 + i * 3.2
        _box(ax, (x, y_c), w, h, f"I$_{{{n}}}$", _CTRL, _CTRL)
        _box(ax, (x, y_p), w, h, f"A$_{{{n}}}$", _PERT, _PERT)
        _box(ax, (x, y_d), w, h, f"δ$_{{{n}}}$", "white", _DELTA, tc=_DELTA)
        # δ = A − I, read downward
        _arrow(ax, (x + w / 2, y_c), (x + w / 2, y_p + h), _CTRL, lw=1.1, ls=":")
        _arrow(ax, (x + w / 2, y_p), (x + w / 2, y_d + h), _DELTA, lw=1.6)
        if i < 2:
            for y, c in ((y_c, _CTRL), (y_p, _PERT)):
                _arrow(ax, (x + w, y + h / 2), (x + 3.2, y + h / 2), c)
                ax.text(x + w + 1.62, y + h / 2 + 0.16, "M", ha="center",
                        va="bottom", fontsize=10, weight="bold", color=c)
            # the forcing enters the perturbed branch only
            ax.text(x + w + 1.62, y_p + h / 2 - 0.42, "+ f", ha="center", va="top",
                    fontsize=9.5, weight="bold", color=_PERT)
        if constrained and i > 0:
            ax.text(x + w / 2, y_d - 0.16, "δq ← 0", ha="center", va="top",
                    fontsize=9, weight="bold", color="#B22222")
            _arrow(ax, (x + w / 2, y_d), (x + w / 2, y_d - 0.14), "#B22222", lw=1.2)

    ax.text(0.15, y_c + h / 2, "control", ha="right", va="center", fontsize=9.5,
            weight="bold", color=_CTRL, rotation=90)
    ax.text(0.15, y_p + h / 2, "perturbed", ha="right", va="center", fontsize=9.5,
            weight="bold", color=_PERT, rotation=90)
    ax.text(0.15, y_d + h / 2, "δ = A − I", ha="right", va="center", fontsize=9.5,
            weight="bold", color=_DELTA, rotation=90)


def plot(*, ic: str = D.DEFAULT_IC, style: str = "note"):
    S.apply()
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.2),
                             gridspec_kw={"height_ratios": [1, 1, 0.72]})

    _trajectory(axes[0], constrained=False)
    axes[0].set_title("(a) a free run — the same operator M drives both branches; "
                      "the forcing f enters only the perturbed one",
                      fontsize=S.FS_LABEL, loc="left")

    _trajectory(axes[1], constrained=True)
    axes[1].set_title("(b) a constrained run — identical, plus one line: after each "
                      "step the moisture part of δ is reset to 0",
                      fontsize=S.FS_LABEL, loc="left")

    # (c) the reduction --------------------------------------------------------
    ax = axes[2]
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    _box(ax, (0.5, 1.1), 2.0, 0.85, "δ  (13 levels\n× 81 × 81)", _DELTA, _DELTA)
    _arrow(ax, (2.5, 1.52), (3.5, 1.52), "#333333")
    _box(ax, (3.5, 1.1), 2.2, 0.85, "ΔPV field\n(PV(I+δ) − PV(I))", "#666666", "#666666")
    _arrow(ax, (5.7, 1.52), (6.7, 1.52), "#333333")
    _box(ax, (6.7, 1.75), 3.2, 0.72,
         "low pole: 700–1000 hPa,  r ≤ 2σ", S.C_LOW, S.C_LOW, fontsize=9)
    _box(ax, (6.7, 0.65), 3.2, 0.72,
         "upper pole: 200–500 hPa,  r ≤ 2σ", S.C_UP, S.C_UP, fontsize=9)
    ax.text(8.3, 0.42, "each box → one number  (max, or its 95th/90th percentile)",
            ha="center", va="top", fontsize=9, style="italic", color=S.C_NOTE)
    ax.set_title("(c) the reduction — how a 3-D response becomes the single number "
                 "every later figure plots", fontsize=S.FS_LABEL, loc="left")

    fig.suptitle("B4 — how the experiment works")
    S.caption(fig, "a diagram, not data; note the x axis of every later figure is the "
                   "iteration count n and a step is nominally 3 h — these are "
                   "perturbation iterations, not a forecast, and n × 3 h must not be "
                   "read against an observation clock", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/b4_method_schematic.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "b4_method_schematic.png"
    S.save(plot(ic=a.ic, style=a.style), out, style=a.style, pdf=not a.no_pdf)
