"""Single style source for the moisture-binding figure set.

Every knob lives in the CONSTANTS block below — change it here and all figures
follow. Nothing else in ``experiments/diabatic_heating/src/`` may set a colour,
a font size, or a colormap.

Two output styles, one code path:

* ``note``  — the research-notebook look: the italic footer caption stays on the
  canvas, so a figure dropped into a markdown note is self-explanatory.
* ``paper`` — submission look: no on-canvas caption (it belongs in the LaTeX
  caption), panel letters, vector PDF alongside the PNG.

Bold axis labels and titles are deliberate and are kept in both styles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# CONSTANTS — the whole style lives here
# --------------------------------------------------------------------------- #
# Init-case hues. Okabe–Ito blue/vermillion: colour-blind safe and print-safe.
# (Revert to matplotlib defaults with C_WEAK, C_STRONG = "tab:blue", "tab:red".)
C_WEAK = "#0072B2"     # init 2025091700 — weak/developing stage
C_STRONG = "#D55E00"   # init 2025092000 — mature/strong stage

# Dipole-pole hues (fixed assignment across every figure that splits the dipole).
C_LOW = "#0072B2"      # low-level 700–1000 hPa PV generation
C_UP = "#E69F00"       # upper-level 200–500 hPa PV destruction

# Energy-partition series.
C_SENS = "#D55E00"     # sensible cp·∫δT dm
C_LAT = "#0072B2"      # latent  Lv·∫δq dm
C_TOTAL = "#222222"    # total
C_INJ = "#7F7F7F"      # injected reference

CMAP_PV = "RdBu_r"     # ΔPV (diverging, perceptually gentler than bwr)
CMAP_Q = "BrBG"        # moisture anomaly (green = moist, brown = dry)

C_GUIDE = "#8C8C8C"    # zero lines, reference lines, forcing-window shading
C_NOTE = "#555555"     # caption / annotation text

FS_TITLE = 13
FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 9.5
FS_ANNOT = 8
FS_CAPTION = 7.5

DPI_PNG = 300

# Font: no Helvetica/Arial on this box; Nimbus Sans is a metric-compatible clone.
# No CJK font is installed, so all on-canvas text is English by necessity.
_FONTS = ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]


def apply() -> None:
    """Install the rcParams. Call once at the top of every figure script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _FONTS,
        "font.weight": "bold",           # bold everywhere incl. tick labels (user preference)
        "axes.labelsize": FS_LABEL,
        "axes.labelweight": "bold",      # bold labels/titles: kept on purpose
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": FS_LEGEND,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "figure.titlesize": FS_TITLE + 1,
        "figure.titleweight": "bold",
        "savefig.dpi": DPI_PNG,
        "savefig.bbox": "tight",
        "mathtext.default": "regular",
    })


def case_colors(inits) -> dict:
    """Map init-time strings to their fixed hue (weak first, then strong)."""
    palette = [C_WEAK, C_STRONG, "#009E73", "#CC79A7"]
    return {init: palette[i % len(palette)] for i, init in enumerate(sorted(inits))}


def zero_line(ax) -> None:
    ax.axhline(0, color=C_GUIDE, lw=0.8, zorder=0)


def forcing_span(ax, t_end: float, *, label: bool = True) -> None:
    """Shade the injection window [0, t_end] — clearer than a single dashed line.

    Every run in this set injects over the first 24 nominal hours; what happens
    to the right of the shading is free evolution, which is where the moisture
    binding shows up.
    """
    ax.axvspan(0, t_end, color=C_GUIDE, alpha=0.10, lw=0, zorder=0)
    ax.axvline(t_end, color=C_GUIDE, lw=1.0, ls=":", zorder=1)
    if label:
        ax.text(t_end, 1.005, f"forcing ends ({t_end:g} h) ", transform=ax.get_xaxis_transform(),
                ha="right", va="bottom", fontsize=FS_ANNOT, style="italic", color=C_NOTE)


def panel_letters(axes, *, start: int = 0, loc=(0.012, 0.985)) -> None:
    """(a) (b) (c) … in the top-left of each axes — required for submission."""
    for k, ax in enumerate(np.asarray(axes).ravel()):
        if not ax.get_visible():
            continue
        ax.text(loc[0], loc[1], f"({chr(97 + start + k)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=FS_LABEL, weight="bold", zorder=6,
                bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none", alpha=0.78))


def caption(fig, text: str, style: str = "note") -> None:
    """The italic footer. Printed in ``note`` style; suppressed for ``paper``."""
    if style != "note" or not text:
        return
    fig.text(0.995, 0.004, text, ha="right", va="bottom", weight="normal",
             fontsize=FS_CAPTION, style="italic", color=C_NOTE)


def save(fig, out_path, *, style: str = "note", pdf: bool = True) -> None:
    """Write PNG (300 dpi) and, for submission, the vector PDF beside it."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI_PNG)
    print(f"[fig] wrote {out}")
    if pdf and style == "paper":
        p = out.with_suffix(".pdf")
        fig.savefig(p)
        print(f"[fig] wrote {p}")


def add_style_args(ap):
    """The two flags every figure script shares."""
    ap.add_argument("--style", choices=("note", "paper"), default="note",
                    help="note = on-canvas caption (default); paper = no caption + PDF")
    ap.add_argument("--no-pdf", action="store_true", help="skip the vector PDF")
    return ap
