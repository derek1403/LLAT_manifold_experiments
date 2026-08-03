"""H2 structure — the q channel is its own direction on the manifold, not scaled heating.

6×3 structure panel for the δq-only sweep (``dq_measured``, ``sweepdq_*``), same
layout as the δq = 0 maps: columns = amplitudes, rows = ΔPV on the upper-min
layer, the low-max layer, and the azimuthal-mean r–z section; 2σ core circled,
extremum marked ×.

Read it against the heating maps (finding 6's comparison table): the δq response
peaks at 850 hPa (heating: 700), destroys at 200 hPa (heating: 400), and its r–z
pattern is an outward-tilting mid/upper positive arm rather than an upright core
column; its extremum stays put with amplitude instead of marching into the core.
Same manifold, two *related but different* directions — the q channel has its own
response structure, it is not a rescaled copy of the heating direction.
"""
from __future__ import annotations

import argparse

import _data as D
import style as S
from fig_H0p_qlock_maps import plot as _plot_maps


def plot(init: str = D.STRONG, style: str = "note", *, ic: str = D.DEFAULT_IC,
         stat: str = D.DEFAULT_STAT):
    return _plot_maps(
        init, style=style, prefix="sweepdq", ic=ic, stat=stat,
        suptitle=(f"H2 structure — δq-only (measured) sweep: the q direction is not scaled "
                  f"heating  ({D.CASE[init]}, hour 24)"),
        caption=("dashed circle = 2σ core, × = where the reduction sits; vs heating: the low pole "
                 "sits at 850 hPa (not 700) and the upper one at 200 hPa (not 400), an "
                 "outward-tilting arm instead of an upright PV column, and its position "
                 "holds still instead of marching inward"))


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--init", default=D.STRONG)
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/<stat>/h2_dq_maps.png")
    a = ap.parse_args()
    out = a.out or D.figs_dir(a.ic, a.stat) / "h2_dq_maps.png"
    S.save(plot(a.init, a.style, ic=a.ic, stat=a.stat), out, style=a.style, pdf=not a.no_pdf)
