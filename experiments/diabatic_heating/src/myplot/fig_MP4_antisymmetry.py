"""MP4 — the cooling experiment: ΔPV sections for +A, −A, and what is not a mirror.

A linear response satisfies ΔPV(−A) = −ΔPV(+A) exactly. Rather than argue about it,
this puts the two sections side by side and adds the part that breaks the mirror:

    row 1   ΔPV(+A)                       the warming response
    row 2   ΔPV(−A)                       the cooling response
    row 3   even = ½[ΔPV(+A) + ΔPV(−A)]   what no linear response can produce

All three rows share **one** colour scale, which is the whole point: if the response
were linear, rows 1 and 2 would be exact colour negatives of each other and row 3
would be blank. How far row 3 is from blank, at each lead, is how far the model has
moved from linearity — measured with no reference run and no channel locking, just the
same forcing applied with both signs. The RMS of row 3 as a percentage of the RMS of
the odd part ½[ΔPV(+A) − ΔPV(−A)] is printed on each panel.

Columns are **iterations of the model operator**, not forecast hours: these are
``snapshot`` runs at frozen valid time, so the environment never changes and the storm
never moves. Forcing goes in over n ≤ 8; everything to the right is the manifold
amplifying what it already has. One figure per ladder member, since the row axis is now
the sign of the forcing rather than the vortex.

Reading the sign question this set was built for: the Deep profile puts its maximum at
600 hPa, so cooling should destroy PV below it and generate PV above — row 2 should be
row 1 upside down in sign. It is, for the first couple of iterations. What happens after
that is the interesting part, and it is not symmetric: moist processes are intrinsically
sign-asymmetric (warming dries the column and there is nothing to evaporate, while
cooling drives it toward saturation and the condensation that follows releases latent
heat *opposing* the imposed cooling), so the cooling response is expected to be the
damped one.

``--summary`` draws the compact line version instead: nonlinear fraction and
cooling/warming amplitude ratio against iteration, for every amplitude at once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402

import _data as D                                                     # noqa: E402
import style as S                                                     # noqa: E402
import _snap as SNAP                                                  # noqa: E402
from fig_MP2_pv_evolution import _collect, R_MAX_KM                   # noqa: E402
from llat_manifold.diagnostics import _idealized as _id               # noqa: E402

_MEMBERS = (D.STRONG, D.SEV)
_AMPS = (1, 2, 5)                    # |A| available with both signs
_ITERS = (1, 2, 4, 8, 16, 24, 32, 40)
FIGS = D.FIGS_ROOT / "myplot"
FORCING_END = 8            # amp_K is spread over the first 8 iterations


def _rms(a):
    return float(np.sqrt((a ** 2).mean()))


def _pair(amp, ns):
    """{member label: {n: (r_km, p_hPa, warm, cool)}} for one |amplitude|."""
    pos = dict(_collect(amp, ns))
    neg = dict(_collect(-amp, ns))
    out = {}
    for lab in pos:
        if lab not in neg:
            continue
        out[lab] = {h: (*pos[lab][h][:2], pos[lab][h][2], neg[lab][h][2])
                    for h in ns if h in pos[lab] and h in neg[lab]}
    return out


def _win(r_km, p_hpa):
    return np.ix_(np.asarray(p_hpa) >= _id.P_TOP_HPA, r_km <= R_MAX_KM)


def plot_sections(amp, lab, panels, ns, *, style: str = "note"):
    """Three-row ΔPV strip for one member: warming, cooling, and the even residual."""
    S.apply()
    have = [h for h in ns if h in panels]
    rows = [(f"warming\nΔPV(+{amp} K)", lambda w, c: w),
            (f"cooling\nΔPV(−{amp} K)", lambda w, c: c),
            ("not a mirror\n½[warm + cool]", lambda w, c: (w + c) / 2)]

    # One scale for all three rows — the residual row must be readable *against* the
    # responses, not rescaled to look as big as them.
    vals = []
    for h in have:
        r_km, p_hpa, w, c = panels[h]
        win = _win(r_km, p_hpa)
        vals += [np.abs(w[win]).ravel(), np.abs(c[win]).ravel()]
    lim = float(np.nanpercentile(np.concatenate(vals), 99.5)) or 1.0
    levels = np.linspace(-lim, lim, 25)

    fig, axes = plt.subplots(len(rows), len(have), squeeze=False, sharey=True,
                             figsize=(2.05 * len(have) + 1.4, 3.5 * len(rows) + 1.0))
    for i, (label, pick) in enumerate(rows):
        for j, h in enumerate(have):
            ax = axes[i, j]
            r_km, p_hpa, w, c = panels[h]
            fld = pick(w, c)
            cf = ax.contourf(r_km, p_hpa, fld, levels=levels, cmap=S.CMAP_PV,
                             extend="both")
            ax.contour(r_km, p_hpa, fld, levels=[0], colors="k", linewidths=0.5)
            if i == 2:      # how far from a mirror, as a fraction of the linear part
                win = _win(r_km, p_hpa)
                odd = _rms((w[win] - c[win]) / 2)
                ax.text(0.96, 0.04, f"{100 * _rms(fld[win]) / odd:.0f}%",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=S.FS_ANNOT + 1, weight="bold", color="#222222",
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CCCCCC",
                                  alpha=0.9))
            ax.set_ylim(1000, _id.P_TOP_HPA)
            ax.set_xlim(0, R_MAX_KM)
            ax.tick_params(labelsize=S.FS_TICK - 2)
            if i == 0:
                forced = " (forcing on)" if SNAP.forcing_on(h) else ""
                ax.set_title(f"{SNAP.label(h)}{forced}", fontsize=S.FS_TICK,
                             color=(S.C_SENS if h <= FORCING_END else "k"))
            if i == len(rows) - 1:
                ax.set_xlabel("radius  (km)", fontsize=S.FS_TICK - 1)
        axes[i, 0].set_ylabel(f"{label}\n\npressure  (hPa)", fontsize=S.FS_TICK)

    cb = fig.colorbar(cf, ax=axes.ravel().tolist(), pad=0.012, fraction=0.02)
    cb.set_label("azimuthal-mean ΔPV  (PVU)", fontsize=S.FS_LABEL - 1, weight="bold")
    fig.suptitle(f"MP4 — same Deep forcing, both signs, at {amp} K  "
                 f"({lab.replace(chr(10), '  ')}, {D.IC_LABEL['axisym']})", wrap=True)
    S.caption(fig, "if the response were linear, row 2 would be row 1 with the colours "
                   "reversed and row 3 would be blank — all three share one colour "
                   "scale so that comparison is honest; the number on each row-3 panel "
                   "is RMS(row 3) as a percentage of RMS of the odd part, i.e. how "
                   "large the non-mirror part is relative to what a linear response "
                   "would give; forcing is injected over n <= 8; "
                   "n counts iterations of the model operator, not forecast hours; the "
                   "environment is frozen, so the growth is the vortex's own",
              style, wrap=True)
    return fig


def plot_summary(data, ns, *, style: str = "note"):
    """The compact line version: nonlinear fraction and cooling/warming ratio."""
    S.apply()
    labels = sorted({lab for d in data.values() for lab in d})
    fig, axes = plt.subplots(2, len(labels), figsize=(6.4 * len(labels), 9.0),
                             sharex=True, squeeze=False)
    colors = [S.C_WEAK, S.C_LAT, S.C_SENS, "#009E73"]
    for j, lab in enumerate(labels):
        for k, amp in enumerate(sorted(data)):
            if lab not in data[amp]:
                continue
            odd, even, warm, cool = data[amp][lab]
            c = colors[k % len(colors)]
            axes[0, j].plot(ns, 100 * even / odd, "-o", color=c, lw=2.2, ms=5,
                            label=f"|A| = {amp} K")
            axes[1, j].plot(ns, cool / warm, "-o", color=c, lw=2.2, ms=5)
        axes[0, j].axhline(100, color=S.C_GUIDE, lw=1.0, ls=":")
        axes[1, j].axhline(1.0, color=S.C_GUIDE, lw=1.0, ls=":")
        for i in (0, 1):
            S.forcing_span(axes[i, j], FORCING_END, label=(i == 0 and j == 0))
            axes[i, j].set_ylim(0, None)
        axes[0, j].set_title(lab.replace("\n", "  "))
        axes[1, j].set_xlabel("Iteration n")
        axes[0, j].legend(loc="upper left")
    axes[0, 0].set_ylabel("nonlinear fraction\nRMS(even) / RMS(odd)  (%)")
    axes[1, 0].set_ylabel("sign asymmetry\nRMS(cooling) / RMS(warming)")
    S.panel_letters(axes.ravel())
    fig.suptitle("MP4 (summary) — how long the heating response stays linear", wrap=True)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    S.caption(fig, "a ratio near 100 % is also what two uncorrelated fields give, so "
                   "only values well above it require genuine same-signed rectification; "
                   "beyond n ≈ 48 h the right-hand end of every curve is the least "
                   "trustworthy part", style, wrap=True)
    return fig


def _summary_stats(amp, ns):
    """{member label: (odd, even, warm, cool) RMS arrays} — for the line version."""
    out = {}
    for lab, panels in _pair(amp, ns).items():
        cols = [[], [], [], []]
        for h in ns:
            if h not in panels:
                for c in cols:
                    c.append(np.nan)
                continue
            r_km, p_hpa, w, c_ = panels[h]
            win = _win(r_km, p_hpa)
            for col, a in zip(cols, [(w[win] - c_[win]) / 2, (w[win] + c_[win]) / 2,
                                     w[win], c_[win]]):
                col.append(_rms(a))
        out[lab] = tuple(np.array(c) for c in cols)
    return out


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--amps", nargs="+", type=int, default=[5],
                    help="|amplitudes| in K available with both signs (default: "
                         "%(default)s; --summary defaults to all of "
                         f"{list(_AMPS)})")
    ap.add_argument("--iters", nargs="+", type=int, default=list(_ITERS),
                    help="iterations to draw as columns (default: %(default)s)")
    ap.add_argument("--summary", action="store_true",
                    help="draw the compact line version instead of the sections")
    ap.add_argument("--outdir", default=None, help="default: figs/myplot/")
    a = ap.parse_args()

    ns = tuple(a.iters)
    outdir = Path(a.outdir) if a.outdir else FIGS

    if a.summary:
        amps = a.amps if a.amps != [5] else list(_AMPS)
        data = {amp: d for amp in amps if (d := _summary_stats(amp, ns))}
        if not data:
            raise SystemExit("[MP4] no ±A pair available")
        S.save(plot_summary(data, ns, style=a.style),
               outdir / "mp4_antisymmetry_summary.png", style=a.style,
               pdf=not a.no_pdf)
        raise SystemExit(0)

    for amp in a.amps:
        pairs = _pair(amp, ns)
        if not pairs:
            print(f"[MP4] |A| = {amp} K: no ±pair on disk, skipping")
            continue
        for lab, panels in pairs.items():
            frac = " ".join(
                f"{h}h:{100 * _rms((panels[h][2] + panels[h][3])[_win(*panels[h][:2])] / 2) / _rms((panels[h][2] - panels[h][3])[_win(*panels[h][:2])] / 2):.0f}%"
                for h in ns if h in panels)
            print(f"[MP4] |A|={amp}K {lab.replace(chr(10), ' ')}: non-mirror {frac}")
            tag = lab.split("(")[-1].split()[0]          # '0920' / '0921'
            S.save(plot_sections(amp, lab, panels, ns, style=a.style),
                   outdir / f"mp4-{amp:02d}_{tag}_pv_sections.png",
                   style=a.style, pdf=not a.no_pdf)
            plt.close("all")
