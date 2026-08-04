"""MP6 — the BG1 background-state panel, but with time along the columns.

[`seminar1/figs/src/make_background_state.py`](/wk2/pc/seminar/seminar1/figs/src/make_background_state.py)
draws five map fields of the axisymmetric IC — 10 m wind, precipitation, 850 hPa
vorticity, 700 hPa ω, TCWV — with the two ladder members as its two columns, at n = 0.
This is the same five rows with the same colormaps and the same limits, but the columns
are **iterations of the model operator**: what one member looks like as the 5 K heating
goes in and the manifold then amplifies it on its own. One figure per member.

These are ``snapshot`` runs at frozen valid time, so the map window is identical in
every column — the vortex never travels and never makes landfall, and n is an iteration
count, not a forecast hour.

The colormaps, level tables, row definitions and wind overlays are *imported* from that
script rather than copied, so the two figures stay comparable by construction — this
module only changes where the fields come from (a run bundle instead of an IC npz) and
what the columns mean.

By default the **perturbed** state is drawn (ū + δ), i.e. what the heating produced.
``--field control`` draws the frozen background in every column (a flat-line check that
nothing in the reference moves); ``--field delta`` draws the
difference, but note the five colormaps were built for absolute fields (precipitation
0–20 mm, wind 0–40 m/s) and a difference field will not use them sensibly.

Column n = 0 is the frozen background ū itself — the reference every iteration is
measured against, and the natural "before" column.

Limits default to ``--paper-limits`` behaviour (the notebook's fixed vmin/vmax) so that
the two member figures, and BG1 itself, can be laid side by side. ``--auto-limits``
rescales each row to what is actually on the canvas instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The BG1 script is the single source of the five colormaps and row specs. It lives in
# the seminar deck, not this repo; fail loudly rather than silently re-inventing them.
_BG1_DIR = Path("/wk2/pc/seminar/seminar1/figs/src")
if not (_BG1_DIR / "make_background_state.py").exists():          # pragma: no cover
    raise SystemExit(f"MP6 needs BG1's colormaps: {_BG1_DIR}/make_background_state.py "
                     "not found — has the seminar deck moved?")
sys.path.insert(0, str(_BG1_DIR))

import cartopy.crs as ccrs                                         # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.ticker import FuncFormatter                        # noqa: E402

import _data as D                                                  # noqa: E402
import make_background_state as BG1                                # noqa: E402
import _snap as SNAP                                               # noqa: E402
from llat_manifold import layout                                   # noqa: E402
from llat_manifold.idealized_vortex import axisymmetric as ax      # noqa: E402
from llat_manifold.idealized_vortex.azimuthal import relative_vorticity  # noqa: E402

_MEMBERS = (D.STRONG, D.SEV)
FIGS = D.FIGS_ROOT / "myplot"
_ITERS = (0, 2, 4, 8, 16, 40)
CATEGORY = {D.STRONG: "Category 1", D.SEV: "Category 4"}


class _State(BG1.Case):
    """A BG1 ``Case`` built from arbitrary (upper, surface) arrays.

    Same attribute names as :class:`make_background_state.Case`, so BG1's
    ``_row_specs`` and ``_overlay`` work on it unchanged; only the source differs.
    ``Case.__init__`` is bypassed because it is hard-wired to an IC npz path.
    """

    def __init__(self, upper, surface, label: str):
        self.label, self.category, self.path = label, "", None
        sv, uv, lv = layout.surface_vars(), layout.upper_vars(), layout.pressure_levels()
        s, u = surface, upper

        self.lats, self.lons = s[:, 0, -1], s[0, :, -2]
        self.flip = self.lats[0] > self.lats[-1]
        self.init_time = None

        self.u10, self.v10 = s[:, :, sv.index("u10")], s[:, :, sv.index("v10")]
        self.ws10 = np.hypot(self.u10, self.v10)
        self.msl = s[:, :, sv.index("msl")] / 100.0
        self.tp = s[:, :, sv.index("tp")] * 1000.0
        self.tcwv = s[:, :, sv.index("tcwv")]
        self.sst = s[:, :, sv.index("sst_filled")]
        self.landmask = s[:, :, sv.index("landmask")]

        def lev(name, hpa):
            return u[lv.index(hpa), :, :, uv.index(name)]

        self.u850, self.v850 = lev("u", BG1.VORT_LEVEL), lev("v", BG1.VORT_LEVEL)
        self.vort850 = relative_vorticity(self.u850, self.v850,
                                          self.lats, self.lons) * 1e5
        self.uw, self.vw = lev("u", BG1.OMEGA_LEVEL), lev("v", BG1.OMEGA_LEVEL)
        self.omega = lev("w", BG1.OMEGA_LEVEL)
        self.us, self.vs = lev("u", BG1.STREAM_LEVEL), lev("v", BG1.STREAM_LEVEL)

        r_km, _p, _z, vt, _th = ax.rz_sections(u, s)
        i = int(np.nanargmax(vt[lv.index(850)]))
        self.peak_vt, self.rmw = float(vt[lv.index(850)][i]), float(r_km[i])
        self.mslp = float(self.msl.min())


def _state_at(init, amp, n, field):
    """One column's state at iteration ``n``.

    ``n = 0`` is the frozen background ū itself — the reference every iteration is
    measured against, so it is the natural "before" column. There is no separate
    control trajectory in snapshot mode: ū *is* the control, at every n.
    """
    run = SNAP.run_dir(amp, init)
    b_up, b_sfc = SNAP._background(run)
    if n == 0:
        return _State(b_up, b_sfc, SNAP.label(0))
    d_up, d_sfc = SNAP.delta(run, n)
    # δ is the forcing response outright — the base inside M is u₀ at every iteration, so
    # f = 0 gives δ ≡ 0 and there is no drift to peel off. ``--field null`` draws the
    # amp = 0 run, which is now the round-off noise floor rather than a background.
    null = SNAP.null_run(run)
    z_up, z_sfc = SNAP.delta(null, n) if null is not None else (0.0, 0.0)
    up, sfc = {"perturbed": (b_up + d_up, b_sfc + d_sfc),
               "control": (b_up, b_sfc),
               "null": (b_up + z_up, b_sfc + z_sfc),
               "delta": (d_up, d_sfc)}[field]
    return _State(up, sfc, SNAP.label(n))


def plot(init, amp, ns, field, *, paper_limits: bool = True):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.weight": "bold", "axes.titleweight": "bold",
        "axes.labelweight": "bold", "axes.grid": False,
    })
    cols = [_state_at(init, amp, h, field) for h in ns]
    rows = BG1._row_specs(cols, paper_limits)
    pc = ccrs.PlateCarree()

    ncol, nrow = len(cols), len(rows)
    fig = plt.figure(figsize=(4.9 * ncol + 1.5, 3.55 * nrow), dpi=200)
    gs = fig.add_gridspec(nrow, ncol, wspace=0.06, hspace=0.07,
                          left=0.045, right=0.885, top=0.94, bottom=0.085)

    for r, spec in enumerate(rows):
        mesh = None
        for c, case in enumerate(cols):
            axx = fig.add_subplot(gs[r, c], projection=pc)
            axx.coastlines(resolution="50m", color="darkslategray", linewidth=0.9)

            shade = dict(cmap=spec["cmap"])
            if "norm" in spec:
                shade["norm"] = spec["norm"]
            else:
                shade.update(vmin=spec["vmin"], vmax=spec["vmax"])
            mesh = axx.pcolormesh(case.lons, case.lats, getattr(case, spec["field"]),
                                  shading="auto", transform=pc, **shade)
            if spec["contour"] == "msl":
                cs = axx.contour(case.lons, case.lats, case.msl,
                                 levels=np.arange(900, 1033, 4), colors="b",
                                 linewidths=1.0, transform=pc)
                axx.clabel(cs, cs.levels[::2], inline=True, fontsize=7, fmt="%d")
            BG1._overlay(axx, case, spec["overlay"], pc)

            axx.set_xticks(np.arange(120, 141, 5), crs=pc)
            axx.set_yticks(np.arange(10, 27, 5), crs=pc)
            axx.xaxis.set_major_formatter(FuncFormatter(BG1._lon_fmt))
            axx.yaxis.set_major_formatter(FuncFormatter(BG1._lat_fmt))
            axx.tick_params(direction="in", labelsize=9, width=1.0, length=4,
                            labelleft=(c == 0), labelbottom=(r == nrow - 1))
            axx.set_extent([case.lons.min(), case.lons.max(),
                            case.lats.min(), case.lats.max()], crs=pc)
            if r == 0:
                forced = "  (forcing on)" if SNAP.forcing_on(ns[c]) else ""
                axx.set_title(f"{case.label}{forced}", fontsize=15, pad=8,
                              color=("#D55E00" if SNAP.forcing_on(ns[c]) else "k"))
                axx.annotate(
                    f"peak $V_t$(850) = {case.peak_vt:.1f} m s$^{{-1}}$\n"
                    f"RMW = {case.rmw:.0f} km   MSLP = {case.mslp:.0f} hPa\n"
                    f"peak $\\zeta_{{850}}$ = {case.vort850.max():.0f}"
                    f" $\\times10^{{-5}}$ s$^{{-1}}$",
                    (0.03, 0.035), xycoords="axes fraction", ha="left", va="bottom",
                    fontsize=8.5, color="black",
                    bbox=dict(fc="white", ec="0.4", lw=0.6, alpha=0.85, pad=2.4))

        pos = axx.get_position()
        cax = fig.add_axes([0.897, pos.y0 + 0.004, 0.013, pos.height - 0.008])
        cb = fig.colorbar(mesh, cax=cax, extend=spec["extend"], extendfrac=0.06,
                          ticks=spec.get("ticks"))
        cb.set_label(spec["label"], fontsize=10.5)
        cb.ax.tick_params(labelsize=9, direction="in")

    what = {"perturbed": "the forcing-attributable state, ū + (δ − δ⁰)",
            "control": "the frozen background ū (identical in every column)",
            "null": "the background's own drift, ū + δ⁰ (no forcing at all)",
            "delta": "the forcing-attributable perturbation, δ − δ⁰"}[field]
    fig.suptitle(f"{CATEGORY.get(init, init)} ({init[4:8]} 00Z) — {what} under "
                 f"{amp} K Deep heating, iteration by iteration", fontsize=17, y=0.978)
    fig.text(0.5, 0.016,
             f"Semi-linear power iteration at FROZEN valid time: the background "
             f"ū = M(u₀) is computed once and every iteration is realigned to it, so "
             f"lat/lon, SST, f, terrain and the land mask are byte-identical in every "
             f"column — the map window does not move and the vortex never travels or "
             f"makes landfall. n counts applications of the model operator; it is NOT "
             f"a forecast clock. Heating goes in over n ≤ 8 (orange headings) at "
             f"{amp}/8 K per iteration and nothing is injected after that. ū is not a "
             f"fixed point of M, so an amp = 0 twin (δ⁰) is subtracted to isolate what "
             f"the forcing did — run --field null to see that drift on its own. Only the "
             f"atmosphere was azimuthally averaged, so real coastlines sit under an "
             f"initially circular vortex, and the asymmetry that grows across the "
             f"columns is the lower boundary re-imprinting itself plus the vortex's own "
             f"response. Colour limits are BG1's fixed ones so the columns, the two "
             f"members and BG1 itself are directly comparable. $\\omega$ is at "
             f"{BG1.OMEGA_LEVEL} hPa; DLAMPty has no 750 hPa level.",
             ha="center", va="bottom", fontsize=8.5, color="#444444", style="italic",
             wrap=True)
    return fig, cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amp", type=int, default=5)
    ap.add_argument("--members", nargs="+", default=list(_MEMBERS))
    ap.add_argument("--iters", nargs="+", type=int, default=list(_ITERS),
                    help="iterations to draw as columns (default: %(default)s)")
    ap.add_argument("--field", choices=("perturbed", "control", "null", "delta"),
                    default="perturbed",
                    help="perturbed = ū + (δ − δ⁰), the forcing-attributable state; "
                         "control = the frozen ū in every column; null = ū + δ⁰, the "
                         "background's own power-iteration drift; delta = δ − δ⁰")
    ap.add_argument("--auto-limits", action="store_true",
                    help="rescale each row to this figure instead of BG1's fixed limits")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()

    outdir = Path(a.outdir) if a.outdir else FIGS
    outdir.mkdir(parents=True, exist_ok=True)
    for init in a.members:
        fig, cols = plot(init, a.amp, tuple(a.iters), a.field,
                         paper_limits=not a.auto_limits)
        out = outdir / f"mp6-{a.amp:02d}_{init[4:8]}_state_evolution.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        if not a.no_pdf:
            fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[MP6] wrote {out}")
        for h, c in zip(a.iters, cols):
            print(f"    n={h:4d}  peak Vt {c.peak_vt:5.1f} m/s  RMW {c.rmw:4.0f} km  "
                  f"MSLP {c.mslp:6.1f} hPa  ws10max {c.ws10.max():5.1f}  "
                  f"tp max {c.tp.max():6.2f} mm  TCWV {c.tcwv.max():5.1f}")
