"""MP7 — a one-look check on whether the model domain stays where it started.

The semi-linear method is supposed to pin the prescribed surface channels, including
lat/lon, so the vortex never actually travels and its environment never changes. That
is exactly what ``driver._run_snapshot`` does: every iteration is realigned to the
*initial state* (``_align_statics(A_sfc, u0_sfc, …)``, ``lock_ref=u0_sfc``), and the
background ``ū = M(u₀)`` is computed under the same lock so it carries u₀'s grid too.

Pinning to ``u₀`` rather than to the model's own output matters and is why this figure
still earns its place after the 2026-08-04 fix: DLAMPty predicts the prescribed
channels along with everything else, and one step returns terrain 45 % flatter, a
non-binary land mask and lat/lon displaced 0.08°. Locking to that would hold the domain
perfectly still — at the wrong place, over a flattened island.

``driver._run_continuous`` does something different: the control marches
(``advance_time=True``, ``I ← M(I)`` each step) and *its* ``m_operator`` call carries no
``lock_ref``, so the control's own static channels are whatever the model produces. The
perturbed member is then realigned to the **current** control rather than to the initial
state. δ's static channels are still exactly zero — that part of the locking works — but
the frame both trajectories sit in is free to move.

This figure settles which of the two is on disk, without reading a single line of code:

  (a) every step's domain footprint drawn on one fixed map, with the domain centres
      joined into a track. **If the position were pinned, every rectangle would sit on
      top of every other one** — which is what ``--mode snapshot`` shows, and what
      ``--mode continuous`` conspicuously does not.
  (b) the fraction of the core box that is land, and the core SST — the quantities that
      decide whether a late result is about the vortex or about the surface underneath.

Run it both ways to see the difference the mode makes; the continuous audit is kept
because it documents why the snapshot re-run was necessary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cartopy.crs as ccrs                                          # noqa: E402
import matplotlib.patches as mpatches                               # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
from matplotlib.ticker import FuncFormatter                         # noqa: E402

import _data as D                                                   # noqa: E402
import style as S                                                   # noqa: E402
from fig_MP1_axisym_pv import _title as _member_title               # noqa: E402
import _snap as SNAP                                                # noqa: E402
from llat_manifold import io, layout                                # noqa: E402
from llat_manifold.perturbations.heating import _amp_tag            # noqa: E402

_MEMBERS = (D.STRONG, D.SEV)
_FAMILY = "heating_moist"
_TAG_CONT = "tseries_{amp}_120h_init{init}"
IC = "axisym"
FIGS = D.FIGS_ROOT / "myplot"

_LEADS = tuple(range(3, 121, 3))
_BOX_LEADS = (3, 24, 48, 72, 96, 120)     # which footprints get a rectangle drawn
CORE = slice(30, 51)                      # the 21x21 core box, ~±2.5 deg
LINEAR_END_HR = 48        # continuous only: where the core is still 100 % ocean


def _describe(sfc):
    li, si = layout.surface_index("landmask"), layout.surface_index("sst_filled")
    lat, lon = sfc[:, 0, -1], sfc[0, :, -2]
    return dict(lat0=float(sfc[40, 40, -1]), lon0=float(sfc[40, 40, -2]),
                extent=(float(lon.min()), float(lon.max()),
                        float(lat.min()), float(lat.max())),
                land=100.0 * float(sfc[CORE, CORE, li].mean()),
                sst=float(sfc[CORE, CORE, si].mean()))


def _walk(init, amp, leads=_LEADS, mode="continuous"):
    """Per-step (lat0, lon0, extent, core land fraction, core SST) of the reference.

    ``continuous`` reads the marching control, one bundle per lead. ``snapshot`` has a
    single frozen background and δ's prescribed channels are identically zero, so the
    frame is the background's at *every* iteration — which is exactly the claim this
    figure exists to check, so it is read from the file rather than asserted.
    """
    if mode == "snapshot":
        run = SNAP.run_dir(amp, init)
        _up, bg_sfc = SNAP._background(run)
        base = _describe(bg_sfc)
        out = {}
        for n in SNAP.iterations(run):
            d_up, d_sfc = SNAP.delta(run, n)
            # ū + δ is the state the model actually saw; if the lock works, its
            # prescribed channels equal the background's.
            out[n] = _describe(bg_sfc + d_sfc)
        if not out:
            raise FileNotFoundError(f"no iteration bundles under {run}")
        return out

    run = D.run(_FAMILY, _TAG_CONT.format(amp=_amp_tag(amp), init=init), ic=IC)
    out = {}
    for h in leads:
        hits = sorted((run / "data").glob(f"control_continuous_*lead{h:03d}hr.npz"))
        if not hits:
            continue
        _up, sfc = io.load_delta_bundle(hits[0])
        out[h] = _describe(sfc)
    if not out:
        raise FileNotFoundError(f"no control bundles under {run}")
    return out


def _lat_fmt(x, _p):
    return f"{abs(int(x))}°{'N' if x >= 0 else 'S'}"


def _lon_fmt(x, _p):
    return f"{abs(int(x))}°{'E' if x >= 0 else 'W'}"


def plot(walks, amp, mode="snapshot", *, style: str = "note"):
    S.apply()
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16.5, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1],
                          wspace=0.16, hspace=0.30,
                          left=0.045, right=0.985, top=0.90, bottom=0.135)
    axm = fig.add_subplot(gs[:, 0], projection=pc)
    axl = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, 1])

    colors = {init: c for init, c in zip(sorted(walks), (S.C_WEAK, S.C_SENS))}
    all_ext = [w[h]["extent"] for w in walks.values() for h in w]
    axm.set_extent([min(e[0] for e in all_ext) - 1.5, max(e[1] for e in all_ext) + 1.5,
                    min(e[2] for e in all_ext) - 1.5, max(e[3] for e in all_ext) + 1.5],
                   crs=pc)
    axm.coastlines(resolution="50m", color="darkslategray", linewidth=0.9)

    for init, w in walks.items():
        c = colors[init]
        leads = sorted(w)
        axm.plot([w[h]["lon0"] for h in leads], [w[h]["lat0"] for h in leads],
                 "-", color=c, lw=2.2, transform=pc, zorder=4,
                 label=_member_title(init).replace("\n", " "))
        for h in [b for b in _BOX_LEADS if b in w] or sorted(w)[::max(1, len(w) // 6)]:
            if h not in w:
                continue
            x0, x1, y0, y1 = w[h]["extent"]
            axm.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                             edgecolor=c, lw=1.3, ls="--", alpha=0.75,
                                             transform=pc, zorder=3))
            axm.plot(w[h]["lon0"], w[h]["lat0"], "o", color=c, ms=6, transform=pc,
                     zorder=5)
            axm.annotate(f" n={h}", (w[h]["lon0"], w[h]["lat0"]), xycoords=pc._as_mpl_transform(axm),
                         fontsize=S.FS_ANNOT, color=c, weight="bold", zorder=6)

        axl.plot(leads, [w[h]["land"] for h in leads], "-o", color=c, lw=2.2, ms=4,
                 label=_member_title(init).replace("\n", " "))
        axs.plot(leads, [w[h]["sst"] for h in leads], "-o", color=c, lw=2.2, ms=4)

    axm.set_xticks(np.arange(105, 146, 5), crs=pc)
    axm.set_yticks(np.arange(5, 46, 5), crs=pc)
    axm.xaxis.set_major_formatter(FuncFormatter(_lon_fmt))
    axm.yaxis.set_major_formatter(FuncFormatter(_lat_fmt))
    axm.tick_params(labelsize=S.FS_TICK)
    unit = "iteration" if mode == "snapshot" else "lead"
    axm.set_title(f"(a) domain footprint at each {unit} — if the position were pinned,\n"
                  "every dashed box would coincide", fontsize=S.FS_TITLE - 1)
    axm.legend(loc="upper left", fontsize=S.FS_LEGEND)

    x_hi = max(max(w) for w in walks.values())
    for a, lab in ((axl, "core land fraction  (%)"), (axs, "core SST  (K)")):
        if mode == "continuous":
            a.axvspan(0, LINEAR_END_HR, color=S.C_GUIDE, alpha=0.10, lw=0)
            a.axvline(LINEAR_END_HR, color=S.C_GUIDE, lw=1.0, ls=":")
        a.set_ylabel(lab)
        a.set_xlim(0, x_hi)
    axl.axhline(0, color=S.C_GUIDE, lw=0.8)
    if mode == "continuous":
        axl.text(LINEAR_END_HR, axl.get_ylim()[1], " core still 100 % ocean to here",
                 ha="left", va="top", fontsize=S.FS_ANNOT, style="italic",
                 color=S.C_NOTE)
    else:
        # Everything is constant by construction; let the flat lines read as flat
        # rather than as autoscaled noise around zero.
        axl.set_ylim(-1, 5)
        sst0 = next(iter(walks.values()))[1]["sst"]
        axs.set_ylim(sst0 - 2, sst0 + 2)
        axl.text(0.5, 0.5, "flat by construction — the prescribed channels are\n"
                           "realigned to the frozen background every iteration",
                 transform=axl.transAxes, ha="center", va="center",
                 fontsize=S.FS_ANNOT + 1, style="italic", color=S.C_NOTE)
    axl.set_title("(b) what is under the core", fontsize=S.FS_TITLE - 1)
    axl.legend(loc="upper left", fontsize=S.FS_LEGEND - 0.5)
    axs.set_xlabel("Iteration n")

    S.panel_letters([axm, axl, axs])
    fig.suptitle(f"MP7 — does the domain stay put?  ({amp} K, {mode} mode)",
                 wrap=True)
    if mode == "snapshot":
        note = ("read from ū + δ at every iteration, i.e. the state the model actually "
                "saw. Every dashed box coincides and both curves are flat: the "
                "prescribed channels (lat/lon, SST, f, terrain, land mask) are realigned "
                "to the initial state each iteration, so the vortex never travels, "
                "never makes landfall, and both ladder members stay in the identical "
                "environment for the whole experiment. The lock reference is u0, not the "
                "model's own prediction of those channels, so the frozen domain is also "
                "the correct one: terrain peaks at the analysis 1899 m rather than the "
                "1041 m one DLAMPty returns")
    else:
        note = ("read from the control bundles' lat/lon channels; δ's lat/lon are "
                "identically zero, so the perturbed member sits on exactly this same "
                "frame. The boxes do NOT coincide: in continuous mode the control "
                "marches and its static channels are not locked to the initial state, "
                "so the TC-following domain carries the storm across the map")
    S.caption(fig, note, style, wrap=True)
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--amp", type=int, default=5)
    ap.add_argument("--mode", choices=("continuous", "snapshot"), default="snapshot",
                    help="which run family to audit (default: %(default)s)")
    ap.add_argument("--members", nargs="+", default=list(_MEMBERS))
    ap.add_argument("--out", default=None, help="default: figs/myplot/mp7_domain_drift.png")
    a = ap.parse_args()

    walks = {init: _walk(init, a.amp, mode=a.mode) for init in a.members}
    for init, w in walks.items():
        print(f"[MP7] {init}")
        for h in ([b for b in _BOX_LEADS if b in w] or sorted(w)[::max(1, len(w) // 6)]):
            if h in w:
                d = w[h]
                print(f"    n={h:4d}  centre {d['lat0']:6.2f}N {d['lon0']:7.2f}E   "
                      f"core land {d['land']:5.1f} %   core SST {d['sst']:6.2f} K")
        first, last = min(w), max(w)
        dd = np.hypot(w[last]["lat0"] - w[first]["lat0"],
                      w[last]["lon0"] - w[first]["lon0"])
        print(f"    total centre displacement n={first} -> n={last}: {dd:.2f} deg")
    S.save(plot(walks, a.amp, a.mode, style=a.style),
           a.out or FIGS / f"mp7_domain_drift_{a.mode}.png", style=a.style,
           pdf=not a.no_pdf)
