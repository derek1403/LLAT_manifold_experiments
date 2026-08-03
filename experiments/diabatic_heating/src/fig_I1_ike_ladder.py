"""I1 — the integrated-kinetic-energy response of the ladder: ΔIKE, not ΔPV.

Every other figure in this set reduces the response to a ΔPV dipole scalar. PV is the
right currency for "did the model generate vorticity where heating theory says it must",
but it is not the quantity a forecaster is asked about. **IKE is.** Ragasa's headline was
a record integrated kinetic energy, and IKE is an *outer-core* quantity — it is dominated
by the broad wind field well outside the RMW, not by the peak wind. So this figure asks
the same intervention questions in the units of the problem:

  (a) ΔIKE(total) vs lead — heating with moisture free (solid) against δq ≡ 0 (dashed).
      If the kinetic-energy response were dry-dynamical the two would coincide.
  (b) the inner (r < 200 km) / outer (r ≥ 200 km) split of the moist response. The outer
      partition is the seminar's actual subject: heating injected inside 2σ = 278 km has
      to be *communicated outward* by the secondary circulation before it becomes IKE.
  (c) ΔIKE at nominal hour 24 (end of injection) and hour 120 (after five days of free
      evolution) against peak V_t, with the δq-only reverse probe overlaid — the
      intensity dependence, and how much of it survives without moisture.

The ΔIKE definition is *not* re-implemented here: ``diagnostics.ike`` wraps Part-1's
``regional_couple.diagnostics.ike`` verbatim (storm-centred polar transform, azimuthal
mean, 0.5·ρ·V_t²·2πr·H·dr over 5 km annuli, ρ=H=1 → TJ), which is the same definition as
the operational dashboard *and* as the DeepTC_Radial-ai training labels. That is what lets
a number on this figure be compared with a number on those.

Main line is **0920 (weak) vs 0921 (strong)** — the controlled weak/strong contrast, both
vortices in 0920's fixed environment. 0922 is drawn thin and grey because a storm at its
intensity ceiling raises a saturation question that is not the weak-vs-strong question;
``--no-vsev`` drops it for the main-line slide, ``--with-0917`` adds the uncontrolled
developing-stage member for range.

Iron rule, unchanged: the x axis is iteration n in **nominal** hours (one model step =
nominal 3 h). It is not physical time and must not be read against an observed clock.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _data as D
import style as S

from llat_manifold import io
from llat_manifold.diagnostics import load_stamp
from llat_manifold.diagnostics.ike import (
    _INNER_KM,
    _baseline_path,
    _components,
    _ike_profile,
    _key,
)

# Families this figure reads, in plot order: the baseline, the moisture denial, and the
# reverse probe. Tag stems are the 5 K / 5-day time series of each family.
_TSERIES = {
    "heating_moist": "tseries_5K_120h_init{init}",
    "heating_qlock": "tseriesq_5K_120h_init{init}",
    "dq_measured": "tseriesdq_5K_120h_init{init}",
}
_LABEL = {
    "heating_moist": "heating ΔT, q free",
    "heating_qlock": "heating ΔT, δq = 0",
    "dq_measured": "δq only (θ̇ = 0)",
}

_FORCING_END_H = 24.0     # 8 steps × nominal 3 h


def ike_series(run_dir: Path):
    """(lead_hr, ΔIKE_total, ΔIKE_inner, ΔIKE_outer) for one run, all in TJ.

    ΔIKE_k = IKE(ū + δ_k) − IKE(ū_k): the perturbed absolute state against its *own*
    control at the same lead, so the storm's natural evolution cancels and what is left
    is the response to the injection alone.
    """
    run_dir = Path(run_dir)
    data = run_dir / "data"
    mode = load_stamp(run_dir).get("resolved_config", {}).get("mode", "")
    leads, tot, inn, out = [], [], [], []
    for b in sorted(data.glob("delta_*.npz"), key=_key):
        base = _baseline_path(b, data, mode)
        if base is None:
            continue
        _, sfc_d = io.load_delta_bundle(b)
        _, sfc_b = io.load_delta_bundle(base)
        r, prof_p = _ike_profile(sfc_d + sfc_b)
        r, prof_b = _ike_profile(sfc_b)
        tp, ip, op = _components(r, prof_p, _INNER_KM)
        tb, ib, ob = _components(r, prof_b, _INNER_KM)
        leads.append(_key(b))
        tot.append(tp - tb)
        inn.append(ip - ib)
        out.append(op - ob)
    if not leads:
        raise RuntimeError(f"no ΔIKE points in {run_dir} (mode={mode!r})")
    order = np.argsort(leads)
    arr = lambda v: np.asarray(v, float)[order]      # noqa: E731
    return arr(leads), arr(tot), arr(inn), arr(out)


def _one(args):
    """Worker: (family, init, run_dir) -> (family, init, series). Top level, so picklable."""
    fam, init, rd = args
    return fam, init, ike_series(rd)


def _collect(inits, ic, *, cache: Path | None = None, recompute: bool = False,
             workers: int = 9):
    """{family: {init: (lead, tot, inn, out)}} — every series this figure needs.

    Each ``_ike_profile`` call costs ~5 s (the Part-1 polar transform, reused verbatim and
    deliberately not optimised), and this figure needs ~960 of them — so the result is
    cached to ``cache`` and the 9 (family, init) pairs are computed in parallel. Delete the
    cache, or pass ``--recompute``, after any run is regenerated.
    """
    if cache is not None and cache.exists() and not recompute:
        z = np.load(cache, allow_pickle=False)
        out: dict = {}
        for key in z.files:
            fam, init, field = key.rsplit("|", 2)
            out.setdefault(fam, {}).setdefault(init, {})[field] = z[key]
        packed = {f: {i: (d["lead"], d["tot"], d["inn"], d["out"])
                      for i, d in v.items()} for f, v in out.items()}
        print(f"[i1] loaded cached series from {cache}")
        return packed

    jobs = []
    for fam, stem in _TSERIES.items():
        for init in inits:
            try:
                jobs.append((fam, init, D.run(fam, stem.format(init=init), ic=ic)))
            except FileNotFoundError:
                print(f"[i1] missing {fam}/{stem.format(init=init)} — skipped")

    out: dict = {f: {} for f in _TSERIES}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for fam, init, series in ex.map(_one, jobs):
            out[fam][init] = series
            print(f"[i1] {fam:14s} {init}  ΔIKE(24h) = "
                  f"{np.interp(24, series[0], series[1]):+7.2f} TJ")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        flat = {f"{fam}|{init}|{name}": arr
                for fam, per in out.items() for init, s in per.items()
                for name, arr in zip(("lead", "tot", "inn", "out"), s)}
        np.savez_compressed(cache, **flat)
        print(f"[i1] cached series to {cache}")
    return out


def _vmax_profiles(inits):
    """{init: (vmax, rmw)} from each member's own axisymmetric IC."""
    import sys
    sys.path.insert(0, str(D.REPO / "src"))
    from llat_manifold.idealized_vortex import axisymmetric as ax, background as bg
    prof = {}
    for init in inits:
        p = D.axisym_ic_path(init)
        if not Path(p).exists():
            continue
        st = bg.load_background(str(p))
        r, vt = ax.tangential_profile(st.upper, st.surface, 850)
        i = int(np.nanargmax(vt))
        prof[init] = (float(vt[i]), float(r[i]))
    return prof


def _window_mean(series, lo, hi, field=1):
    """Mean of a ΔIKE component over a nominal-hour window.

    A window mean, not a single-lead read: the free-evolution curves oscillate by a few TJ
    from step to step, so any one lead is a coin flip about where in that oscillation you
    land. Averaging over the window is what the comparison actually rests on.
    """
    lead = series[0]
    m = (lead >= lo) & (lead <= hi)
    return float(np.nanmean(series[field][m])) if m.any() else np.nan


def plot(*, ic: str = "axisym", with_vsev: bool = True, with_weak: bool = False,
         style: str = "note", cache: Path | None = None, recompute: bool = False):
    if ic != "axisym":
        raise SystemExit("I1 is an axisymmetric-ladder figure; use --ic axisym")
    S.apply()

    main = [D.STRONG, D.SEV]                       # the weak/strong contrast
    inits = ([D.WEAK] if with_weak else []) + main + ([D.VSEV] if with_vsev else [])
    series = _collect(inits, ic, cache=cache, recompute=recompute)
    series = {f: {i: s for i, s in per.items() if i in inits}
              for f, per in series.items()}
    prof = _vmax_profiles(inits)
    color = S.case_colors(inits)
    # 0922 (and 0917) are context, not the argument: thin, grey, low z-order.
    context = {D.VSEV, D.WEAK}
    styling = {i: (dict(lw=1.3, alpha=0.55, zorder=2, color="#8C8C8C")
                   if i in context else
                   dict(lw=2.6, alpha=1.0, zorder=4, color=color[i]))
               for i in inits}

    have = [i for i in inits if i in series["heating_moist"]]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # (a) total ΔIKE vs lead: moisture free vs denied -------------------------
    for init in have:
        st = styling[init]
        lead, tot, _, _ = series["heating_moist"][init]
        axA.plot(lead, tot, ls="-", label=f"{D.CASE[init]} — q free", **st)
        if init in series["heating_qlock"]:
            lead_q, tot_q, _, _ = series["heating_qlock"][init]
            axA.plot(lead_q, tot_q, ls="--", label=f"{D.CASE[init]} — δq = 0", **st)
    S.zero_line(axA)
    S.forcing_span(axA, _FORCING_END_H, label=False)   # label collides with the axes title
    axA.annotate("injection", (_FORCING_END_H / 2, 0.03), xycoords=axA.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=S.FS_ANNOT, style="italic",
                 color=S.C_NOTE)
    axA.set_xlabel("iteration  [nominal hours, 3 h per step]")
    axA.set_ylabel("ΔIKE  (perturbed − control)  [TJ]")
    axA.set_title("(a) kinetic-energy response, and what moisture carries")
    axA.legend(loc="upper left", fontsize=S.FS_LEGEND - 1)

    # (b) inner / outer split of the moist response ---------------------------
    for init in have:
        st = styling[init]
        lead, _, inn, out = series["heating_moist"][init]
        axB.plot(lead, out, ls="-", label=f"{D.CASE[init]} — outer (r ≥ 200 km)", **st)
        axB.plot(lead, inn, ls=":", label=f"{D.CASE[init]} — inner (r < 200 km)", **st)
    S.zero_line(axB)
    S.forcing_span(axB, _FORCING_END_H, label=False)
    axB.set_xlabel("iteration  [nominal hours, 3 h per step]")
    axB.set_ylabel("ΔIKE by partition  [TJ]")
    axB.set_title("(b) where the energy ends up — outer core vs inner core")
    axB.legend(loc="upper left", fontsize=S.FS_LEGEND - 1)

    # (c) intensity dependence, as window means -------------------------------
    # Two windows rather than two leads: during the injection (nothing happens) and over
    # the post-forcing plateau (where the takeoff lives). Single leads are too noisy.
    cinits = [i for i in have if i in prof]
    xs = np.array([prof[i][0] for i in cinits])
    fam_color = {"heating_moist": S.C_TOTAL, "heating_qlock": S.C_SENS,
                 "dq_measured": S.C_LAT}
    windows = ((_FORCING_END_H + 36, 120, "-", "o", 8, 1.0, "post-forcing (h60–120)"),
               (0, _FORCING_END_H, ":", "o", 6, 0.5, "during injection (h0–24)"))
    for fam, c in fam_color.items():
        for lo, hi, ls, mk, ms, alpha, wlab in windows:
            ys = [_window_mean(series[fam][i], lo, hi) if i in series[fam] else np.nan
                  for i in cinits]
            axC.plot(xs, ys, ls=ls, marker=mk, color=c, lw=2.2, ms=ms, alpha=alpha,
                     mfc=c if alpha > 0.8 else "white",
                     label=f"{_LABEL[fam]} — {wlab}")
    for i, x in zip(cinits, xs):
        axC.annotate(D.CASE[i].split()[0],
                     (x, _window_mean(series["heating_moist"][i],
                                      _FORCING_END_H + 36, 120)),
                     xytext=(0, 10), textcoords="offset points", ha="center",
                     fontsize=S.FS_ANNOT, color=S.C_NOTE)
    S.zero_line(axC)
    axC.set_xlabel("peak azimuthal-mean $V_t$ at 850 hPa  [m/s]")
    axC.set_ylabel("ΔIKE total, window mean  [TJ]")
    axC.set_title("(c) intensity dependence of the kinetic-energy gain")
    axC.legend(loc="upper left", fontsize=S.FS_LEGEND - 2.5)

    S.panel_letters([axA, axB, axC])
    fig.suptitle("I1 — integrated kinetic energy response to a 5 K heating injection, "
                 "weak (0920) vs strong (0921) in one fixed environment")
    S.caption(fig, "ΔIKE on the Part-1 dashboard definition (azimuthal-mean 10 m V_t, "
                   "5 km annuli). 0920/0921/0922 share 0920's environment, so only the "
                   "vortex differs; 0922 is drawn as context because a storm at its "
                   "ceiling is a saturation question, not a weak/strong one. Iteration n "
                   "is a nominal-hour count, not physical time", style)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = D.add_data_args(
        S.add_style_args(argparse.ArgumentParser(description=__doc__)))
    ap.add_argument("--no-vsev", action="store_true",
                    help="drop 0922 entirely (main-line slide version)")
    ap.add_argument("--with-0917", action="store_true",
                    help="add the uncontrolled developing-stage member for range")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore the cached ΔIKE series and recompute (~10 min, 9 workers)")
    ap.add_argument("--out", default=None,
                    help="default: figs/<ic>/ike/i1_ike_ladder.png")
    a = ap.parse_args()
    ike_dir = D.FIGS_ROOT / a.ic / "ike"
    out = a.out or (ike_dir / "i1_ike_ladder.png")
    S.save(plot(ic=a.ic, with_vsev=not a.no_vsev, with_weak=a.with_0917,
                style=a.style, cache=ike_dir / "_i1_series_cache.npz",
                recompute=a.recompute),
           out, style=a.style, pdf=not a.no_pdf)
