"""LR3 — identical heating into six vortices that differ only in intensity.

``fig_LR2_confinement`` shows the strong RAGASA stage confining the injected warmth
more than the weak one, in the direction the local Rossby radius requires. It cannot
show *why*: those two states also differ in SST, shear, humidity, translation speed,
and for 0920 a domain holding ~6 % land and 1.9 km of terrain.

Here the environment is one fixed quiescent background and the only thing that
changes between members is the vortex's peak wind (``scripts/run_lr_sweep.sh``):

    vmax  0 → 50 m/s     L_R_core  3491 → 391 km

which brackets the real pair (weak 1346 km, strong 364 km). Two branches per vortex:
``moist`` (q free) and ``qlock`` (δq ≡ 0).

**What this sweep does and does not settle.** It settles the *energy* question
cleanly and monotonically: the more intense the vortex, the less of the injected heat
survives in the column, and the q-locked branch does the same thing, so the disposal
route is dynamical rather than moist. It does **not** settle the confinement scaling,
for a reason panel (c) makes visible — after the forcing stops the anomaly collapses
toward zero at *every* intensity, so there is no post-forcing balanced state whose
radius could be measured, and the hour-24 radius is contaminated by the injection
still in progress (a member that retains nothing simply shows the injected Gaussian
and scores as "confined"). The confinement evidence therefore rests on the real and
axisymmetric RAGASA pairs in LR2, not on this sweep.

  (a) retained sensible fraction at hour 24 vs vmax, both branches — the clean result;
  (b) the control vortex's own core ascent vs vmax — the proposed disposal mechanism;
  (c) retention vs lead: the post-forcing collapse that blocks a confinement read;
  (d) r50 and core fraction vs vmax, shown with that caveat attached.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import matplotlib.pyplot as plt

import _data as D
import style as S

CATEGORY = "diabatic_heating_lrsweep"
_BAND = (400.0, 700.0)
_CORE_KM = 278.0
# branch -> (label, run-tag prefix, linestyle, marker, colour). The q-locked family
# keeps run_amp_sweep's standard "sweepq" prefix, not "sweep".
_BRANCH = {
    "moist": ("inject ΔT, q free", "sweep", "-", "o", S.C_STRONG),
    "qlock": ("inject ΔT, δq ≡ 0", "sweepq", "--", "s", S.C_LOW),
}
_LEADS = (12, 24, 30, 36, 42, 48)


def _members(amp_tag: str = "5K", steps_h: int = 48):
    """[(vmax, branch, run_dir)] discovered from the sweep category."""
    root = D.REPO / "outputs" / CATEGORY
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found — run the sweep first:\n  bash scripts/run_lr_sweep.sh")
    out = []
    for fam in sorted(root.iterdir()):
        m = re.match(r"v(\d+)_rmw(\w+?)_(moist|qlock)$", fam.name)
        if not m:
            continue
        branch = m.group(3)
        run = fam / (f"{_BRANCH[branch][1]}_{amp_tag}_{steps_h}h_init{D.WEAK}")
        if run.exists():
            out.append((int(m.group(1)), branch, run))
    if not out:
        raise FileNotFoundError(f"no {amp_tag}/{steps_h}h runs under {root}")
    return sorted(out)


def _control_ascent(run, lead_hr):
    """Core-mean −ω over the heated band of the *control* state [Pa s⁻¹].

    How hard the vortex's own secondary circulation is overturning the air the
    anomaly was injected into — the candidate mechanism for the retention decline.
    """
    from llat_manifold import io, layout
    from llat_manifold.diagnostics.response import _mass_weights, _grid_frame, _lead_paths
    _d, c = _lead_paths(run, lead_hr)
    up, sfc = io.load_delta_bundle(c)
    p = np.asarray(layout.pressure_levels(), dtype=float)[: up.shape[0]]
    band = (p >= _BAND[0]) & (p <= _BAND[1])
    dm = _mass_weights(up.shape[0])
    r2d, _area = _grid_frame(sfc)
    w = np.average(up[..., layout.upper_index("w")][band], axis=0, weights=dm[band])
    return float(-w[r2d <= _CORE_KM].mean())


def plot(lead_hr: int = 24, amp_tag: str = "5K", style: str = "note"):
    from llat_manifold.diagnostics.response import confinement_radii, energy_series
    from llat_manifold.diagnostics import waves
    from llat_manifold import io
    from llat_manifold.diagnostics.response import _continuous_pairs

    S.apply()
    rows = []
    for vmax, branch, run in _members(amp_tag):
        s = energy_series(str(run))
        inj = s["e_inj"][s["hour"].index(lead_hr)]
        m = confinement_radii(str(run), lead_hr, p_band=_BAND)
        _l, _d, c = _continuous_pairs(run)[0]
        up, sfc = io.load_delta_bundle(c)
        rows.append({
            "vmax": vmax, "branch": branch,
            "L_R": waves.rossby_radius_summary(up, sfc)["L_R_core_km"],
            "ret": s["e_sens"][s["hour"].index(lead_hr)] / inj,
            "ret_tot": (s["e_sens"][s["hour"].index(lead_hr)]
                        + s["e_lat"][s["hour"].index(lead_hr)]) / inj,
            "ret_series": [s["e_sens"][s["hour"].index(h)] / inj for h in _LEADS],
            "w_core": _control_ascent(run, lead_hr),
            **m})

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.8))
    (axA, axB), (axC, axD) = axes

    for branch, (lab, _tag, ls, mk, col) in _BRANCH.items():
        sel = [r for r in rows if r["branch"] == branch]
        if not sel:
            continue
        v = [r["vmax"] for r in sel]
        axA.plot(v, [r["ret"] for r in sel], ls, marker=mk, color=col, lw=2.1,
                 ms=8, mec="k", mew=0.7, label=f"{lab} — sensible")
        if branch == "moist":
            axA.plot(v, [r["ret_tot"] for r in sel], ":", marker="^", color=S.C_LAT,
                     lw=1.8, ms=7, mec="k", mew=0.6,
                     label="inject ΔT, q free — sensible + latent")
            axB.plot(v, [r["w_core"] for r in sel], "-", marker="D", color=S.C_TOTAL,
                     lw=2.1, ms=7, mec="k", mew=0.6)
        for r in sel:
            axC.plot(_LEADS, r["ret_series"], ls, color=col, lw=1.0, alpha=0.55)
        axD.plot(v, [r["r50"] for r in sel], ls, marker=mk, color=col, lw=2.0,
                 ms=8, mec="k", mew=0.7, label=f"{lab} — r50")

    axA.set_xlabel("vortex peak wind $v_{max}$  [m s$^{-1}$]")
    axA.set_ylabel("retained fraction of the injected energy")
    axA.set_title(f"the stronger the vortex, the less survives (hour {lead_hr})")
    axA.legend(loc="upper right", fontsize=S.FS_LEGEND - 1.0)
    S.zero_line(axA)

    axB.set_xlabel("vortex peak wind $v_{max}$  [m s$^{-1}$]")
    axB.set_ylabel("core-mean $-\\omega$, control, 400–700 hPa  [Pa s$^{-1}$]")
    axB.set_title("the proposed disposal route: the vortex's own overturning")
    S.zero_line(axB)

    S.forcing_span(axC, 24)
    axC.set_xlabel("lead [h]")
    axC.set_ylabel("retained sensible fraction")
    axC.set_title("after the forcing stops every member collapses\n"
                  "(so no post-forcing radius can be measured here)")
    S.zero_line(axC)

    axD.set_xlabel("vortex peak wind $v_{max}$  [m s$^{-1}$]")
    axD.set_ylabel("r50  [km]")
    axD.set_title("r50 is NOT monotone in this sweep — see (c);\n"
                  "the confinement evidence is the RAGASA pairs in LR2")
    axD.legend(loc="upper left", fontsize=S.FS_LEGEND - 1.0)

    S.panel_letters(axes)
    fig.suptitle("LR3 — identical heating into vortices that differ only in intensity "
                 f"({amp_tag} runs)\none fixed quiescent background · balanced vortex, "
                 "RMW 1.0° · no terrain, no land, no shear, no translation · "
                 "$L_R$ 3491 → 391 km")
    S.caption(fig, "the energy result is clean and monotone and survives δq ≡ 0, so the "
                   "disposal is dynamical, not moist; the radius result is not readable "
                   "from this sweep and is not claimed from it", style)
    fig.tight_layout()

    print(f"\n{'vmax':>4s} {'branch':>7s} {'L_R':>6s} {'ret_s':>7s} {'ret_tot':>8s} "
          f"{'-w_core':>8s} {'r50':>6s} {'corefrac':>9s} {'dT_band':>8s} "
          f"{'dT_all':>7s} {'@p':>5s}")
    for r in rows:
        print(f"{r['vmax']:4d} {r['branch']:>7s} {r['L_R']:6.0f} {r['ret']:7.3f} "
              f"{r['ret_tot']:8.3f} {r['w_core']:8.4f} {r['r50']:6.0f} "
              f"{r['core_frac']:9.3f} {r['dT_band_max']:8.3f} {r['dT_all_max']:7.3f} "
              f"{r['dT_all_max_p']:5.0f}")
    return fig


if __name__ == "__main__":
    ap = S.add_style_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--lead", type=int, default=24)
    ap.add_argument("--amp", default="5K", help="amplitude tag: 5K (default) or 2K")
    ap.add_argument("--out", default=None,
                    help="default: figs/lr/lr3_intensity_sweep_<amp>.png")
    a = ap.parse_args()
    out = a.out or (D.FIGS_ROOT / "lr" / f"lr3_intensity_sweep_{a.amp}.png")
    S.save(plot(a.lead, a.amp, style=a.style), out, style=a.style, pdf=not a.no_pdf)
