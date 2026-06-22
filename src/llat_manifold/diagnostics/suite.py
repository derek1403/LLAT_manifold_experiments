"""Run the standard diagnostic plot set for one experiment run.

Called by ``scripts/run_experiment.py`` after the model run. Produces, into the run's
``plots/`` folder, the high-value figures that apply.

For the perturbation modes (snapshot/continuous) the driver also saves the background
ū / control trajectory, so diagnostics can form **physical** fields:
  * PV–θ and divergence–θ are drawn as Δ-fields  (PV(ū+δ) − PV(ū));
  * the 2D field grid, wind-balance and wind-circulation use the absolute state ū+δ
    (which also avoids degenerate lat/lon coords on a pure δ).
For ``forward`` the bundles are already absolute states, so no baseline is needed.
The radius–time Hovmöller always uses the δ field (it is about the perturbation's
outward propagation).
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import io
from . import load_stamp
from . import pv, divergence, fields, waves, wind_profile, hydrostatic


def _key(p: Path) -> int:
    m = re.search(r"(?:lead|iter)(\d+)", p.name)
    return int(m.group(1)) if m else -1


def _resolve_baseline(data: Path, rep: Path, mode: str):
    """Background/control bundle to turn a δ bundle into physical fields (or None)."""
    if mode == "snapshot":
        bg = data / "background.npz"
        return bg if bg.exists() else None
    if mode == "continuous":
        ctrl = data / rep.name.replace("delta_", "control_")
        return ctrl if ctrl.exists() else None
    return None  # forward: bundles are already absolute states


def _absolute_bundle(rep: Path, baseline, data: Path):
    """Write (and return) an absolute bundle ū+δ for the full-field plots; else rep."""
    if baseline is None:
        return rep
    up, sfc = io.load_delta_bundle(rep)
    bup, bsfc = io.load_delta_bundle(baseline)
    out = data / f"_abs_{rep.stem}.npz"
    io.save_delta_bundle(out, up + bup, sfc + bsfc)
    return out


def standard_plots(run_dir) -> list[str]:
    """Generate the standard figures for ``run_dir``; return the files written."""
    run_dir = Path(run_dir)
    data = run_dir / "data"
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    bundles = sorted(data.glob("delta_*.npz"), key=_key)
    if not bundles:
        print(f"[suite] no δ bundles in {data}")
        return []
    cfg = load_stamp(run_dir).get("resolved_config", {})
    mode = cfg.get("mode", "")
    rep = bundles[-1]                       # representative: last lead / iteration
    baseline = _resolve_baseline(data, rep, mode)
    abs_rep = _absolute_bundle(rep, baseline, data)
    add = baseline is not None
    produced: list[str] = []

    def _try(fn, out_name, *a, **k):
        out = plots / out_name
        try:
            fn(*a, out_png=str(out), **k)
            produced.append(str(out))
        except Exception as e:                       # noqa: BLE001
            print(f"[suite] {out_name} skipped: {type(e).__name__}: {e}")

    # PV / divergence: Δ-fields when a baseline exists, else absolute.
    _try(pv.plot_pv_theta_cross_section, "PV_Theta_tengential.png", rep,
         baseline_path=baseline, add_to_baseline=add)
    _try(divergence.plot_div_theta_cross_section, "div_Theta_uv.png", rep,
         baseline_path=baseline, add_to_baseline=add)
    # Full-field plots use the absolute state (ū+δ or the forward state).
    _try(pv.plot_wind_circulation_cross_section, "wind_circulation.png", abs_rep)
    _try(wind_profile.plot_wind_balance_profile, "wind_balance_profile.png", abs_rep)
    _try(fields.plot_fields, "fields.png", abs_rep)
    # Non-hydrostatic balance checks (each also emits a *_xsection.png).
    _try(hydrostatic.plot_nonhydrostatic_epsilon, "hydrostatic_eps_profile.png", abs_rep)
    _try(hydrostatic.plot_hydrostatic_thermo, "hydrostatic_thermo_profile.png", abs_rep)
    # Gravity-wave propagation uses the δ field directly.
    if len(bundles) >= 3 and mode in ("continuous", "forward"):
        _try(waves.hovmoller, "hovmoller_msl.png", data, surface_var="msl")

    # Per-iteration/lead evolution: frames + GIFs + growth curves over ALL bundles.
    # Imported here (not at module top) to avoid a circular import — evolution reuses
    # this module's bundle helpers. Never let it break the run.
    if len(bundles) >= 2:
        try:
            from . import evolution
            produced.extend(evolution.evolution_plots(run_dir))
        except Exception as e:                       # noqa: BLE001
            print(f"[suite] evolution skipped: {type(e).__name__}: {e}")

    print(f"[suite] produced {len(produced)} figure(s) in {plots}")
    return produced
