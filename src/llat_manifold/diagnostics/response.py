"""Forcing–response curves: ΔPV vs heating amplitude, and ΔPV vs iteration vs theory.

Port target: the ``Heating_vs_DeltaP_Response`` / ``RatePrecip_vs_DeltaP_Response``
figures under ``idealized_exp/pic/`` and ``plot_vorticity.py``. Collects a scalar
response across a sweep of perturbation amplitudes and plots response vs forcing —
the model's sensitivity curve, the most direct probe of how the LLAT manifold reacts
to a given physical intervention.

Two figures, both on continuous-mode runs:

* **Amplitude sweep** (:func:`plot_amplitude_sweep`) — a ΔPV scalar at a fixed lead
  vs ``amp_K``, one curve per init time, each point annotated with *where* the
  extremum sits (pressure level, radius from centre), with a linear reference through
  the origin (slope pinned by the smallest amplitude). Departure from that line is
  the O(‖δ‖²) manifold-curvature signal of ``docs/perturbation_method.md`` §5.
* **ΔPV series vs theory** (:func:`plot_pv_timeseries`) — model ΔPV extrema per
  iteration against the accumulated diabatic-source expectation. The Ertel-PV
  source at leading order (Haynes–McIntyre) is a *rate* equation in the material
  heating rate θ̇ = Dθ/Dt:

      D(PV)/Dt = PV · ∂θ̇/∂θ   ⇒   ΔPV_theory(n) = Σ_{i≤n_forced} PV_ctrl·(∂Δθ_step/∂p)/(∂θ_ctrl/∂p)

  accumulated per forced step on the control state (θ̇·Δt = Δθ_step, so Δt cancels)
  and reduced with the same dipole metric as the model each lead. After the forcing
  stops the source vanishes and the theory line is **flat** (material PV
  conservation); the model's later drift is itself the conservation check. Model
  above theory = the manifold generates/holds more ΔPV than the injected heating
  supports; below = it has dissipated or exported it.

The ΔPV scalars are the **dipole** the heating must build: PV generation below the
heating maximum (low-level tower) and destruction above it, both within the heated
core (r ≤ 2σ). The x-axis is the iteration count in nominal hours (model steps are
nominal 3 h, not strict physical time).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import io, layout
from . import experiment_title, load_stamp
from ._idealized import CP, RD, calculate_pv_spherical, fields_from_bundle
from ..perturbations.heating import gaussian_centered, vertical_profile

# Dipole search boxes (hPa): generation below the Deep heating max (~600 hPa),
# destruction above it; both restricted to the heated core (r ≤ CORE_FACTOR·σ).
_LOW_P = (700.0, 1000.0)
_UP_P = (200.0, 500.0)
_CORE_FACTOR = 2.0
_KAPPA = RD / CP


def _key(p: Path) -> int:
    m = re.search(r"(?:lead|iter)(\d+)", p.name)
    return int(m.group(1)) if m else -1


def central_pressure_drop(delta_path) -> float:
    """Minimum (most negative) MSL perturbation in a delta bundle, in the field's units.

    A proxy for intensification: how much the storm's central pressure dropped due to
    the perturbation. Uses the DLAMPty surface ``msl`` channel.
    """
    _, dsfc = io.load_delta_bundle(delta_path)
    return float(np.min(dsfc[:, :, layout.surface_index("msl")]))


# --------------------------------------------------------------------------- #
# Per-bundle analysis: ΔPV field, dipole extrema + their locations
# --------------------------------------------------------------------------- #
def _pv_of(upper, sfc) -> np.ndarray:
    u, v, t, _z, _w, f, lat2d, lon2d, p_hpa = fields_from_bundle(upper, sfc)
    return calculate_pv_spherical(u, v, t, f, p_hpa, lat2d, lon2d)


def delta_pv_field(delta_path, baseline_path) -> np.ndarray:
    """ΔPV = PV(control+δ) − PV(control) on the 13 isobaric levels, in PVU.

    Both upper *and* surface deltas are added to the baseline: the surface bundle
    carries the Coriolis and lat/lon channels the PV metric needs.
    """
    d_up, d_sfc = io.load_delta_bundle(delta_path)
    b_up, b_sfc = io.load_delta_bundle(baseline_path)
    return _pv_of(b_up + d_up, b_sfc + d_sfc) - _pv_of(b_up, b_sfc)


def _core_mask(ny: int, nx: int, sigma: float, factor: float = _CORE_FACTOR):
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    r2 = (yy - ny // 2) ** 2 + (xx - nx // 2) ** 2
    return r2 <= (factor * sigma) ** 2


def _extremum(dpv, lev_mask, core, mode):
    """(value, (k,j,i)) of the max/min of dpv over (lev_mask levels × core points)."""
    sub = np.where(core[None, :, :], dpv, np.nan)
    sub = np.where(lev_mask[:, None, None], sub, np.nan)
    flat = np.nanargmax(sub) if mode == "max" else np.nanargmin(sub)
    kji = np.unravel_index(flat, dpv.shape)
    return float(dpv[kji]), kji


def _radius_km(j, i, lat2d, lon2d) -> float:
    cy, cx = lat2d.shape[0] // 2, lat2d.shape[1] // 2
    dy = (lat2d[j, i] - lat2d[cy, cx]) * 111.32
    dx = (lon2d[j, i] - lon2d[cy, cx]) * 111.32 * np.cos(np.deg2rad(lat2d[cy, cx]))
    return float(np.hypot(dy, dx))


def analyze_pair(delta_path, control_path, *, sigma: float = 5.0):
    """Full analysis of one δ/control pair.

    Returns ``(metrics, aux)``. ``metrics`` holds the dipole scalars and where they
    sit: ``lowlevel_max``/``upperlevel_min`` [PVU], ``*_p`` [hPa], ``*_r`` [km],
    ``*_kji`` (grid indices), and ``center_850``. ``aux`` carries the fields needed
    by the theory evaluation (perturbed PV, perturbed T, pressure levels).
    """
    d_up, d_sfc = io.load_delta_bundle(delta_path)
    c_up, c_sfc = io.load_delta_bundle(control_path)
    up, sfc = c_up + d_up, c_sfc + d_sfc
    u, v, t, _z, _w, f, lat2d, lon2d, p_hpa = fields_from_bundle(up, sfc)
    pv_pert = calculate_pv_spherical(u, v, t, f, p_hpa, lat2d, lon2d)
    cu, cv, ct, _cz, _cw, cf, _clat, _clon, _cp = fields_from_bundle(c_up, c_sfc)
    pv_ctrl = calculate_pv_spherical(cu, cv, ct, cf, p_hpa, lat2d, lon2d)
    dpv = pv_pert - pv_ctrl

    p = np.asarray(p_hpa, dtype=float)[: dpv.shape[0]]
    core = _core_mask(dpv.shape[1], dpv.shape[2], sigma)
    v_low, kji_low = _extremum(dpv, (p >= _LOW_P[0]) & (p <= _LOW_P[1]), core, "max")
    v_up, kji_up = _extremum(dpv, (p >= _UP_P[0]) & (p <= _UP_P[1]), core, "min")
    cy, cx = dpv.shape[1] // 2, dpv.shape[2] // 2
    i850 = int(np.argmin(np.abs(p - 850.0)))

    metrics = {
        "lowlevel_max": v_low,
        "lowlevel_p": float(p[kji_low[0]]),
        "lowlevel_r": _radius_km(kji_low[1], kji_low[2], lat2d, lon2d),
        "lowlevel_kji": kji_low,
        "upperlevel_min": v_up,
        "upperlevel_p": float(p[kji_up[0]]),
        "upperlevel_r": _radius_km(kji_up[1], kji_up[2], lat2d, lon2d),
        "upperlevel_kji": kji_up,
        "center_850": float(dpv[i850, cy, cx]),
    }
    aux = {"pv_pert": pv_pert, "t_pert": t, "pv_ctrl": pv_ctrl, "t_ctrl": ct,
           "p_hpa": p, "dpv": dpv, "z": _z, "lat2d": lat2d, "lon2d": lon2d}
    return metrics, aux


# --------------------------------------------------------------------------- #
# Per-run series: model extrema + state-based theory at the same locations
# --------------------------------------------------------------------------- #
def _continuous_pairs(run_dir: Path):
    """Sorted [(lead_hr, delta_path, control_path)] of a continuous run."""
    data = Path(run_dir) / "data"
    pairs = []
    for d in sorted(data.glob("delta_continuous_*.npz"), key=_key):
        c = data / d.name.replace("delta_", "control_")
        if c.exists():
            pairs.append((_key(d), d, c))
    if not pairs:
        raise FileNotFoundError(f"no continuous delta/control pairs in {data}")
    return pairs


def _pert_params(run_dir) -> dict:
    cfg = load_stamp(run_dir).get("resolved_config", {})
    pert = dict(cfg.get("perturbation", {}) or {})
    if pert.get("type") not in ("heating", "moisture"):
        raise ValueError(f"{run_dir}: not a heating/moisture run ({pert.get('type')!r})")
    pert.setdefault("sigma", 5.0)
    pert.setdefault("amp_mode", "spread")
    pert.setdefault("forcing_steps", 1)
    return pert


def pv_response_series(run_dir) -> dict:
    """Model ΔPV extrema and the accumulated diabatic-source theory per iteration.

    Theory: the Ertel-PV diabatic source at leading order (Haynes–McIntyre),
        D(PV)/Dt = PV · ∂θ̇/∂θ,      θ̇ = Dθ/Dt = the injected heating rate,
    integrated in time on the CONTROL state (the semi-linear expectation):
        ΔPV_th(n) = Σ_{i ≤ min(n, steps)} PV_ctrl(i) · (∂Δθ_step/∂p) / (∂θ_ctrl(i)/∂p)
    (θ̇·Δt = Δθ_step per 3 h step, so Δt cancels). The accumulated 3-D theory field
    gets the SAME dipole reduction as the model ΔPV at every lead. After the forcing
    stops the source is zero and the field is frozen — materially, PV is conserved —
    so the theory line goes flat; the model's later drift is itself the conservation
    check. Eulerian caveat: the frozen field ignores advection of the generated PV.
    """
    pert = _pert_params(run_dir)
    is_heating = pert.get("type") == "heating"     # δq-only runs have no injected Δθ
    amp = float(pert["amp_K"])
    steps = int(pert["forcing_steps"])
    if pert["amp_mode"] == "spread":
        amp /= steps
    sigma = float(pert["sigma"])

    pairs = _continuous_pairs(run_dir)
    up0, _ = io.load_delta_bundle(pairs[0][2])
    nz, ny, _nx = up0.shape[:3]
    p_pa = np.asarray(layout.pressure_levels(), dtype=float)[:nz] * 100.0
    p3 = p_pa[:, None, None]

    if is_heating:
        # Per-step injected Δθ (= θ̇·Δt) and its pressure gradient.
        dT_step = amp * (vertical_profile(pert["heat_type"], p_pa / 100.0)[:, None, None]
                         * gaussian_centered(ny, sigma)[None, :, :])
        ddtheta_dp_step = np.gradient(dT_step * (1.0e5 / p3) ** _KAPPA, p_pa, axis=0)
        th_field = None

    out: dict = {"hour": []}
    for lead, d, c in pairs:
        m, aux = analyze_pair(d, c, sigma=sigma)
        if is_heating:
            if th_field is None:
                th_field = np.zeros_like(aux["pv_ctrl"])
            if lead // 3 <= steps:      # source active: accumulate PV·∂θ̇/∂θ·Δt
                theta_ctrl = aux["t_ctrl"] * (1.0e5 / p3) ** _KAPPA
                dtheta_dp = np.gradient(theta_ctrl, p_pa, axis=0)
                th_field += aux["pv_ctrl"] * ddtheta_dp_step / dtheta_dp
            th_low, th_up, _ = _dpv_reduce(th_field, sigma=sigma)
            m["theory_low"], m["theory_up"] = th_low, th_up
        out["hour"].append(lead)
        for k, v in m.items():
            if not k.endswith("_kji"):
                out.setdefault(k, []).append(v)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
_C_LOW, _C_UP = "tab:blue", "tab:orange"      # dipole metric hues (fixed assignment)
_C_WEAK, _C_STRONG = "tab:blue", "tab:red"    # init-case hues in the sweep figure


def plot_pv_timeseries(run_dir, out_png=None):
    """ΔPV per iteration, model (solid) vs state-based theory (dashed), per pole."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    s = pv_response_series(run_dir)
    pert = _pert_params(run_dir)
    t_end = int(pert["forcing_steps"]) * 3

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="grey", lw=0.8)
    has_theory = "theory_low" in s                 # absent for δq-only (moisture) runs
    if s["hour"][0] < t_end < s["hour"][-1]:
        what = "heating" if has_theory else "forcing"
        ax.axvline(t_end, color="grey", lw=1.0, ls=":",
                   label=f"{what} ends ({t_end} h)")
    ax.plot(s["hour"], s["lowlevel_max"], color=_C_LOW, lw=2.0,
            marker="o", ms=4, label="model  max ΔPV 700–1000 hPa")
    theory_lab = r"theory  $\int \mathrm{PV}\,(\partial\dot\theta/\partial\theta)\,\mathrm{d}t$"
    if has_theory:
        ax.plot(s["hour"], s["theory_low"], color=_C_LOW, lw=2.0, ls="--",
                label=f"{theory_lab}  low pole")
    ax.plot(s["hour"], s["upperlevel_min"], color=_C_UP, lw=2.0,
            marker="s", ms=4, label="model  min ΔPV 200–500 hPa")
    if has_theory:
        ax.plot(s["hour"], s["theory_up"], color=_C_UP, lw=2.0, ls="--",
                label=f"{theory_lab}  upper pole")
    ax.set_xlabel("Iteration n  (nominal hour)", fontsize=13, weight="bold")
    ax.set_ylabel("ΔPV  [PVU]", fontsize=13, weight="bold")
    ax.set_xlim(s["hour"][0], s["hour"][-1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    prefix = "ΔPV vs diabatic theory:" if has_theory else "ΔPV response (δq-only, θ̇=0):"
    ax.set_title(experiment_title(run_dir, prefix=prefix), fontsize=13, weight="bold")
    if has_theory:
        fig.text(0.995, 0.005,
                 r"theory = diabatic PV source $\mathrm{PV}\cdot\partial\dot\theta/\partial\theta$"
                 r" ($\dot\theta$ = injected heating rate) accumulated per forced step on the"
                 " control state, same dipole reduction as the model; flat after forcing ends"
                 " = material conservation — the model's later drift is the conservation check",
                 ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    else:
        fig.text(0.995, 0.005,
                 "no theory line: a δq-only injection has θ̇ = 0, so the dry semi-linear "
                 "expectation is ΔPV ≈ 0 — everything shown is q-channel-routed response",
                 ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def _find_runs(category_dir, pattern: str, family: str | None = None):
    """Run dirs matching ``pattern`` under the category (or one family below).

    Runs are grouped as ``<category>/<family>/<run>``; with ``family=None`` the
    search covers the category root plus every family folder, so tag patterns
    stay usable without knowing the grouping (tag prefixes are disjoint across
    families by convention).
    """
    base = Path(category_dir)
    if family:
        return sorted((base / family).glob(pattern))
    return sorted(list(base.glob(pattern)) + list(base.glob(f"*/{pattern}")))


def _sweep_points(category_dir, lead_hr: int, pattern: str, family: str | None = None):
    """{init_time: [(amp_K, metrics)]} at the requested lead over sweep run dirs."""
    by_init: dict[str, list] = {}
    for run in _find_runs(category_dir, pattern, family):
        data = run / "data"
        hits = sorted(data.glob(f"delta_continuous_*lead{lead_hr:03d}hr.npz"))
        if not hits:
            continue
        d = hits[0]
        c = data / d.name.replace("delta_", "control_")
        if not c.exists():
            continue
        cfg = load_stamp(run).get("resolved_config", {})
        pert = _pert_params(run)
        m, _aux = analyze_pair(d, c, sigma=float(pert["sigma"]))
        by_init.setdefault(str(cfg.get("init_time", "?")), []).append(
            (float(pert["amp_K"]), m))
    if not by_init:
        raise FileNotFoundError(
            f"no sweep runs matching {pattern!r} with lead {lead_hr:03d}h under {category_dir}")
    return {k: sorted(v) for k, v in sorted(by_init.items())}


def _annotate_loc(ax, amps, vals, pts, pole, above):
    """Tiny '(hPa, km)' tag above/below each sweep point."""
    for a, v, (_a, m) in zip(amps, vals, pts):
        ax.annotate(f"{m[f'{pole}_p']:.0f}hPa\n{m[f'{pole}_r']:.0f}km",
                    (a, v), textcoords="offset points",
                    xytext=(0, 7 if above else -7),
                    ha="center", va="bottom" if above else "top",
                    fontsize=5.5, color="dimgray", linespacing=0.9)


def plot_amplitude_sweep(category_dir, out_png=None, *, lead_hr: int = 24,
                         pattern: str = "sweep_*", family: str | None = None):
    """ΔPV dipole scalars at a fixed lead vs heating amplitude, per init time.

    Each point carries a small ``(pressure level, radius from centre)`` tag locating
    its extremum. The dotted grey line through the origin is pinned to each curve's
    smallest amplitude — the semi-linear expectation; departure is manifold curvature.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = _sweep_points(category_dir, lead_hr, pattern, family)
    case_color = {}
    palette = [_C_WEAK, _C_STRONG, "tab:green", "tab:purple"]
    for i, init in enumerate(points):
        case_color[init] = palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axhline(0, color="grey", lw=0.8)
    ref_labeled = False
    for init, pts in points.items():
        amps = np.array([a for a, _ in pts])
        low = np.array([m["lowlevel_max"] for _, m in pts])
        up = np.array([m["upperlevel_min"] for _, m in pts])
        c = case_color[init]
        ax.plot(amps, low, color=c, lw=2.0, marker="o", ms=5,
                label=f"init {init}  max ΔPV 700–1000 hPa")
        ax.plot(amps, up, color=c, lw=2.0, ls="-.", marker="s", ms=5,
                mfc="none", label=f"init {init}  min ΔPV 200–500 hPa")
        _annotate_loc(ax, amps, low, pts, "lowlevel", above=True)
        _annotate_loc(ax, amps, up, pts, "upperlevel", above=False)
        # Semi-linear reference through the origin, slope from the smallest amp.
        xs = np.array([0.0, amps.max()])
        for y in (low, up):
            ax.plot(xs, xs * y[0] / amps[0], color="grey", lw=1.2, ls=":",
                    alpha=0.8, label=("linear scaling (from smallest amp)"
                                      if not ref_labeled else None))
            ref_labeled = True
    ax.set_xlabel("Heating amplitude amp_K  [K]", fontsize=13, weight="bold")
    ax.set_ylabel(f"ΔPV at nominal hour {lead_hr}  [PVU]", fontsize=13, weight="bold")
    ax.set_xlim(left=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"Diabatic heating amplitude sweep — ΔPV response (nominal hour {lead_hr})",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def _azimuthal_mean(dpv, lat2d, lon2d, *, r_max_km=550.0, dr_km=25.0):
    """Azimuthal-mean ΔPV(level, radius) around the domain centre."""
    cy, cx = lat2d.shape[0] // 2, lat2d.shape[1] // 2
    dy = (lat2d - lat2d[cy, cx]) * 111.32
    dx = (lon2d - lon2d[cy, cx]) * 111.32 * np.cos(np.deg2rad(lat2d[cy, cx]))
    r = np.hypot(dy, dx)
    edges = np.arange(0.0, r_max_km + dr_km, dr_km)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full((dpv.shape[0], centers.size), np.nan)
    for b in range(centers.size):
        m = (r >= edges[b]) & (r < edges[b + 1])
        if m.any():
            out[:, b] = dpv[:, m].mean(axis=1)
    return centers, out


def plot_sweep_maps(category_dir, init_time, out_png=None, *,
                    amps=(0.5, 2.0, 4.0, 6.0, 8.0, 10.0), lead_hr: int = 24,
                    per_K: bool = False, pattern: str = "sweep_{amp}_{h}h_init{init}",
                    zoom_deg: float = 5.0):
    """6×3 structure panel: columns = amplitudes, rows = ΔPV views at one lead.

    Row 1: lon–lat ΔPV on each column's own upper-level *min* layer (200–500 hPa);
    Row 2: lon–lat ΔPV on each column's own low-level *max* layer (700–1000 hPa);
    Row 3: azimuthal-mean ΔPV radius–height section. One shared symmetric colour
    scale per row. ``per_K=True`` plots ΔPV/amp_K instead — if the response were
    semi-linear all six columns would look identical, so the column where the shape
    starts to deform is where nonlinearity kicks in.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..perturbations.heating import _amp_tag

    category_dir = Path(category_dir)
    cols = []
    for amp in amps:
        name = pattern.format(amp=_amp_tag(amp), h=lead_hr, init=init_time)
        hits = _find_runs(category_dir, name)
        if not hits:
            raise FileNotFoundError(f"no run named {name!r} under {category_dir} "
                                    f"(searched the category root and its families)")
        run = hits[0]
        d = sorted((run / "data").glob(f"delta_continuous_*lead{lead_hr:03d}hr.npz"))
        if not d:
            raise FileNotFoundError(f"no lead-{lead_hr:03d}h bundle under {run}")
        c = d[0].parent / d[0].name.replace("delta_", "control_")
        pert = _pert_params(run)
        m, aux = analyze_pair(d[0], c, sigma=float(pert["sigma"]))
        scale = 1.0 / amp if per_K else 1.0
        r_km, az = _azimuthal_mean(aux["dpv"], aux["lat2d"], aux["lon2d"])
        cols.append({"amp": amp, "m": m, "dpv": aux["dpv"] * scale,
                     "az": az * scale, "r_km": r_km,
                     "z_km": np.nanmean(aux["z"], axis=(1, 2)) / 1000.0,
                     "lat2d": aux["lat2d"], "lon2d": aux["lon2d"]})

    ny, nx = cols[0]["lat2d"].shape
    cy, cx = ny // 2, nx // 2
    half = int(round(zoom_deg / 0.25))
    ys, xs = slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1)

    def _row_lim(fields):
        return max(float(np.nanmax(np.abs(f))) for f in fields) or 1.0

    lim_up = _row_lim([c["dpv"][c["m"]["upperlevel_kji"][0]][ys, xs] for c in cols])
    lim_low = _row_lim([c["dpv"][c["m"]["lowlevel_kji"][0]][ys, xs] for c in cols])
    lim_az = _row_lim([c["az"] for c in cols])
    unit = "PVU K$^{-1}$" if per_K else "PVU"

    fig, axes = plt.subplots(3, len(cols), figsize=(3.1 * len(cols), 10.5),
                             gridspec_kw={"height_ratios": [1, 1, 1.35]})
    rows = [("upperlevel", "min", lim_up), ("lowlevel", "max", lim_low)]
    for j, col in enumerate(cols):
        lat, lon, m = col["lat2d"], col["lon2d"], col["m"]
        for i, (pole, tag, lim) in enumerate(rows):
            ax = axes[i, j]
            k, pj, pi = m[f"{pole}_kji"]
            im = ax.pcolormesh(lon[ys, xs], lat[ys, xs], col["dpv"][k][ys, xs],
                               cmap="bwr", vmin=-lim, vmax=lim, shading="auto")
            ax.plot(lon[pj, pi], lat[pj, pi], "kx", ms=6, mew=1.5)
            ax.set_title(f"{col['amp']:g} K · {m[f'{pole}_p']:.0f} hPa ({tag})",
                         fontsize=9, weight="bold")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if j:
                ax.set_yticklabels([])
        ax = axes[2, j]
        imz = ax.contourf(col["r_km"], col["z_km"], col["az"],
                          levels=np.linspace(-lim_az, lim_az, 41), cmap="bwr",
                          extend="both")
        ax.set_ylim(0, 15)
        ax.set_title(f"{col['amp']:g} K · azimuthal mean", fontsize=9, weight="bold")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("radius [km]", fontsize=8)
        if j:
            ax.set_yticklabels([])
    axes[0, 0].set_ylabel("lat [°]  (upper ΔPV)", fontsize=9, weight="bold")
    axes[1, 0].set_ylabel("lat [°]  (low-level ΔPV)", fontsize=9, weight="bold")
    axes[2, 0].set_ylabel("altitude [km]", fontsize=9, weight="bold")

    for i, lim in enumerate((lim_up, lim_low, lim_az)):
        sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(-lim, lim))
        fig.colorbar(sm, ax=list(axes[i]), pad=0.01, fraction=0.02,
                     label=f"ΔPV{'/amp' if per_K else ''}  [{unit}]")

    kind = "ΔPV/amp_K (semi-linearity view)" if per_K else "ΔPV (absolute)"
    forcing = "δq-only" if _pert_params(run).get("type") == "moisture" else "Heating"
    fig.suptitle(f"{forcing} amplitude sweep structure — init {init_time}, "
                 f"nominal hour {lead_hr} — {kind}", fontsize=13, weight="bold")
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def plot_sweep_comparison(category_dir, out_png=None, *, lead_hr: int = 24,
                          pattern_a: str = "sweep_*", label_a: str = "moist (q free)",
                          pattern_b: str = "sweepq_*", label_b: str = "q-locked (δq=0)",
                          suptitle: str = "Moisture binding of the heating→PV "
                          "response — moist vs q-locked"):
    """Moist vs q-locked ΔPV amplitude curves — the moisture-binding probe.

    Solid = runs where the moisture channel co-evolves with the heating; dashed =
    runs with δq pinned to the control each step. The gap between the two curves is
    the moisture-mediated share of the PV response: how much of "thermodynamics" the
    manifold routes through q. Left panel: low-level generation; right: destruction aloft.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts_a = _sweep_points(category_dir, lead_hr, pattern_a)
    pts_b = _sweep_points(category_dir, lead_hr, pattern_b)
    palette = [_C_WEAK, _C_STRONG, "tab:green", "tab:purple"]
    inits = sorted(set(pts_a) | set(pts_b))
    color = {init: palette[i % len(palette)] for i, init in enumerate(inits)}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for ax, key, name in ((axl, "lowlevel_max", "max ΔPV 700–1000 hPa"),
                          (axr, "upperlevel_min", "min ΔPV 200–500 hPa")):
        ax.axhline(0, color="grey", lw=0.8)
        for init in inits:
            c = color[init]
            for pts, ls, mk, mfc, lab in ((pts_a.get(init), "-", "o", c, label_a),
                                          (pts_b.get(init), "--", "s", "none", label_b)):
                if not pts:
                    continue
                amps = [a for a, _ in pts]
                ax.plot(amps, [m[key] for _, m in pts], color=c, ls=ls, lw=2.0,
                        marker=mk, ms=5, mfc=mfc, label=f"init {init}  {lab}")
        ax.set_xlabel("Heating amplitude amp_K  [K]", fontsize=12, weight="bold")
        ax.set_title(name, fontsize=12, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)
    axl.set_ylabel(f"ΔPV at nominal hour {lead_hr}  [PVU]", fontsize=12, weight="bold")
    axl.legend(loc="best", fontsize=9)
    fig.suptitle(suptitle, fontsize=13, weight="bold")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


_CP_E, _LV_E = 1004.0, 2.5e6

# Human-readable "what was injected" labels for the run families (figure titles).
_FAMILY_LABEL = {
    "heating_moist": "inject ΔT (heating) — q free",
    "heating_qlock": "inject ΔT (heating) — δq locked to 0",
    "dq_latent":     "inject δq only — latent-equivalent (equal energy to heating)",
    "dq_measured":   "inject δq only — measured scaling (≈0.4–0.5× heating energy)",
}


def _mass_weights(nz: int) -> np.ndarray:
    """Per-level mass dm = Δp/g [kg m⁻²] (midpoint layer edges)."""
    p = np.asarray(layout.pressure_levels(), dtype=float)[:nz]
    edges = np.concatenate([[p[0]], (p[:-1] + p[1:]) / 2, [p[-1]]])
    return np.diff(edges) * 100.0 / 9.80665


def energy_series(run_dir, *, sigma: float | None = None) -> dict:
    """Core-mean column-energy partition per iteration, plus the injected reference.

    For each lead: sensible cp·∫δT dm and latent Lv·∫δq dm, averaged over the
    heated core (r ≤ 2σ), in MJ m⁻². ``e_inj`` is the cumulative injected energy
    under the same core-mean convention (ramps during forcing, flat after) —
    identical for a heating run and its latent-equivalent δq twin by design.
    """
    run_dir = Path(run_dir)
    pert = _pert_params(run_dir)
    sigma = float(pert["sigma"]) if sigma is None else sigma
    t_idx, q_idx = layout.upper_index("t"), layout.upper_index("q")

    pairs = _continuous_pairs(run_dir)
    up0, _ = io.load_delta_bundle(pairs[0][1])
    nz, ny, nx = up0.shape[:3]
    dm = _mass_weights(nz)
    yy, xx = np.mgrid[0:ny, 0:nx]
    core = (yy - ny // 2) ** 2 + (xx - nx // 2) ** 2 <= (2 * sigma) ** 2

    # Injected column energy per forcing step (core-mean of the Gaussian footprint).
    amp = float(pert["amp_K"])
    steps = int(pert["forcing_steps"])
    per_step = amp / steps if pert["amp_mode"] == "spread" else amp
    if pert.get("type") == "heating" or pert.get("profile") is None:
        prof, rate = vertical_profile(pert.get("heat_type", "Deep"),
                                      layout.pressure_levels())[:nz], _CP_E
        if pert.get("type") == "moisture":            # latent-equivalent scaling
            prof, rate = prof * float(pert.get("gkg_per_K", _CP_E / _LV_E * 1e3)) * 1e-3, _LV_E
    else:                                              # measured δq profile
        prof, rate = (np.asarray(pert["profile"], dtype=float)[:nz]
                      * float(pert["gkg_per_K"]) * 1e-3), _LV_E
    g_mean = float(gaussian_centered(ny, sigma)[core].mean())
    e_step = rate * per_step * float(np.sum(prof * dm)) * g_mean

    out = {"hour": [], "e_sens": [], "e_lat": [], "e_inj": []}
    for lead, d, _c in pairs:
        up, _ = io.load_delta_bundle(d)
        e_s = _CP_E * np.einsum("kyx,k->yx", up[..., t_idx], dm)[core].mean()
        e_l = _LV_E * np.einsum("kyx,k->yx", up[..., q_idx], dm)[core].mean()
        out["hour"].append(lead)
        out["e_sens"].append(e_s / 1e6)
        out["e_lat"].append(e_l / 1e6)
        out["e_inj"].append(min(lead // 3, steps) * e_step / 1e6)
    return out


def plot_energy_partition(run_dirs, out_png=None, *, ncols: int = 3):
    """Sensible/latent/total core column energy vs iteration, one panel per run.

    Reads the moisture-reservoir question directly: after forcing ends, a decaying
    latent curve with rising sensible = one-shot condensation (δq consumed); a
    persistent latent curve while the total grows past the injected reference =
    the reservoir/catalyst regime (energy is being amplified, not converted).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = [Path(r) for r in run_dirs]
    n = len(run_dirs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows),
                             sharex=True, squeeze=False)
    for ax, run in zip(axes.ravel(), run_dirs):
        s = energy_series(run)
        t_end = int(_pert_params(run)["forcing_steps"]) * 3
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(t_end, color="grey", lw=1.0, ls=":")
        ax.plot(s["hour"], s["e_inj"], color="0.45", lw=1.6, ls=":",
                label="injected (cumulative)")
        ax.plot(s["hour"], s["e_sens"], color="tab:red", lw=2.0, marker="o", ms=3,
                label="sensible  cp·∫δT dm")
        ax.plot(s["hour"], s["e_lat"], color="tab:blue", lw=2.0, marker="s", ms=3,
                label="latent  Lv·∫δq dm")
        tot = np.asarray(s["e_sens"]) + np.asarray(s["e_lat"])
        ax.plot(s["hour"], tot, color="0.15", lw=1.8, ls="--", label="total")
        cfg = load_stamp(run).get("resolved_config", {})
        fam = run.parent.name
        what = _FAMILY_LABEL.get(fam, fam)
        ax.set_title(f"{what}\n{cfg.get('tag', run.name)}  ·  init {cfg.get('init_time', '?')}",
                     fontsize=10.5, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(s["hour"][0], s["hour"][-1])
    for ax in axes[-1, :]:
        ax.set_xlabel("Iteration n  (nominal hour)", fontsize=11, weight="bold")
    for row in axes:
        row[0].set_ylabel("core-mean column energy\n[MJ m$^{-2}$]", fontsize=10, weight="bold")
    axes[0, 0].legend(loc="best", fontsize=8.5)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Energy partition of the perturbation — sensible vs latent vs injected",
                 fontsize=13.5, weight="bold")
    fig.text(0.995, 0.005,
             "latent decaying while sensible rises = one-shot condensation (δq consumed); "
             "latent persisting while total outgrows the injected line = reservoir/amplifier regime",
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def plot_moisture_hovmoller(run_dirs, out_png=None, *, dr_km: float = 25.0,
                            r_max_km: float = 900.0):
    """Radius–time Hovmöller of the column latent energy Lv·∫δq dm — one panel per run.

    Answers "where does the re-moistening come from": inward-sloping positive
    streaks = moisture converging from the environment into the core; a drying
    (negative) collar outside the core is the signature of the circulation
    harvesting environmental moisture. Dashed line marks the heated core (2σ).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = [Path(r) for r in run_dirs]
    n = len(run_dirs)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 6.4), sharey=True, squeeze=False)
    axes = axes[0]
    q_idx = layout.upper_index("q")

    panels = []
    vmax = 0.0
    for run in run_dirs:
        pairs = _continuous_pairs(run)
        up0, sfc0 = io.load_delta_bundle(pairs[0][2])
        lat2d, lon2d = sfc0[:, :, -1], sfc0[:, :, -2]
        nz = up0.shape[0]
        dm = _mass_weights(nz)
        sigma = float(_pert_params(run)["sigma"])
        # heated-core edge (2σ) in km, from the actual grid metric
        cy, cx = lat2d.shape[0] // 2, lat2d.shape[1] // 2
        core_km = abs((lon2d[cy, cx + int(2 * sigma)] - lon2d[cy, cx])
                      * 111.32 * np.cos(np.deg2rad(lat2d[cy, cx])))
        hours, rows = [], []
        for lead, d, _c in pairs:
            up, _ = io.load_delta_bundle(d)
            e_lat = _LV_E * np.einsum("kyx,k->yx", up[..., q_idx], dm) / 1e6
            r_km, az = _azimuthal_mean(e_lat[None], lat2d, lon2d,
                                       r_max_km=r_max_km, dr_km=dr_km)
            hours.append(lead)
            rows.append(az[0])
        hov = np.asarray(rows)
        vmax = max(vmax, np.nanpercentile(np.abs(hov), 98))
        panels.append((run, hours, r_km, hov, core_km))

    for ax, (run, hours, r_km, hov, core_km) in zip(axes, panels):
        pm = ax.pcolormesh(r_km, hours, hov, cmap="BrBG", vmin=-vmax, vmax=vmax,
                           shading="nearest")
        t_end = int(_pert_params(run)["forcing_steps"]) * 3
        ax.axhline(t_end, color="k", lw=1.0, ls=":")
        ax.text(r_max_km * 0.99, t_end, " forcing ends ", ha="right", va="bottom",
                fontsize=8, style="italic")
        ax.axvline(core_km, color="k", lw=1.2, ls="--")
        ax.text(core_km, ax.get_ylim()[0], " heated core (2σ) ", ha="left",
                va="bottom", fontsize=8, style="italic", rotation=90)
        cfg = load_stamp(run).get("resolved_config", {})
        what = _FAMILY_LABEL.get(Path(run).parent.name, Path(run).parent.name)
        ax.set_title(f"{what.replace(' — ', chr(10))}\n{cfg.get('tag', Path(run).name)}"
                     f"  ·  init {cfg.get('init_time', '?')}", fontsize=9.5, weight="bold")
        ax.set_xlabel("radius [km]", fontsize=11, weight="bold")
    axes[0].set_ylabel("Iteration n  (nominal hour)", fontsize=12, weight="bold")
    cb = fig.colorbar(pm, ax=list(axes), pad=0.01, shrink=0.9)
    cb.set_label("azimuthal-mean column latent energy  Lv·∫δq dm  [MJ m$^{-2}$]",
                 fontsize=10, weight="bold")
    fig.suptitle("Moisture Hovmöller — where does the re-moistening come from?",
                 fontsize=13, weight="bold")
    fig.text(0.995, 0.005,
             "green = moist anomaly, brown = dry; inward-sloping green streaks with a "
             "brown collar outside the core = the circulation harvesting environmental moisture",
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def imbalance_series(run_dir, *, level_hpa: int = 850,
                     r_km_range: tuple = (50.0, 500.0)) -> dict:
    """Excess gradient-wind imbalance of the perturbed state, per iteration.

    For each lead, reconstruct the absolute state (control + δ) and the control,
    take the azimuthal-mean tangential wind and geopotential at ``level_hpa``,
    derive the gradient wind, and average |v_t − v_gr| over ``r_km_range``.
    Returned ``excess`` = imbalance(perturbed) − imbalance(control) [m/s]:
    ≈0 means the perturbation (and any channel lock) left the vortex in gradient
    balance; a sustained positive excess means the intervention broke the balance
    (the response is then dying of dynamical inconsistency, not of fuel loss).
    """
    from .wind_profile import _radial_mean, _winds_isobaric

    OMEGA, R_EARTH = 7.292e-5, 6.371e6
    run_dir = Path(run_dir)
    pairs = _continuous_pairs(run_dir)
    lev = layout.pressure_levels()
    k = lev.index(level_hpa)
    ui, vi, zi = (layout.upper_index(n) for n in ("u", "v", "z"))

    out = {"hour": [], "excess": [], "pert": [], "ctrl": []}
    grid = None
    for lead, d, c in pairs:
        dup, dsfc = io.load_delta_bundle(d)
        cup, csfc = io.load_delta_bundle(c)
        if grid is None:
            ny, nx = csfc.shape[:2]
            ic, jc = ny // 2, nx // 2
            f_c = csfc[ic, jc, layout.surface_index("f")]
            clat = np.degrees(np.arcsin(np.clip(f_c / (2 * OMEGA), -1, 1)))
            dx = 0.25 * np.pi / 180 * R_EARTH * np.cos(np.radians(clat))
            dy = 0.25 * np.pi / 180 * R_EARTH
            yj, xj = np.indices((ny, nx))
            xd, yd = (xj - jc) * dx, (ic - yj) * dy
            r2d, th2d = np.hypot(xd, yd), np.arctan2(yd, xd)
            bins = np.arange(0, r2d.max() + 1e4, 1e4)
            r1d = 0.5 * (bins[:-1] + bins[1:])
            sel = (r1d / 1e3 >= r_km_range[0]) & (r1d / 1e3 <= r_km_range[1])
            grid = (f_c, r2d, th2d, bins, r1d, sel)
        f_c, r2d, th2d, bins, r1d, sel = grid

        def _imb(up):
            vt = -up[k, :, :, ui] * np.sin(th2d) + up[k, :, :, vi] * np.cos(th2d)
            vt1 = _radial_mean(vt, r2d, bins, r1d)
            z1 = _radial_mean(up[k, :, :, zi], r2d, bins, r1d)
            _vg, vgr = _winds_isobaric(z1, np.where(r1d == 0, 1e-5, r1d), f_c)
            return float(np.nanmean(np.abs(vt1 - vgr)[sel]))

        i_p, i_c = _imb(cup + dup), _imb(cup)
        out["hour"].append(lead)
        out["pert"].append(i_p)
        out["ctrl"].append(i_c)
        out["excess"].append(i_p - i_c)
    return out


def plot_imbalance(run_dirs, out_png=None, *, labels=None, level_hpa: int = 850,
                   suptitle: str = "Excess gradient-wind imbalance under channel locks"):
    """Excess |v_t − v_gr| vs iteration for N runs — the collapse-mode discriminator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = [Path(r) for r in run_dirs]
    labels = list(labels) if labels else [r.parent.name for r in run_dirs]
    colors = ["0.1", "tab:blue", "tab:green", "tab:purple", "tab:red", "tab:orange"]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    t_end = int(_pert_params(run_dirs[0])["forcing_steps"]) * 3
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(t_end, color="grey", lw=1.0, ls=":", label=f"forcing ends ({t_end} h)")
    for i, (run, lab) in enumerate(zip(run_dirs, labels)):
        s = imbalance_series(run, level_hpa=level_hpa)
        ax.plot(s["hour"], s["excess"], color=colors[i % len(colors)],
                lw=2.6 if i == 0 else 1.8, marker="o", ms=3,
                mfc=colors[i % len(colors)] if i == 0 else "none", label=lab)
    ax.set_xlabel("Iteration n  (nominal hour)", fontsize=12, weight="bold")
    ax.set_ylabel(f"excess ⟨|v$_t$ − v$_{{gr}}$|⟩ at {level_hpa} hPa, "
                  "r = 50–500 km  [m s$^{-1}$]", fontsize=11, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    cfg = load_stamp(run_dirs[0]).get("resolved_config", {})
    ax.set_title(f"{suptitle} — init {cfg.get('init_time', '?')}",
                 fontsize=12.5, weight="bold")
    fig.text(0.995, 0.005,
             "≈0 = intervention keeps the vortex in gradient balance (collapse = fuel "
             "starvation); sustained positive = the lock itself broke the balance "
             "(collapse = dynamical inconsistency)",
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def loop_series(run_dir) -> dict:
    """The T–q–w–z loop rendered as five secondary-circulation observables.

    Per iteration, in physical units:
      * ``warm_core``   core-mean δT, mass-weighted 400–700 hPa            [K]
      * ``moisture``    core-mean column ∫δq dm                            [kg m⁻²]
      * ``updraft``     core-mean −δw, 400–700 hPa (upward positive)
      * ``inflow``      ring-mean (150–400 km) −δu_r, 850–1000 hPa         [m s⁻¹]
      * ``q_import``    ring-mean −δ(u_r·q), 700–1000 hPa mass-weighted
                        (the harvester's intake: positive = anomalous
                        inward moisture flux)                              [kg m⁻² · m s⁻¹ /1e3]
    Core = r ≤ 2σ around the domain centre. This maps the abstract channel locks
    onto the textbook secondary circulation: inflow → ascent → latent heating →
    warm core → pressure gradient → inflow (CISK/WISHE loop).
    """
    run_dir = Path(run_dir)
    sigma = float(_pert_params(run_dir)["sigma"])
    pairs = _continuous_pairs(run_dir)
    ui, vi, ti, qi, wi = (layout.upper_index(n) for n in ("u", "v", "t", "q", "w"))
    p = np.asarray(layout.pressure_levels(), dtype=float)
    dm = _mass_weights(len(p))
    mid = (p >= 400) & (p <= 700)
    low = (p >= 850)
    lowq = (p >= 700)

    grid = None
    out = {k: [] for k in ("hour", "warm_core", "moisture", "updraft",
                           "inflow", "q_import")}
    for lead, d, c in pairs:
        dup, dsfc = io.load_delta_bundle(d)
        cup, csfc = io.load_delta_bundle(c)
        if grid is None:
            lat2d, lon2d = csfc[:, :, -1], csfc[:, :, -2]
            ny, nx = lat2d.shape
            cy, cx = ny // 2, nx // 2
            dy = (lat2d - lat2d[cy, cx]) * 111.32
            dx = (lon2d - lon2d[cy, cx]) * 111.32 * np.cos(np.deg2rad(lat2d[cy, cx]))
            r = np.hypot(dy, dx)
            th = np.arctan2(dy, dx)
            core = r <= 2 * sigma * 0.25 * 111.32 * np.cos(np.deg2rad(lat2d[cy, cx]))
            ring = (r >= 150) & (r <= 400)
            grid = (th, core, ring)
        th, core, ring = grid

        w_mid = np.average(dup[..., wi][mid], axis=0, weights=dm[mid])
        t_mid = np.average(dup[..., ti][mid], axis=0, weights=dm[mid])
        # radial velocity (positive outward) of δ and of the total/control fields
        dur = dup[..., ui] * np.cos(th) + dup[..., vi] * np.sin(th)
        ur_p = (cup + dup)[..., ui] * np.cos(th) + (cup + dup)[..., vi] * np.sin(th)
        ur_c = cup[..., ui] * np.cos(th) + cup[..., vi] * np.sin(th)
        flux = ((ur_p * (cup + dup)[..., qi]) - (ur_c * cup[..., qi]))
        out["hour"].append(lead)
        out["warm_core"].append(float(t_mid[core].mean()))
        out["moisture"].append(float(
            np.einsum("kyx,k->yx", dup[..., qi], dm)[core].mean()))
        out["updraft"].append(float(-w_mid[core].mean()))
        out["inflow"].append(float(
            -np.average(dur[low], axis=0, weights=dm[low])[ring].mean()))
        out["q_import"].append(float(
            -np.einsum("kyx,k->yx", flux[lowq], dm[lowq])[ring].mean() / 1e3))
    return out


_LOOP_STYLE = [("warm_core", "warm core δT (400–700)", "tab:red"),
               ("moisture", "core column δq", "tab:blue"),
               ("updraft", "core ascent −δw (400–700)", "tab:orange"),
               ("inflow", "low-level inflow −δu$_r$ (150–400 km)", "tab:green"),
               ("q_import", "moisture import −δ(u$_r$q)", "tab:purple")]


def plot_loop_sequence(run_dirs, out_png=None, *, labels=None, ncols: int = 2,
                       suptitle: str = "Death sequence of the T–q–w–z loop"):
    """Normalized loop observables per run — the domino-order figure.

    Every series is normalized by its own max |value| in the FIRST run (the free
    reference), so a curve at 1 means "as strong as the free run ever got" and the
    order in which curves die reveals the causal chain each lock severs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = [Path(r) for r in run_dirs]
    labels = list(labels) if labels else [r.parent.name for r in run_dirs]
    series = [loop_series(r) for r in run_dirs]
    norm = {k: max(1e-12, float(np.max(np.abs(series[0][k]))))
            for k, _l, _c in _LOOP_STYLE}

    n = len(run_dirs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    t_end = int(_pert_params(run_dirs[0])["forcing_steps"]) * 3
    for ax, s, lab in zip(axes.ravel(), series, labels):
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(t_end, color="grey", lw=1.0, ls=":")
        for k, name, col in _LOOP_STYLE:
            ax.plot(s["hour"], np.asarray(s[k]) / norm[k], color=col, lw=1.8,
                    marker="o", ms=2.5, label=name)
        ax.set_title(lab, fontsize=11.5, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
    for ax in axes[-1, :]:
        ax.set_xlabel("Iteration n  (nominal hour)", fontsize=11, weight="bold")
    for row in axes:
        row[0].set_ylabel("normalized by free-run max", fontsize=10, weight="bold")
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    cfg = load_stamp(run_dirs[0]).get("resolved_config", {})
    fig.suptitle(f"{suptitle} — init {cfg.get('init_time', '?')}",
                 fontsize=13.5, weight="bold")
    fig.text(0.995, 0.005,
             "each series normalized by its own max in the free run; the order in "
             "which curves collapse after a lock reveals the causal chain "
             "(inflow → ascent → latent heating → warm core → inflow)",
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def plot_tseries_multi(run_dirs, out_png=None, *, labels=None,
                       suptitle: str = "ΔPV persistence under channel locks"):
    """ΔPV extrema vs iteration for N runs — two panels (low-level | upper-level).

    Built for the lock-control comparison (free / lock q / lock w / lock z), but
    takes any list of continuous runs. First run is drawn as the reference (black,
    thicker); the rest follow the categorical palette in order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = [Path(r) for r in run_dirs]
    labels = list(labels) if labels else [r.parent.name for r in run_dirs]
    colors = ["0.1", "tab:blue", "tab:green", "tab:purple", "tab:red", "tab:orange"]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    t_end = int(_pert_params(run_dirs[0])["forcing_steps"]) * 3
    for ax, key, name in ((axl, "lowlevel_max", "max ΔPV 700–1000 hPa"),
                          (axr, "upperlevel_min", "min ΔPV 200–500 hPa")):
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(t_end, color="grey", lw=1.0, ls=":")
        for i, (run, lab) in enumerate(zip(run_dirs, labels)):
            s = pv_response_series(run)
            ax.plot(s["hour"], s[key], color=colors[i % len(colors)],
                    lw=2.6 if i == 0 else 1.8, marker="o", ms=3,
                    mfc=colors[i % len(colors)] if i == 0 else "none", label=lab)
        ax.set_xlabel("Iteration n  (nominal hour)", fontsize=12, weight="bold")
        ax.set_title(name, fontsize=12, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
    axl.set_ylabel("ΔPV  [PVU]", fontsize=12, weight="bold")
    axl.legend(loc="best", fontsize=10)
    cfg = load_stamp(run_dirs[0]).get("resolved_config", {})
    fig.suptitle(f"{suptitle} — init {cfg.get('init_time', '?')}",
                 fontsize=13.5, weight="bold")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def plot_sweep_ratio(category_dir, out_png=None, *, lead_hr: int = 24,
                     pattern_a: str = "sweep_*", label_a: str = "heating",
                     pattern_b: str = "sweepdql_*", label_b: str = "latent-equiv δq",
                     band: tuple = (0.8, 1.25),
                     suptitle: str = "Latent-heat equivalence ratio",
                     caption: str | None = None):
    """Ratio ΔPV(B)/ΔPV(A) vs amplitude on a log-2 axis — the equivalence view.

    A ratio of 1 means run set B reproduces run set A exactly; the shaded band is
    the "practically equivalent" zone. One curve per (init, dipole pole): init
    carries the hue (weak blue / strong red), the pole the line style (low-level
    solid / upper-level dashed) — same conventions as the comparison figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts_a = _sweep_points(category_dir, lead_hr, pattern_a)
    pts_b = _sweep_points(category_dir, lead_hr, pattern_b)
    inits = sorted(set(pts_a) & set(pts_b))
    palette = [_C_WEAK, _C_STRONG, "tab:green", "tab:purple"]
    color = {init: palette[i % len(palette)] for i, init in enumerate(inits)}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhspan(band[0], band[1], color="0.85", alpha=0.5, zorder=0)
    ax.axhline(1.0, color="0.35", lw=1.2, zorder=1)
    ax.text(0.995, 1.0, " ratio = 1: perfect equivalence ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9, color="0.35", style="italic")
    ax.text(0.995, band[0], f" ±{int((band[1]-1)*100)}% band ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.5", style="italic")

    for init in inits:
        a = {amp: m for amp, m in pts_a[init]}
        b = {amp: m for amp, m in pts_b[init]}
        amps = sorted(set(a) & set(b))
        c = color[init]
        for key, name, ls, mk in (("lowlevel_max", "low-level max (700–1000 hPa)", "-", "o"),
                                  ("upperlevel_min", "upper-level min (200–500 hPa)", "--", "s")):
            r = [b[x][key] / a[x][key] for x in amps]
            hero = key == "lowlevel_max"
            ax.plot(amps, r, color=c, ls=ls, lw=2.6 if hero else 1.8,
                    marker=mk, ms=5 if hero else 4, mfc=c if ls == "-" else "none",
                    label=f"init {init}  {name}", zorder=3)
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8])
    ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2", "3", "4", "6", "8"])
    ax.set_xlabel("Nominal amplitude amp_K  [K]", fontsize=13, weight="bold")
    ax.set_ylabel(f"ΔPV ratio   {label_b} ÷ {label_a}", fontsize=13, weight="bold")
    ax.set_xlim(left=0)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(f"{suptitle} — nominal hour {lead_hr}", fontsize=14, weight="bold")
    if caption is None:
        caption = (f"ratio of {label_b} to {label_a} response at equal nominal K; "
                   "a curve inside the band means the manifold converts moisture and heat "
                   "at (near) the physical cp/Lv exchange rate on that pathway")
    fig.text(0.995, 0.005, caption,
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def plot_tseries_comparison(run_moist, run_qlock, out_png=None, *,
                            label_a: str = "moist", label_b: str = "q-locked",
                            title_prefix: str = "Moisture binding — moist vs q-locked:",
                            caption: str = "gap between solid and dashed = the "
                            "moisture-mediated share of the heating→PV response "
                            "(δq pinned to control in the q-locked run)"):
    """Model ΔPV extrema per iteration, run A (solid) vs run B (dashed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm = pv_response_series(run_moist)
    sq = pv_response_series(run_qlock)
    t_end = int(_pert_params(run_moist)["forcing_steps"]) * 3

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="grey", lw=0.8)
    if sm["hour"][0] < t_end < sm["hour"][-1]:
        ax.axvline(t_end, color="grey", lw=1.0, ls=":", label=f"forcing ends ({t_end} h)")
    ax.plot(sm["hour"], sm["lowlevel_max"], color=_C_LOW, lw=2.0, marker="o", ms=4,
            label=f"{label_a}  max ΔPV 700–1000 hPa")
    ax.plot(sq["hour"], sq["lowlevel_max"], color=_C_LOW, lw=2.0, ls="--", marker="o",
            ms=4, mfc="none", label=f"{label_b}  max ΔPV 700–1000 hPa")
    ax.plot(sm["hour"], sm["upperlevel_min"], color=_C_UP, lw=2.0, marker="s", ms=4,
            label=f"{label_a}  min ΔPV 200–500 hPa")
    ax.plot(sq["hour"], sq["upperlevel_min"], color=_C_UP, lw=2.0, ls="--", marker="s",
            ms=4, mfc="none", label=f"{label_b}  min ΔPV 200–500 hPa")
    ax.set_xlabel("Iteration n  (nominal hour)", fontsize=13, weight="bold")
    ax.set_ylabel("ΔPV  [PVU]", fontsize=13, weight="bold")
    ax.set_xlim(sm["hour"][0], sm["hour"][-1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(experiment_title(run_moist, prefix=title_prefix),
                 fontsize=13, weight="bold")
    fig.text(0.995, 0.005, caption,
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


# --------------------------------------------------------------------------- #
# Additivity: moist ≟ q-locked + δq-only — the T×q synergy test
# --------------------------------------------------------------------------- #
# The three sweep families that decompose the moist response into its severed
# halves. All three share the same control trajectory per init, so their ΔPV
# fields live on the same state and may be summed point-by-point.
_ADD_SETS = (("moist", "heating_moist", "sweep_*"),
             ("qlock", "heating_qlock", "sweepq_*"),
             ("dq", "dq_measured", "sweepdq_*"))


def _dpv_reduce(dpv, *, sigma: float = 5.0):
    """The standard dipole reduction applied to an arbitrary ΔPV field."""
    p = np.asarray(layout.pressure_levels(), dtype=float)[: dpv.shape[0]]
    core = _core_mask(dpv.shape[1], dpv.shape[2], sigma)
    low, kji_low = _extremum(dpv, (p >= _LOW_P[0]) & (p <= _LOW_P[1]), core, "max")
    up, _kji = _extremum(dpv, (p >= _UP_P[0]) & (p <= _UP_P[1]), core, "min")
    return low, up, kji_low


def _lead_pairs(category_dir, pattern, family, lead_hr):
    """{(init, amp_K): (delta_path, control_path)} over one sweep family."""
    out = {}
    for run in _find_runs(category_dir, pattern, family):
        hits = sorted((run / "data").glob(f"delta_continuous_*lead{lead_hr:03d}hr.npz"))
        if not hits:
            continue
        d = hits[0]
        c = d.parent / d.name.replace("delta_", "control_")
        if not c.exists():
            continue
        cfg = load_stamp(run).get("resolved_config", {})
        out[(str(cfg.get("init_time", "?")), float(_pert_params(run)["amp_K"]))] = (d, c)
    return out


def additivity_series(category_dir, *, lead_hr: int = 24, sigma: float = 5.0):
    """Field-level additivity check: ΔPV_moist ≟ ΔPV_qlock + ΔPV_dq.

    The sum field gets the SAME dipole reduction as the moist field (extrema of
    the summed field, not sums of extrema, so the comparison is not polluted by
    the poles sitting at different points). The gap ``moist − (qlock + dq)`` is
    the T×q synergy: what heating and its own moisture response build together
    that the two severed halves cannot.
    """
    idx = {name: _lead_pairs(category_dir, pat, fam, lead_hr)
           for name, fam, pat in _ADD_SETS}
    keys = sorted(set(idx["moist"]) & set(idx["qlock"]) & set(idx["dq"]))
    if not keys:
        raise FileNotFoundError(
            f"no (init, amp) present in all three additivity families under {category_dir}")
    out: dict = {}
    for init, amp in keys:
        rec = out.setdefault(init, {"amps": []})
        rec["amps"].append(amp)
        fields = {name: delta_pv_field(*idx[name][(init, amp)]) for name in idx}
        fields["sum"] = fields["qlock"] + fields["dq"]
        for name in ("moist", "qlock", "dq", "sum"):
            low, up, _ = _dpv_reduce(fields[name], sigma=sigma)
            rec.setdefault(f"{name}_low", []).append(low)
            rec.setdefault(f"{name}_up", []).append(up)
    return out


def plot_additivity(category_dir, out_png=None, *, lead_hr: int = 24,
                    sigma: float = 5.0, band: tuple = (0.8, 1.25)):
    """3×N additivity figure, one column per init.

    Rows 1–2: low-level max / upper-level min ΔPV vs amplitude — moist (solid
    black) against the reconstruction qlock+dq (dashed black), with the severed
    halves as thin colored lines. Row 3: synergy factor moist ÷ (qlock+dq) on a
    log-2 axis; 1 = the decomposition is complete, >1 = super-additive (the T×q
    coupling creates response neither half owns alone).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = additivity_series(category_dir, lead_hr=lead_hr, sigma=sigma)
    inits = sorted(data)
    fig, axes = plt.subplots(3, len(inits), figsize=(7.6 * len(inits), 13.5),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.85]})
    comp = (("moist", "moist (heating, q free)", "black", "-", 2.6, "o"),
            ("sum", "q-locked + δq-only (reconstruction)", "black", "--", 2.2, None),
            ("qlock", "q-locked alone (T path)", _C_UP, "-", 1.3, None),
            ("dq", "δq-only alone (q path)", _C_LOW, "-", 1.3, None))
    for j, init in enumerate(inits):
        rec = data[init]
        amps = np.asarray(rec["amps"])
        for i, (pole, name) in enumerate((("low", "max ΔPV 700–1000 hPa"),
                                          ("up", "min ΔPV 200–500 hPa"))):
            ax = axes[i, j]
            ax.axhline(0, color="grey", lw=0.8)
            for key, lab, c, ls, lw, mk in comp:
                ax.plot(amps, rec[f"{key}_{pole}"], color=c, ls=ls, lw=lw,
                        marker=mk, ms=4, label=lab if (i, j) == (0, 0) else None)
            ax.set_ylabel(f"{name}  [PVU]", fontsize=11, weight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                ax.set_title(f"init {init}", fontsize=13, weight="bold")
        ax = axes[2, j]
        ax.axhspan(band[0], band[1], color="0.85", alpha=0.5, zorder=0)
        ax.axhline(1.0, color="0.35", lw=1.2, zorder=1)
        for pole, name, c, ls in (("low", "low-level max", _C_LOW, "-"),
                                  ("up", "upper-level min", _C_UP, "--")):
            r = np.asarray(rec[f"moist_{pole}"]) / np.asarray(rec[f"sum_{pole}"])
            r = np.where(r > 0, r, np.nan)      # log axis: mask sign flips
            ax.plot(amps, r, color=c, ls=ls, lw=2.0, marker="o", ms=4,
                    mfc=c if ls == "-" else "none",
                    label=f"synergy factor, {name}" if j == 0 else None)
        ax.set_yscale("log", base=2)
        ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 3])
        ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
        ax.set_ylabel("moist ÷ (qlock + δq)", fontsize=11, weight="bold")
        ax.set_xlabel("Nominal amplitude amp_K  [K]", fontsize=12, weight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.set_xlim(left=0)
        if j == 0:
            ax.legend(loc="best", fontsize=9)
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle(f"Additivity of the ΔPV response — moist vs (q-locked + δq-only), "
                 f"nominal hour {lead_hr}", fontsize=14, weight="bold")
    fig.text(0.995, 0.005,
             "ΔPV fields summed point-by-point on the shared control trajectory, THEN reduced: "
             "the dashed curve is the extremum of the summed field, not the sum of the two "
             "extrema — where the halves' lobes are spatially misaligned (upper pole) they "
             "partially cancel and the dashed curve can sit above both thin curves; "
             "read S quantitatively only where the lobes align (low-level pole)",
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.012, 1, 1))
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def plot_additivity_maps(category_dir, out_png=None, *, amp: float = 5.0,
                         lead_hr: int = 24, sigma: float = 5.0,
                         zoom_deg: float = 5.0):
    """N×3 synergy maps at one amplitude: moist | qlock+δq | residual.

    Each row is one init; all three panels sit on the moist run's own low-level
    extremum layer with one shared symmetric colour scale, so the residual panel
    reads directly as "how much of the moist field the reconstruction misses".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = {name: _lead_pairs(category_dir, pat, fam, lead_hr)
           for name, fam, pat in _ADD_SETS}
    inits = sorted({i for i, a in idx["moist"]
                    if a == amp and (i, a) in idx["qlock"] and (i, a) in idx["dq"]})
    if not inits:
        raise FileNotFoundError(f"amp {amp:g} K not present in all three families")

    rows = []
    for init in inits:
        fields = {name: delta_pv_field(*idx[name][(init, amp)]) for name in idx}
        fields["sum"] = fields["qlock"] + fields["dq"]
        fields["resid"] = fields["moist"] - fields["sum"]
        _low, _up, kji = _dpv_reduce(fields["moist"], sigma=sigma)
        c_up, c_sfc = io.load_delta_bundle(idx["moist"][(init, amp)][1])
        *_f, lat2d, lon2d, p_hpa = fields_from_bundle(c_up, c_sfc)
        rows.append({"init": init, "fields": fields, "k": kji[0],
                     "p": float(p_hpa[kji[0]]), "lat2d": lat2d, "lon2d": lon2d})

    ny, nx = rows[0]["lat2d"].shape
    cy, cx = ny // 2, nx // 2
    half = int(round(zoom_deg / 0.25))
    ys, xs = slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1)

    cols = (("moist", "moist (heating, q free)"),
            ("sum", "q-locked + δq-only (reconstruction)"),
            ("resid", "residual = T×q synergy"))
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 4.4 * len(rows)),
                             squeeze=False)
    for i, row in enumerate(rows):
        k = row["k"]
        lim = max(float(np.nanmax(np.abs(row["fields"][n][k][ys, xs])))
                  for n, _ in cols) or 1.0
        share = (np.nanmax(np.abs(row["fields"]["resid"][k][ys, xs]))
                 / (np.nanmax(np.abs(row["fields"]["moist"][k][ys, xs])) or 1.0))
        for j, (name, lab) in enumerate(cols):
            ax = axes[i, j]
            im = ax.pcolormesh(row["lon2d"][ys, xs], row["lat2d"][ys, xs],
                               row["fields"][name][k][ys, xs],
                               cmap="bwr", vmin=-lim, vmax=lim, shading="auto")
            extra = f"  (peak {share:.0%} of moist)" if name == "resid" else ""
            ax.set_title(f"{lab}{extra}", fontsize=10.5, weight="bold")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            if j:
                ax.set_yticklabels([])
        axes[i, 0].set_ylabel(f"init {row['init']}\nlat [°]  ·  {row['p']:.0f} hPa",
                              fontsize=10, weight="bold")
        fig.colorbar(im, ax=list(axes[i]), pad=0.01, fraction=0.02,
                     label="ΔPV  [PVU]")
    fig.suptitle(f"ΔPV additivity maps — {amp:g} K, nominal hour {lead_hr}, "
                 "moist low-level extremum layer", fontsize=13, weight="bold")
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ΔPV forcing–response figures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep", help="ΔPV vs heating amplitude at a fixed lead")
    s.add_argument("category_dir", help="e.g. outputs/diabatic_heating")
    s.add_argument("--lead", type=int, default=24, help="lead hour (default 24)")
    s.add_argument("--pattern", default="sweep_*")
    s.add_argument("--family", default=None,
                   help="restrict the search to one family folder (default: all)")
    s.add_argument("--out", default="pv_amplitude_sweep.png")
    t = sub.add_parser("timeseries", help="ΔPV per iteration, model vs theory")
    t.add_argument("run_dir", help="one continuous-mode run dir")
    t.add_argument("--out", default="pv_timeseries.png")
    g = sub.add_parser("maps", help="6×3 ΔPV structure panel across amplitudes")
    g.add_argument("category_dir", help="e.g. outputs/diabatic_heating")
    g.add_argument("--init", required=True, help="init time YYYYMMDDHH")
    g.add_argument("--amps", nargs="+", type=float, default=[0.5, 2, 4, 6, 8, 10])
    g.add_argument("--lead", type=int, default=24)
    g.add_argument("--per-K", action="store_true", help="plot ΔPV/amp_K instead")
    g.add_argument("--prefix", default="sweep", help="run-folder prefix (sweep|sweepq)")
    g.add_argument("--out", default="pv_sweep_maps.png")
    cs = sub.add_parser("compare-sweep", help="amplitude curves, run set A vs B")
    cs.add_argument("category_dir")
    cs.add_argument("--lead", type=int, default=24)
    cs.add_argument("--pattern-a", default="sweep_*")
    cs.add_argument("--label-a", default="moist (q free)")
    cs.add_argument("--pattern-b", default="sweepq_*")
    cs.add_argument("--label-b", default="q-locked (δq=0)")
    cs.add_argument("--title", default="Moisture binding of the heating→PV response "
                                       "— moist vs q-locked")
    cs.add_argument("--out", default="pv_sweep_qlock_comparison.png")
    lp = sub.add_parser("loop", help="T–q–w–z loop observables (death-sequence figure)")
    lp.add_argument("run_dirs", nargs="+", help="first run = free reference")
    lp.add_argument("--labels", nargs="+", default=None)
    lp.add_argument("--title", default="Death sequence of the T–q–w–z loop")
    lp.add_argument("--out", default="pv_loop_sequence.png")
    im = sub.add_parser("imbalance", help="excess gradient-wind imbalance vs iteration")
    im.add_argument("run_dirs", nargs="+")
    im.add_argument("--labels", nargs="+", default=None)
    im.add_argument("--level", type=int, default=850)
    im.add_argument("--title", default="Excess gradient-wind imbalance under channel locks")
    im.add_argument("--out", default="pv_imbalance.png")
    mt = sub.add_parser("multi-tseries", help="ΔPV series overlay for N runs (lock controls)")
    mt.add_argument("run_dirs", nargs="+")
    mt.add_argument("--labels", nargs="+", default=None)
    mt.add_argument("--title", default="ΔPV persistence under channel locks")
    mt.add_argument("--out", default="pv_tseries_multi.png")
    qh = sub.add_parser("qhov", help="radius–time Hovmöller of column latent energy")
    qh.add_argument("run_dirs", nargs="+", help="continuous run dirs (one panel each)")
    qh.add_argument("--r-max", type=float, default=900.0)
    qh.add_argument("--out", default="pv_moisture_hovmoller.png")
    en = sub.add_parser("energy", help="sensible/latent/total column energy vs iteration")
    en.add_argument("run_dirs", nargs="+", help="continuous run dirs (one panel each)")
    en.add_argument("--ncols", type=int, default=3)
    en.add_argument("--out", default="pv_energy_partition.png")
    rs = sub.add_parser("ratio-sweep", help="ΔPV(B)/ΔPV(A) vs amplitude (equivalence view)")
    rs.add_argument("category_dir")
    rs.add_argument("--lead", type=int, default=24)
    rs.add_argument("--pattern-a", default="sweep_*")
    rs.add_argument("--label-a", default="heating")
    rs.add_argument("--pattern-b", default="sweepdql_*")
    rs.add_argument("--label-b", default="latent-equiv δq")
    rs.add_argument("--title", default="Latent-heat equivalence ratio")
    rs.add_argument("--caption", default=None, help="footer note (default: cp/Lv wording)")
    rs.add_argument("--out", default="pv_sweep_ratio.png")
    ad = sub.add_parser("additivity", help="moist vs (q-locked + δq-only) — T×q synergy")
    ad.add_argument("category_dir")
    ad.add_argument("--lead", type=int, default=24)
    ad.add_argument("--out", default="pv_sweep_additivity.png")
    ad.add_argument("--maps-out", default=None,
                    help="also render the N×3 synergy maps at --map-amp")
    ad.add_argument("--map-amp", type=float, default=5.0)
    ct = sub.add_parser("compare-tseries", help="ΔPV series, run A (solid) vs B (dashed)")
    ct.add_argument("run_moist")
    ct.add_argument("run_qlock")
    ct.add_argument("--label-a", default="moist")
    ct.add_argument("--label-b", default="q-locked")
    ct.add_argument("--title-prefix", default="Moisture binding — moist vs q-locked:")
    ct.add_argument("--caption", default="gap between solid and dashed = the "
                    "moisture-mediated share of the heating→PV response "
                    "(δq pinned to control in the q-locked run)")
    ct.add_argument("--out", default="pv_tseries_qlock_comparison.png")
    a = ap.parse_args()
    if a.cmd == "sweep":
        plot_amplitude_sweep(a.category_dir, a.out, lead_hr=a.lead, pattern=a.pattern,
                             family=a.family)
    elif a.cmd == "maps":
        plot_sweep_maps(a.category_dir, a.init, a.out, amps=tuple(a.amps),
                        lead_hr=a.lead, per_K=a.per_K,
                        pattern=a.prefix + "_{amp}_{h}h_init{init}")
    elif a.cmd == "compare-sweep":
        plot_sweep_comparison(a.category_dir, a.out, lead_hr=a.lead,
                              pattern_a=a.pattern_a, label_a=a.label_a,
                              pattern_b=a.pattern_b, label_b=a.label_b,
                              suptitle=a.title)
    elif a.cmd == "energy":
        plot_energy_partition(a.run_dirs, a.out, ncols=a.ncols)
    elif a.cmd == "qhov":
        plot_moisture_hovmoller(a.run_dirs, a.out, r_max_km=a.r_max)
    elif a.cmd == "multi-tseries":
        plot_tseries_multi(a.run_dirs, a.out, labels=a.labels, suptitle=a.title)
    elif a.cmd == "imbalance":
        plot_imbalance(a.run_dirs, a.out, labels=a.labels, level_hpa=a.level,
                       suptitle=a.title)
    elif a.cmd == "loop":
        plot_loop_sequence(a.run_dirs, a.out, labels=a.labels, suptitle=a.title)
    elif a.cmd == "ratio-sweep":
        plot_sweep_ratio(a.category_dir, a.out, lead_hr=a.lead,
                         pattern_a=a.pattern_a, label_a=a.label_a,
                         pattern_b=a.pattern_b, label_b=a.label_b, suptitle=a.title,
                         caption=a.caption)
    elif a.cmd == "additivity":
        plot_additivity(a.category_dir, a.out, lead_hr=a.lead)
        if a.maps_out:
            plot_additivity_maps(a.category_dir, a.maps_out, amp=a.map_amp,
                                 lead_hr=a.lead)
    elif a.cmd == "compare-tseries":
        plot_tseries_comparison(a.run_moist, a.run_qlock, a.out,
                                label_a=a.label_a, label_b=a.label_b,
                                title_prefix=a.title_prefix, caption=a.caption)
    else:
        plot_pv_timeseries(a.run_dir, a.out)
