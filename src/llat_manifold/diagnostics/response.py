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
  iteration against the **state-based** semi-linear expectation. Since
  PV ∝ (ζ+f)·∂θ/∂p, an injected (accumulated) Δθ changes the static stability and
  the PV it should support:

      ΔPV_theory(n) = PV_pert · (∂Δθ_acc/∂p) / (∂θ_pert/∂p)

  evaluated **at the model extremum's own location at iteration n**, with the
  model's current perturbed PV (so the expectation tracks the model state; it keeps
  evolving after the heating stops because PV_pert, θ_pert and the location do).
  Model above theory = the manifold holds more ΔPV than the injected stability
  change supports; below = it has diffused/redistributed it.

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
    dpv = pv_pert - _pv_of(c_up, c_sfc)

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
    aux = {"pv_pert": pv_pert, "t_pert": t, "p_hpa": p,
           "dpv": dpv, "z": _z, "lat2d": lat2d, "lon2d": lon2d}
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
    if pert.get("type") != "heating":
        raise ValueError(f"{run_dir}: not a heating run ({pert.get('type')!r})")
    pert.setdefault("sigma", 5.0)
    pert.setdefault("amp_mode", "spread")
    pert.setdefault("forcing_steps", 1)
    return pert


def pv_response_series(run_dir) -> dict:
    """Model ΔPV extrema and the state-based theory per iteration.

    Theory at iteration n, per dipole pole, at the model extremum's grid point:
        ΔPV_th = PV_pert · (∂Δθ_acc/∂p) / (∂θ_pert/∂p)
    where Δθ_acc is the injected temperature bump accumulated up to n (converted to
    potential temperature), i.e. the stability change the heating imposed on the
    column, and PV_pert/θ_pert are the model's *current* perturbed fields.
    """
    pert = _pert_params(run_dir)
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

    # Per-step injected Δθ and its pressure gradient (accumulation is n × this).
    dT_step = amp * (vertical_profile(pert["heat_type"], p_pa / 100.0)[:, None, None]
                     * gaussian_centered(ny, sigma)[None, :, :])
    ddtheta_dp_step = np.gradient(dT_step * (1.0e5 / p3) ** _KAPPA, p_pa, axis=0)

    out: dict = {"hour": []}
    for lead, d, c in pairs:
        m, aux = analyze_pair(d, c, sigma=sigma)
        n_forced = min(lead // 3, steps)
        theta_pert = aux["t_pert"] * (1.0e5 / p3) ** _KAPPA
        dtheta_dp = np.gradient(theta_pert, p_pa, axis=0)
        ratio = n_forced * ddtheta_dp_step / dtheta_dp
        m["theory_low"] = float(aux["pv_pert"][m["lowlevel_kji"]] * ratio[m["lowlevel_kji"]])
        m["theory_up"] = float(aux["pv_pert"][m["upperlevel_kji"]] * ratio[m["upperlevel_kji"]])
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
    if s["hour"][0] < t_end < s["hour"][-1]:
        ax.axvline(t_end, color="grey", lw=1.0, ls=":",
                   label=f"heating ends ({t_end} h)")
    ax.plot(s["hour"], s["lowlevel_max"], color=_C_LOW, lw=2.0,
            marker="o", ms=4, label="model  max ΔPV 700–1000 hPa")
    ax.plot(s["hour"], s["theory_low"], color=_C_LOW, lw=2.0, ls="--",
            label="theory  PV·∂Δθ/∂p ÷ ∂θ/∂p  at model max")
    ax.plot(s["hour"], s["upperlevel_min"], color=_C_UP, lw=2.0,
            marker="s", ms=4, label="model  min ΔPV 200–500 hPa")
    ax.plot(s["hour"], s["theory_up"], color=_C_UP, lw=2.0, ls="--",
            label="theory  PV·∂Δθ/∂p ÷ ∂θ/∂p  at model min")
    ax.set_xlabel("Iteration n  (nominal hour)", fontsize=13, weight="bold")
    ax.set_ylabel("ΔPV  [PVU]", fontsize=13, weight="bold")
    ax.set_xlim(s["hour"][0], s["hour"][-1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(experiment_title(run_dir, prefix="ΔPV vs diabatic theory:"),
                 fontsize=13, weight="bold")
    fig.text(0.995, 0.005,
             "theory = model's current PV × injected stability change, at the model "
             "extremum's own location each iteration (state-based semi-linear)",
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def _sweep_points(category_dir, lead_hr: int, pattern: str):
    """{init_time: [(amp_K, metrics)]} at the requested lead over sweep run dirs."""
    by_init: dict[str, list] = {}
    for run in sorted(Path(category_dir).glob(pattern)):
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
                         pattern: str = "sweep_*"):
    """ΔPV dipole scalars at a fixed lead vs heating amplitude, per init time.

    Each point carries a small ``(pressure level, radius from centre)`` tag locating
    its extremum. The dotted grey line through the origin is pinned to each curve's
    smallest amplitude — the semi-linear expectation; departure is manifold curvature.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = _sweep_points(category_dir, lead_hr, pattern)
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
        run = category_dir / pattern.format(amp=_amp_tag(amp), h=lead_hr, init=init_time)
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
    fig.suptitle(f"Heating amplitude sweep structure — init {init_time}, "
                 f"nominal hour {lead_hr} — {kind}", fontsize=13, weight="bold")
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def plot_sweep_comparison(category_dir, out_png=None, *, lead_hr: int = 24,
                          pattern_a: str = "sweep_*", label_a: str = "moist (q free)",
                          pattern_b: str = "sweepq_*", label_b: str = "q-locked (δq=0)"):
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
    fig.suptitle("Moisture binding of the heating→PV response — moist vs q-locked",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[response] wrote {out_png}")
    return fig


def plot_tseries_comparison(run_moist, run_qlock, out_png=None):
    """Model ΔPV extrema per iteration, moist (solid) vs q-locked (dashed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm = pv_response_series(run_moist)
    sq = pv_response_series(run_qlock)
    t_end = int(_pert_params(run_moist)["forcing_steps"]) * 3

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="grey", lw=0.8)
    if sm["hour"][0] < t_end < sm["hour"][-1]:
        ax.axvline(t_end, color="grey", lw=1.0, ls=":", label=f"heating ends ({t_end} h)")
    ax.plot(sm["hour"], sm["lowlevel_max"], color=_C_LOW, lw=2.0, marker="o", ms=4,
            label="moist  max ΔPV 700–1000 hPa")
    ax.plot(sq["hour"], sq["lowlevel_max"], color=_C_LOW, lw=2.0, ls="--", marker="o",
            ms=4, mfc="none", label="q-locked  max ΔPV 700–1000 hPa")
    ax.plot(sm["hour"], sm["upperlevel_min"], color=_C_UP, lw=2.0, marker="s", ms=4,
            label="moist  min ΔPV 200–500 hPa")
    ax.plot(sq["hour"], sq["upperlevel_min"], color=_C_UP, lw=2.0, ls="--", marker="s",
            ms=4, mfc="none", label="q-locked  min ΔPV 200–500 hPa")
    ax.set_xlabel("Iteration n  (nominal hour)", fontsize=13, weight="bold")
    ax.set_ylabel("ΔPV  [PVU]", fontsize=13, weight="bold")
    ax.set_xlim(sm["hour"][0], sm["hour"][-1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(experiment_title(run_moist,
                                  prefix="Moisture binding — moist vs q-locked:"),
                 fontsize=13, weight="bold")
    fig.text(0.995, 0.005,
             "gap between solid and dashed = the moisture-mediated share of the "
             "heating→PV response (δq pinned to control in the q-locked run)",
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.tight_layout()
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
    cs = sub.add_parser("compare-sweep", help="moist vs q-locked amplitude curves")
    cs.add_argument("category_dir")
    cs.add_argument("--lead", type=int, default=24)
    cs.add_argument("--out", default="pv_sweep_qlock_comparison.png")
    ct = sub.add_parser("compare-tseries", help="moist vs q-locked ΔPV series")
    ct.add_argument("run_moist")
    ct.add_argument("run_qlock")
    ct.add_argument("--out", default="pv_tseries_qlock_comparison.png")
    a = ap.parse_args()
    if a.cmd == "sweep":
        plot_amplitude_sweep(a.category_dir, a.out, lead_hr=a.lead, pattern=a.pattern)
    elif a.cmd == "maps":
        plot_sweep_maps(a.category_dir, a.init, a.out, amps=tuple(a.amps),
                        lead_hr=a.lead, per_K=a.per_K,
                        pattern=a.prefix + "_{amp}_{h}h_init{init}")
    elif a.cmd == "compare-sweep":
        plot_sweep_comparison(a.category_dir, a.out, lead_hr=a.lead)
    elif a.cmd == "compare-tseries":
        plot_tseries_comparison(a.run_moist, a.run_qlock, a.out)
    else:
        plot_pv_timeseries(a.run_dir, a.out)
