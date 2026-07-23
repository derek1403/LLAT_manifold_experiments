"""Output-directory resolution, self-documenting config stamps, and array IO.

Array IO is reused verbatim from Part 1 (``regional_couple.io.arrays``): one
compressed ``.npz`` per array under the key ``data``, with transparent ``.npy``
fallback on read. This module adds the experiment output layout and the
"every output records the config that produced it" guarantee.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path

import yaml

# Reuse Part 1's unified array IO (editable-installed regional_couple).
from regional_couple.io.arrays import save_array, load_array, array_exists  # noqa: F401

from . import config


def experiment_output_dir(category: str, run_name: str, create: bool = True,
                          family: str | None = None) -> Path:
    """Resolve ``<output_root>/<category>[/<family>]/<run_name>`` (the run root).

    ``family`` groups related runs one level below the category (e.g. the
    diabatic_heating families ``heating_moist`` / ``heating_qlock`` /
    ``dq_measured`` / ``dq_latent`` / ``snapshot``); configs set it with a
    ``family:`` key. When ``create`` is set, also makes the mandated ``data/``
    and ``plots/`` subfolders so figures and arrays never mix at the run root.
    """
    d = config.output_root() / category
    if family:
        d = d / family
    d = d / run_name
    if create:
        (d / "data").mkdir(parents=True, exist_ok=True)
        (d / "plots").mkdir(parents=True, exist_ok=True)
    return d


def data_dir(run_dir) -> Path:
    """The ``data/`` subfolder of a run (all ``.npz`` bundles)."""
    d = Path(run_dir) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plots_dir(run_dir) -> Path:
    """The ``plots/`` subfolder of a run (all ``.png`` figures)."""
    d = Path(run_dir) / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def delta_basename(mode: str, param_tag: str, init_str: str, lead_hr: int) -> str:
    """Traceable output stem (no extension).

    Not valid-time-strict — encodes the experiment intent: mode, parameters, the
    init time, and the lead hour, e.g.::

        delta_continuous_5K_7d_init2025091700_lead072hr
    """
    return f"delta_{mode}_{param_tag}_init{init_str}_lead{lead_hr:03d}hr"


def save_delta_bundle(path, dlampty_upper, dlampty_sfc) -> str:
    """Save a DLAMPty perturbation/state bundle as one compressed ``.npz``.

    Keys ``dlampty_upper`` / ``dlampty_sfc``. (The experiments run DLAMPty alone, so
    there is no FCNv2 field.)
    """
    import numpy as np
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dlampty_upper=dlampty_upper, dlampty_sfc=dlampty_sfc)
    return path


def load_delta_bundle(path):
    """Load a bundle; returns ``(dlampty_upper, dlampty_sfc)``."""
    import numpy as np
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    with np.load(path) as z:
        return z["dlampty_upper"], z["dlampty_sfc"]


def _git_hash(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def make_run_name(cfg: dict, tag: str) -> str:
    """High-identification run name binding the IC nature + init time.

    Ensures the folder name reveals *what* the perturbation was added to, e.g.
    ``ic_bump_Deep_10K_init2025091700`` or ``10K_ideal_vortex_init2025091700``.
    The init token is appended unless ``tag`` already carries it.
    """
    init = str(cfg.get("init_time", "")).strip()
    name = tag
    if init and f"init{init}" not in name and init not in name:
        name = f"{name}_init{init}"
    return name


def _ic_description(cfg: dict) -> str:
    """Human description of the initial field the perturbation was applied to."""
    tc = cfg.get("tc_id", "?")
    init = cfg.get("init_time", "?")
    pert = cfg.get("perturbation", {}) or {}
    ptype = pert.get("type", "none")
    if ptype == "vortex":
        return f"idealized vortex `{pert.get('vortex_name', '?')}` (TC {tc}, geometry init {init})"
    return f"real-case background IC for TC **{tc}**, init **{init}** (FCNv2 analysis + DLAMPty combined NetCDF)"


def _readme_figures(figs):
    """Trim the inline-figure list for a friendlier README.

    Drops animated GIFs and, for each per-iteration *evolution* sequence
    (``plots/evolution/<diag>/<frame>.png``), keeps only the first and last frame.
    The full frame set and the GIFs stay on disk — they are simply not embedded in
    the README (which otherwise inlines 100+ images and bloats the repo).
    """
    figs = [Path(p) for p in figs]
    seq: dict[Path, list[str]] = {}
    for f in figs:
        if f.suffix.lower() != ".gif" and f.parent.parent.name == "evolution":
            seq.setdefault(f.parent, []).append(f.name)
    keep = {d: {min(names), max(names)} for d, names in seq.items()}
    out = []
    for f in figs:
        if f.suffix.lower() == ".gif":
            continue
        if f.parent.parent.name == "evolution" and f.name not in keep[f.parent]:
            continue
        out.append(f)
    return out


def generate_run_readme(run_dir, cfg: dict, plot_files=None) -> Path:
    """Write an auto-documented ``README.md`` at the run root.

    Includes the data source, run info (mode/steps/perturbation params), and Markdown
    image links to the figures in ``plots/``. To keep the README (and the repo) light,
    per-iteration *evolution* sequences are reduced to their first and last frame and
    GIFs are not embedded — see :func:`_readme_figures`.
    """
    run_dir = Path(run_dir)
    pert = cfg.get("perturbation", {}) or {}
    mode = cfg.get("mode", "?")

    horizon = {}
    for k in ("total_steps", "iterations", "fore_hour"):
        if k in cfg:
            horizon[k] = cfg[k]

    # Discover figures actually present (prefer the passed list, else scan plots/
    # recursively to include the evolution frames). Top-level figures first.
    plots = run_dir / "plots"
    if plot_files:
        figs = [Path(p) for p in plot_files]
    else:
        figs = list(plots.rglob("*.png")) + list(plots.rglob("*.gif"))
        figs.sort(key=lambda f: (len(f.relative_to(plots).parts), f.as_posix()))
    figs = _readme_figures(figs)

    lines = [
        f"# Run: {cfg.get('category', '?')} / {run_dir.name}",
        "",
        f"_Auto-generated by `scripts/run_experiment.py`._",
        "",
        "## Data source",
        "",
        f"- {_ic_description(cfg)}",
        "",
        "## Run info",
        "",
        f"- **Mode:** `{mode}`",
        f"- **Horizon:** " + (", ".join(f"`{k}={v}`" for k, v in horizon.items()) or "n/a"),
        f"- **Perturbation:** `{pert.get('type', 'none')}` — "
        + (", ".join(f"{k}={v}" for k, v in pert.items() if k != "type") or "(no params)"),
        "",
        "## Visuals",
        "",
    ]
    if figs:
        for f in figs:
            rel = f.relative_to(run_dir) if f.is_absolute() else f
            # Label evolution frames by their diagnostic subfolder (e.g. PV_Theta/iter001).
            label = f"{f.parent.name}/{f.stem}" if f.parent.parent.name == "evolution" else f.stem
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"![{label}]({rel.as_posix()})")
            lines.append("")
    else:
        lines.append("_No figures generated yet. Run the diagnostics suite._")
        lines.append("")

    lines += [
        "## Files",
        "",
        "- `data/` — perturbation/state bundles (`*.npz`, keys "
        "`fcnv2` / `dlampty_upper` / `dlampty_sfc`).",
        "- `plots/` — figures linked above.",
        "- `config_used.yaml` — the exact resolved config + provenance.",
        "",
    ]
    dest = run_dir / "README.md"
    dest.write_text("\n".join(lines))
    return dest


def stamp_config(out_dir: Path, resolved_config: dict) -> Path:
    """Write the resolved experiment config + provenance to ``config_used.yaml``.

    Guarantees no output is ever orphaned from the parameters that produced it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = {
        "resolved_config": resolved_config,
        "provenance": {
            "stamped_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "llat_manifold_git": _git_hash(config.repo_root()),
        },
    }
    dest = out_dir / "config_used.yaml"
    with open(dest, "w") as f:
        yaml.safe_dump(stamp, f, sort_keys=False, allow_unicode=True)
    return dest
