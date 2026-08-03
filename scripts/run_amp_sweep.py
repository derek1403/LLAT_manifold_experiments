#!/usr/bin/env python
"""Batch amplitude sweep of the diabatic-heating experiment (continuous mode).

    python scripts/run_amp_sweep.py --amps-range 5 10 0.5 \
        --inits 2025091700 2025092000 --steps 8 --forcing-steps 8

Builds one continuous-mode config per (init_time, amp_K) on top of the category
``base.yaml``, loads DLAMPty once, and runs them back to back. Each run lands in
``outputs/diabatic_heating/sweep_<amp>_<hours>h_init<init>/`` with the usual
``config_used.yaml`` + auto-README stamping. Runs whose ``data/`` already holds a
complete set of bundles are skipped (``--force`` reruns them), so refining the sweep
later (e.g. 0.1 K spacing over a curved interval) is an incremental call.

The per-step injected bump is ``amp_K / forcing_steps`` (``amp_mode: spread``): the
full ``amp_K`` is injected over the first ``forcing_steps`` steps, after which the
perturbation evolves freely. Keep ``forcing_steps`` identical across sweep and
long time-series runs so their first ``forcing_steps`` leads are directly comparable.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for p in (str(_SRC), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_experiment import load_config  # noqa: E402
from llat_manifold import driver, io, operators  # noqa: E402
from llat_manifold.perturbations.heating import _amp_tag  # noqa: E402

_CFG_DIR = _HERE.parent / "experiments" / "diabatic_heating" / "configs"
_BASE_CFG = _CFG_DIR / "base.yaml"
_DQ_MEASURED = _CFG_DIR / "dq_measured.yaml"

# --ic axisym: same families, same tag prefixes, separate output category, and the
# prebuilt azimuthally averaged IC per init. Keeping the names identical means every
# downstream tag glob and figure script works on either set unchanged.
_AXISYM_CATEGORY = "diabatic_heating_axisym"
_AXISYM_DQ = _CFG_DIR / "dq_measured_axisym.yaml"


def _axisym_ic(tc_id: str, init_time: str, env_from: str | None = None) -> str:
    from llat_manifold import config
    stem = f"axisym_{tc_id}_{init_time}"
    if env_from and env_from != init_time:
        # Same-vortex, reference-environment member of an intensity ladder: the
        # atmosphere is init_time's, the environment (SST/f/radiation/lat-lon/clock)
        # is env_from's. The run folder still carries init_time (the vortex label);
        # only the physics clock, read from the npz, is env_from's.
        stem += f"_env{env_from}"
    p = config.output_root() / "axisymmetric_ic" / f"{stem}.npz"
    if not p.exists():
        hint = (f"--tc-id {tc_id} --inits {init_time}"
                + (f" --env-from {env_from}" if env_from else ""))
        raise FileNotFoundError(
            f"{p} not found — build it first with:\n"
            f"  python scripts/make_axisymmetric_ic.py {hint}")
    return str(p)


def _dq_measured(init_time: str, layer: str = "full", table=None) -> dict:
    """Per-init measured δq scaling (gkg_per_K + 13-level profile) for --pert moisture.

    ``layer`` restricts the vertical profile for the layered-injection probe:
    ``bl`` keeps 850–1000 hPa, ``ft`` keeps 400–700 hPa. The masked profile is
    rescaled so its mass-weighted column ∫δq dm equals the FULL profile's at the
    same nominal amp (equal-column normalisation — PV response and the energy
    budget are column quantities; the price is a different in-layer peak).
    """
    import yaml
    path = pathlib.Path(table) if table else _DQ_MEASURED
    with open(path) as fh:
        tbl = yaml.safe_load(fh)
    if init_time not in tbl:
        raise KeyError(f"{path} has no entry for init {init_time!r}")
    entry = dict(tbl[init_time])
    if layer == "full":
        return entry
    import numpy as np
    p = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                 dtype=float)
    edges = np.concatenate([[p[0]], (p[:-1] + p[1:]) / 2, [p[-1]]])
    dm = np.diff(edges)                          # ∝ mass per level (g cancels)
    prof = np.asarray(entry["profile"], dtype=float)
    mask = (p >= 850) if layer == "bl" else ((p >= 400) & (p <= 700))
    lay = np.where(mask, prof, 0.0)
    lay *= float(np.sum(prof * dm) / np.sum(lay * dm))   # equal column amount
    entry["profile"] = [round(float(v), 4) for v in lay]
    return entry


def build_config(base_cfg: dict, *, init_time: str, amp_K: float, steps: int,
                 forcing_steps: int, heat_type: str, sigma: float,
                 tag_prefix: str, family: str, locks=(),
                 pert: str = "heating", dq_scaling: str = "measured",
                 dq_layer: str = "full", dq_offset=(0, 0),
                 category: str | None = None, ic_npz: str | None = None,
                 dq_table: str | None = None) -> dict:
    cfg = dict(base_cfg)
    cfg["init_time"] = init_time
    if category:
        cfg["category"] = category
    if ic_npz:
        # Prebuilt idealized IC; the driver loads it instead of the analysis.
        cfg["ic_npz"] = ic_npz
    cfg["mode"] = "continuous"
    cfg["total_steps"] = steps
    cfg["tag"] = f"{tag_prefix}_{_amp_tag(amp_K)}_{steps * 3}h"
    # Runs are grouped one level below the category by experiment family.
    cfg["family"] = family
    if locks:
        # Pin these upper-channel deltas back to the control every step (e.g. [q]
        # denies moisture co-evolution; [t] denies the temperature response).
        cfg["lock_upper_vars"] = list(locks)
    cfg["perturbation"] = {
        "type": pert,
        "injection": "per_step",
        "amp_K": float(amp_K),
        "forcing_steps": forcing_steps,
        "amp_mode": "spread",
        "sigma": sigma,
    }
    if pert == "heating":
        cfg["perturbation"]["heat_type"] = heat_type
    elif dq_scaling == "measured":
        # δq-only reverse probe: inject the δq the moist amp_K heating run grew
        # (optionally restricted to one layer, equal column amount).
        cfg["perturbation"].update(
            _dq_measured(init_time, layer=dq_layer, table=dq_table))
    if pert == "moisture" and tuple(dq_offset) != (0, 0):
        cfg["perturbation"]["offset_pts"] = list(dq_offset)
    else:                                     # latent-equivalent: cp/Lv per K
        cfg["perturbation"]["heat_type"] = heat_type
    return cfg


# The four standard (tag_prefix, family) pairs; anything beyond them must be
# named explicitly with --tag-prefix AND --family (prefixes stay disjoint so
# response.py tag patterns never cross families).
_STANDARD = {
    ("heating", "", "measured"):  ("sweep",    "heating_moist"),
    ("heating", "q", "measured"): ("sweepq",   "heating_qlock"),
    ("moisture", "", "measured"): ("sweepdq",  "dq_measured"),
    ("moisture", "", "latent"):   ("sweepdql", "dq_latent"),
}


def _is_complete(out_dir: Path, steps: int) -> bool:
    """A continuous run is complete when every lead has its delta + control pair."""
    data = out_dir / "data"
    return (len(list(data.glob("delta_continuous_*.npz"))) >= steps
            and len(list(data.glob("control_continuous_*.npz"))) >= steps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    amps = ap.add_mutually_exclusive_group(required=True)
    amps.add_argument("--amps", nargs="+", type=float, help="explicit amp_K list [K]")
    amps.add_argument("--amps-range", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                      help="inclusive amp_K range, e.g. 5 10 0.5")
    ap.add_argument("--inits", nargs="+", default=["2025091700", "2025092000"],
                    help="init times YYYYMMDDHH (default: weak 0917 + strong 0920)")
    ap.add_argument("--steps", type=int, default=8, help="total 3 h steps (8 = 24 h)")
    ap.add_argument("--forcing-steps", type=int, default=8,
                    help="steps over which amp_K is spread (default 8 = 24 h)")
    ap.add_argument("--heat-type", default="Deep", choices=["Deep", "Shallow", "Stratiform"])
    ap.add_argument("--sigma", type=float, default=5.0, help="Gaussian sigma [grid pts]")
    ap.add_argument("--pert", default="heating", choices=["heating", "moisture"],
                    help="perturbation type: heating (ΔT) or moisture (δq-only reverse probe)")
    ap.add_argument("--dq-scaling", default="measured", choices=["measured", "latent"],
                    help="moisture amplitude scaling: measured (dq_measured.yaml, per init) "
                         "or latent-equivalent (cp/Lv per K, heat_type profile)")
    ap.add_argument("--dq-layer", default="full", choices=["full", "bl", "ft"],
                    help="restrict the measured δq profile to the boundary layer "
                         "(850–1000 hPa) or free troposphere (400–700 hPa), rescaled "
                         "to the same column amount (layered-injection probe)")
    ap.add_argument("--dq-offset", nargs=2, type=int, default=[0, 0], metavar=("DY", "DX"),
                    help="shift the injection centre by (DY, DX) grid points from the "
                         "domain centre (0.25°/pt) — the off-vortex state-dependence probe")
    ap.add_argument("--tag-prefix", default=None,
                    help="run-folder prefix (default: sweep / sweepq with --lock-q / "
                         "sweepdq / sweepdql for moisture measured/latent)")
    ap.add_argument("--lock", nargs="+", default=[], metavar="VAR",
                    choices=["u", "v", "t", "q", "z", "w"],
                    help="pin these upper-channel deltas to the control each step "
                         "(lock_upper_vars); e.g. --lock q, --lock t, --lock w")
    ap.add_argument("--lock-q", action="store_true",
                    help="shorthand for --lock q; with the default prefix the tag "
                         "becomes sweepq_* so moist runs are never overwritten")
    ap.add_argument("--family", default=None,
                    help="output family folder under outputs/<category>/ "
                         "(default: derived for the four standard combos; required "
                         "for any new pert/lock combination)")
    ap.add_argument("--ic", choices=["ragasa", "axisym"], default="ragasa",
                    help="initial condition: ragasa = the real analysis (default, "
                         "category diabatic_heating); axisym = the azimuthally "
                         "averaged idealized vortex from "
                         "scripts/make_axisymmetric_ic.py (category "
                         "diabatic_heating_axisym, same families and tag prefixes)")
    ap.add_argument("--env-from", default=None,
                    help="with --ic axisym: build an intensity-ladder member whose "
                         "vortex is --inits but whose environment (SST / f / radiation "
                         "/ lat-lon / clock) is this reference init's — i.e. load "
                         "axisym_<tc>_<init>_env<envfrom>.npz. Holds the environment "
                         "fixed as a controlled constant across the vortices. Build "
                         "these with make_axisymmetric_ic.py --env-from first.")
    ap.add_argument("--ic-npz", default=None,
                    help="any prebuilt InitialState npz to start from, overriding "
                         "--ic — e.g. one member of the intensity family written by "
                         "scripts/make_vortex_intensity_ic.py. Requires --category so "
                         "the runs land somewhere of their own; --inits then only "
                         "supplies the time stamp the npz was derived from.")
    ap.add_argument("--category", default=None,
                    help="output category under outputs/ (default: from base.yaml, or "
                         "diabatic_heating_axisym with --ic axisym). Required with "
                         "--ic-npz.")
    ap.add_argument("--dq-table", default=None,
                    help="measured-\u03b4q YAML to read (default: dq_measured.yaml, or "
                         "dq_measured_axisym.yaml with --ic axisym)")
    ap.add_argument("--dry-run", action="store_true",
                    help="stamp configs + READMEs only; no model runs")
    ap.add_argument("--force", action="store_true", help="rerun complete runs")
    ap.add_argument("--plots", action="store_true",
                    help="also render the per-run standard plot suite (slow)")
    args = ap.parse_args(argv)

    if args.amps_range:
        start, stop, step = args.amps_range
        amp_list, a = [], start
        while a <= stop + 1e-9:
            amp_list.append(round(a, 6))
            a += step
    else:
        amp_list = args.amps

    locks = sorted(set(args.lock) | ({"q"} if args.lock_q else set()))
    combo = (args.pert, "".join(locks),
             args.dq_scaling if args.pert == "moisture" else "measured")
    std = _STANDARD.get(combo)
    if args.dq_layer != "full":
        # Layered injection is its own family — never reuses dq_measured names.
        std = ((f"sweepdq{args.dq_layer}", f"dq_{args.dq_layer}")
               if std == ("sweepdq", "dq_measured") else None)
    if tuple(args.dq_offset) != (0, 0):
        # Off-centre injection is its own family too.
        std = (("sweepdqoff", "dq_offcore")
               if std == ("sweepdq", "dq_measured") else None)
    if std is None and not (args.tag_prefix and args.family):
        ap.error(f"non-standard combo pert={args.pert} lock={locks} "
                 f"dq_scaling={args.dq_scaling}: pass BOTH --tag-prefix and --family "
                 f"(pick a tag prefix disjoint from sweep/sweepq/sweepdq/sweepdql)")
    if args.tag_prefix:
        tag_prefix = args.tag_prefix
        if locks == ["q"] and args.pert == "heating" and not tag_prefix.endswith("q"):
            tag_prefix += "q"          # legacy: tseries -> tseriesq
    else:
        tag_prefix = std[0]
    family = args.family or (std[1] if std else None)

    base_cfg = load_config(_BASE_CFG)
    tc_id = str(base_cfg.get("tc_id", "202518W"))
    if args.ic_npz and not args.category:
        ap.error("--ic-npz needs --category: an arbitrary IC must not write into the "
                 "diabatic_heating / diabatic_heating_axisym trees, whose runs are "
                 "indexed by --ic elsewhere")
    if args.ic_npz and not pathlib.Path(args.ic_npz).exists():
        ap.error(f"--ic-npz {args.ic_npz} not found")
    axisym = args.ic == "axisym" and not args.ic_npz
    if args.env_from and not axisym:
        ap.error("--env-from only applies to --ic axisym (it selects the "
                 "reference-environment IC axisym_<tc>_<init>_env<envfrom>.npz)")
    category = args.category or (_AXISYM_CATEGORY if axisym else None)
    dq_table = args.dq_table or (str(_AXISYM_DQ) if axisym else None)
    if axisym and args.pert == "moisture" and args.dq_scaling == "measured" \
            and not pathlib.Path(dq_table).exists():
        ap.error(f"{dq_table} not found — the axisymmetric \u03b4q scaling is measured "
                 f"off the axisymmetric moist runs, so run those first, then:\n"
                 f"  python scripts/measure_dq_scaling.py --ic axisym")
    jobs = []
    for init in args.inits:
        for amp in amp_list:
            cfg = build_config(base_cfg, init_time=str(init), amp_K=amp,
                               steps=args.steps, forcing_steps=args.forcing_steps,
                               heat_type=args.heat_type, sigma=args.sigma,
                               tag_prefix=tag_prefix, family=family, locks=locks,
                               pert=args.pert, dq_scaling=args.dq_scaling,
                               dq_layer=args.dq_layer, dq_offset=tuple(args.dq_offset),
                               category=category, dq_table=dq_table,
                               ic_npz=args.ic_npz or (
                                   _axisym_ic(tc_id, str(init), args.env_from)
                                   if axisym else None))
            jobs.append(cfg)

    print(f"[sweep] ic={args.ic_npz or args.ic}"
          f"{' env=' + args.env_from if args.env_from else ''} "
          f"category={category or base_cfg['category']}")
    print(f"[sweep] {len(jobs)} run(s): amps={amp_list} inits={list(args.inits)} "
          f"steps={args.steps} forcing_steps={args.forcing_steps} pert={args.pert} "
          f"lock={locks} layer={args.dq_layer} prefix={tag_prefix} family={family}")

    models = None
    n_done = n_skip = 0
    for i, cfg in enumerate(jobs, 1):
        run_name = io.make_run_name(cfg, cfg["tag"])
        out_dir = io.experiment_output_dir(cfg["category"], run_name, create=True,
                                           family=cfg.get("family"))
        if not args.force and _is_complete(out_dir, args.steps):
            print(f"[sweep] {i}/{len(jobs)} {run_name}: complete, skipping")
            n_skip += 1
            continue
        io.stamp_config(out_dir, cfg)
        print(f"[sweep] {i}/{len(jobs)} {run_name} -> {out_dir}")
        if args.dry_run:
            io.generate_run_readme(out_dir, cfg)
            continue
        if models is None:
            models = operators.load_models()
        saved = driver.run(cfg, models, out_dir)
        plots = []
        if args.plots:
            from llat_manifold.diagnostics.suite import standard_plots
            plots = standard_plots(out_dir)
        io.generate_run_readme(out_dir, cfg, plot_files=plots)
        print(f"[sweep] {run_name}: {len(saved)} bundle(s)")
        n_done += 1

    print(f"[sweep] done: {n_done} ran, {n_skip} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
