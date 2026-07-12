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

_BASE_CFG = _HERE.parent / "experiments" / "diabatic_heating" / "configs" / "base.yaml"


def build_config(base_cfg: dict, *, init_time: str, amp_K: float, steps: int,
                 forcing_steps: int, heat_type: str, sigma: float,
                 tag_prefix: str, lock_q: bool = False) -> dict:
    cfg = dict(base_cfg)
    cfg["init_time"] = init_time
    cfg["mode"] = "continuous"
    cfg["total_steps"] = steps
    cfg["tag"] = f"{tag_prefix}_{_amp_tag(amp_K)}_{steps * 3}h"
    if lock_q:
        # Deny the perturbation any moisture co-evolution: δq is pinned to the
        # control every step, isolating the non-moisture-mediated PV response.
        cfg["lock_upper_vars"] = ["q"]
    cfg["perturbation"] = {
        "type": "heating",
        "injection": "per_step",
        "amp_K": float(amp_K),
        "heat_type": heat_type,
        "forcing_steps": forcing_steps,
        "amp_mode": "spread",
        "sigma": sigma,
    }
    return cfg


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
    ap.add_argument("--tag-prefix", default=None,
                    help="run-folder prefix (default: sweep, or sweepq with --lock-q)")
    ap.add_argument("--lock-q", action="store_true",
                    help="pin δq to the control each step (lock_upper_vars: [q]); "
                         "tag prefix gains a 'q' suffix so moist runs are never overwritten")
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

    tag_prefix = args.tag_prefix or "sweep"
    if args.lock_q and not args.tag_prefix:
        tag_prefix = "sweepq"
    elif args.lock_q and not tag_prefix.endswith("q"):
        tag_prefix += "q"

    base_cfg = load_config(_BASE_CFG)
    jobs = []
    for init in args.inits:
        for amp in amp_list:
            cfg = build_config(base_cfg, init_time=str(init), amp_K=amp,
                               steps=args.steps, forcing_steps=args.forcing_steps,
                               heat_type=args.heat_type, sigma=args.sigma,
                               tag_prefix=tag_prefix, lock_q=args.lock_q)
            jobs.append(cfg)

    print(f"[sweep] {len(jobs)} run(s): amps={amp_list} inits={list(args.inits)} "
          f"steps={args.steps} forcing_steps={args.forcing_steps} "
          f"lock_q={args.lock_q} prefix={tag_prefix}")

    models = None
    n_done = n_skip = 0
    for i, cfg in enumerate(jobs, 1):
        run_name = io.make_run_name(cfg, cfg["tag"])
        out_dir = io.experiment_output_dir(cfg["category"], run_name, create=True)
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
