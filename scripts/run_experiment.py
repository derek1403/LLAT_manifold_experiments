#!/usr/bin/env python
"""Run one idealized experiment from a YAML config.

    python scripts/run_experiment.py --config experiments/diabatic_heating/configs/continuous_5K_7d.yaml

Resolves the perturbation + mode, creates the output directory, stamps the resolved
config (so outputs are self-documenting), loads the coupled models once, and runs the
driver. Supports single-level ``extends:`` for Base + Overrides configs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make the package importable whether or not it is pip-installed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llat_manifold import driver, io, operators  # noqa: E402
from llat_manifold.diagnostics.suite import standard_plots  # noqa: E402


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        cfg = _deep_merge(load_config(parent_path), cfg)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="experiment YAML")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + stamp config and write the README scaffold (no model run)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the standard diagnostic plot suite after the run")
    args = ap.parse_args(argv)

    cfg = load_config(args.config.resolve())
    category = cfg["category"]
    tag = cfg.get("tag", args.config.stem)
    run_name = io.make_run_name(cfg, tag)

    out_dir = io.experiment_output_dir(category, run_name, create=True,
                                       family=cfg.get("family"))
    io.stamp_config(out_dir, cfg)
    print(f"[run] category={category} run={run_name} mode={cfg['mode']} -> {out_dir}")

    if args.dry_run:
        io.generate_run_readme(out_dir, cfg)
        print("[run] --dry-run: config + README scaffold written; not running models.")
        return 0

    models = operators.load_models()
    saved = driver.run(cfg, models, out_dir)
    print(f"[run] model run done: {len(saved)} bundle(s) under {out_dir/'data'}")

    plots = [] if args.no_plots else standard_plots(out_dir)
    io.generate_run_readme(out_dir, cfg, plot_files=plots)
    print(f"[run] README + {len(plots)} figure(s) written under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
