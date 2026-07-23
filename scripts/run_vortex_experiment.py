#!/usr/bin/env python
"""Run one ring-vortex barotropic-instability experiment from a YAML config.

    python scripts/run_vortex_experiment.py --config experiments/idealized_vortex/configs/base.yaml

Isolated twin of ``run_experiment.py`` for the idealized_vortex category: same
config resolution (single-level ``extends:``), output layout and config stamp,
but drives :mod:`llat_manifold.idealized_vortex.driver_vortex` (snapshot loop
with ``background_npz`` + boundary frame relaxation) and finishes with the
azimuthal-wavenumber diagnostics instead of the shared plot suite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable whether or not it is pip-installed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llat_manifold import io, operators                      # noqa: E402
from llat_manifold.idealized_vortex import azimuthal, driver_vortex  # noqa: E402
from run_experiment import load_config                       # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="experiment YAML")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + stamp config and write the README scaffold (no model run)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the azimuthal diagnostics after the run")
    args = ap.parse_args(argv)

    cfg = load_config(args.config.resolve())
    category = cfg["category"]
    tag = cfg.get("tag", args.config.stem)
    run_name = io.make_run_name(cfg, tag)

    out_dir = io.experiment_output_dir(category, run_name, create=True,
                                       family=cfg.get("family"))
    io.stamp_config(out_dir, cfg)
    print(f"[vortex] category={category} run={run_name} mode={cfg.get('mode')} "
          f"-> {out_dir}")

    if args.dry_run:
        io.generate_run_readme(out_dir, cfg)
        print("[vortex] --dry-run: config + README scaffold written; not running models.")
        return 0

    models = operators.load_models()
    saved = driver_vortex.run(cfg, models, out_dir)
    print(f"[vortex] model run done: {len(saved)} bundle(s) under {out_dir/'data'}")

    plots = []
    if not args.no_plots:
        res = azimuthal.analyze_run(out_dir)
        plots = res.get("plots", [])
        print(f"[vortex] dominant m = {res['m_dominant']}; σ/day = "
              f"{ {m: round(s, 3) for m, s in res['sigma_per_day'].items() if s == s} }")
    io.generate_run_readme(out_dir, cfg, plot_files=plots)
    print(f"[vortex] README + {len(plots)} figure(s) written under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
