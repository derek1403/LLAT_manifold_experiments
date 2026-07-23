#!/usr/bin/env python
"""Build the quiescent horizontally-uniform background for the ring-vortex runs.

    python scripts/make_quiescent_background.py --tc-id 202518W --init 2025091700

Loads the real-analysis DLAMPty IC for (tc_id, init), flattens it into a
quiescent uniform-ocean environment (winds→0, thermo→domain-mean profiles,
hgt/landmask→0, sst→mean; f/solar/time encodings/lat-lon kept) and saves the
npz consumed via the ``background_npz`` config key. Default output:
``outputs/idealized_vortex/backgrounds/quiescent_<tc>_<init>.npz``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llat_manifold import operators                          # noqa: E402
from llat_manifold.idealized_vortex import background as bg  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tc-id", default="202518W")
    ap.add_argument("--init", default="2025091700")
    ap.add_argument("--out", default=None, help="output npz (default under outputs/)")
    args = ap.parse_args(argv)

    dlampty = operators.load_models()
    path = bg.build_and_save(args.tc_id, args.init, dlampty, args.out)
    print(f"[background] quiescent background saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
