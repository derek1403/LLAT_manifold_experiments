"""Map a perturbation ``type`` (from an experiment yaml) to a built instance.

Config shape::

    perturbation:
      type: heating
      injection: per_step
      amp_K: 5.0
      forcing_steps: 28

The ``type`` selects the class; the remaining keys are passed as keyword arguments
to its constructor.
"""
from __future__ import annotations

from .base import Perturbation
from .heating import HeatingPerturbation
from .sst import SSTPerturbation
from .terrain import TerrainPerturbation
from .vortex import VortexPerturbation
from .moisture import MoisturePerturbation
from .wind import WindPerturbation

_REGISTRY = {
    "heating": HeatingPerturbation,
    "sst": SSTPerturbation,
    "terrain": TerrainPerturbation,
    "vortex": VortexPerturbation,
    "moisture": MoisturePerturbation,
    "wind": WindPerturbation,
    "none": Perturbation,
}


def build_perturbation(cfg: dict) -> Perturbation:
    """Construct the Perturbation named by ``cfg['type']`` with the other keys."""
    cfg = dict(cfg or {})
    ptype = cfg.pop("type", "none")
    if ptype not in _REGISTRY:
        raise KeyError(f"Unknown perturbation type {ptype!r}. "
                       f"Known: {sorted(_REGISTRY)}")
    return _REGISTRY[ptype](**cfg)


def register(name: str, cls) -> None:
    """Register a custom Perturbation subclass under ``name``."""
    _REGISTRY[name] = cls
