"""Idealized ring-vortex barotropic-instability experiment (isolated subpackage).

Everything specific to the ``idealized_vortex`` experiment lives here so the
shared modules (``driver.py``, ``perturbations/``, ``diagnostics/suite.py``)
stay untouched:

  * :mod:`.balance`       — analytic ring ζ(r) → V(r) → gradient-wind Φ′ +
    hydrostatic T′ construction (the balanced IC builder)
  * :mod:`.ring`          — :class:`RingVortexPerturbation` (``apply_ic`` only)
  * :mod:`.background`    — quiescent horizontally-uniform background builder
  * :mod:`.driver_vortex` — snapshot loop with ``background_npz`` + optional
    boundary frame relaxation
  * :mod:`.azimuthal`     — azimuthal-wavenumber (m) decomposition diagnostics

Entry points: ``scripts/make_quiescent_background.py``,
``scripts/run_vortex_experiment.py``, ``scripts/plot_ic_vs_ragasa.py``.
"""
