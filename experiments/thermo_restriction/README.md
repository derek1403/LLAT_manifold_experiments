# Experiment: Thermodynamic restriction (scaffold)

## Physical design

Constrain a thermodynamic field during the integration — e.g. cap column moisture,
fix the boundary-layer temperature, or clamp `tcwv`/`q` to a profile — to ask which
thermodynamic pathway the model relies on for intensification. Unlike the IC
interventions, this is naturally a **per-step** intervention: it re-imposes the
restriction every step inside the loop (`apply_step`), making it a good second test of
the hook API after diabatic heating.

## Status

Scaffold (no perturbation class yet). When implemented, add a `ThermoRestriction`
perturbation whose `apply_step` projects the relevant channel back onto the allowed
set each step, and register it in `perturbations/registry.py`.
