# LLAT_manifold_experiments

Idealized **perturbation experiments** for the LLAT regional model (DLAMPty coupled
to FCNv2). This is **Part 2** of the RegionalCouple_AI split: Part 1
(`../claude_RegionalCouple_AI`) is the clean *operational* forecast pipeline; this
repo is the *experimental* sandbox.

## The thesis: model manifold vs. physical world

A trained AI weather model defines a **manifold** — the set of states and evolutions
it can represent, learned from data. The physical atmosphere obeys exact balances and
conservation laws (hydrostatic and gradient-wind balance, geostrophic adjustment,
potential-vorticity dynamics, gravity-wave radiation). The two need not coincide.
These experiments apply *controlled physical interventions* and ask: **does the LLAT
manifold respond the way physics says it must?** Where it does, the model has learned
dynamics; where it doesn't, we have mapped a gap.

## How it works (one paragraph)

We never copy the model code. The DLAMPty model, weights, IO and config patterns are
reused from Part 1 via an editable install of `regional_couple`. On top of that this
repo adds: the **model operator `M`** — one clean 3-hour DLAMPty step, no FCNv2 and no
coupling (`operators.py`); a **perturbation-evolution driver** with `snapshot`,
`continuous`, and `forward` modes (`driver.py`); a **hook/injection system**
(`perturbations/`) with `apply_ic` (initial-condition interventions) and `apply_step`
(per-step forcing `f`), plus **dynamic static-variable masking**; config-driven
experiments classified by physics (`experiments/`); self-documenting outputs (every run
stamps `config_used.yaml` + an auto-README); and advanced PV / divergence / wind /
gravity-wave diagnostics (`diagnostics/`).

> The method is a **finite-time *nonlinear* perturbation evolution**, *not* a strict
> "semi-linear"/tangent-linear method. See [`docs/perturbation_method.md`](docs/perturbation_method.md).

## Install (once)

```bash
conda activate pangu_env
pip install -e ../claude_RegionalCouple_AI   # Part 1: models, weights, io, config
pip install -e .                              # this repo
```

## Run an experiment

```bash
python scripts/run_experiment.py --config experiments/diabatic_heating/configs/continuous_5K_7d.yaml
# outputs/<category>/<tag>/  ->  delta_*.npz  +  config_used.yaml
```

Add `--dry-run` to resolve + stamp the config without launching the models.

## Diagnose

```bash
python -m llat_manifold.diagnostics.waves outputs/diabatic_heating/continuous_5K_7d --var msl --out hov.png
python -m llat_manifold.diagnostics.pv    outputs/diabatic_heating/continuous_5K_7d/<bundle>.npz --out theta.png
```

## Experiment catalogue

| category | status | intervention |
| --- | --- | --- |
| [`diabatic_heating`](experiments/diabatic_heating) | **implemented** | Gaussian heating: IC bump + continuous forcing + snapshot mode |
| [`sst_modification`](experiments/sst_modification) | runnable (demo) | uniform ΔSST; demonstrates dynamic masking |
| [`terrain`](experiments/terrain) | scaffold | remove/alter orography (`hgt`, `landmask`) |
| [`thermo_restriction`](experiments/thermo_restriction) | scaffold | per-step thermodynamic constraint |
| [`idealized_vortex`](experiments/idealized_vortex) | scaffold | axisymmetric vortex IC |

## Layout

```
config/        paths.yaml + model_layout.yaml (channel indices, static-lock set)
src/llat_manifold/
  operators.py   model operator M (single 3h DLAMPty step)
  driver.py      snapshot / continuous / forward modes + locking
  layout.py      variable indices, default lock, active_lock_indices (dynamic mask)
  perturbations/ Perturbation ABC + heating, sst, terrain, vortex, moisture, wind + registry
  diagnostics/   pv, divergence, wind_profile, hydrostatic, ike, waves, fields,
                 response, modal(reserved) — see diagnostics/README.md for how each is computed
experiments/   per-category README + configs (Base + Overrides via `extends:`)
scripts/run_experiment.py
docs/perturbation_method.md   tests/test_offline.py
```

## Tests

```bash
python tests/test_offline.py      # layout, dynamic masking, heating regression, driver bookkeeping (no models)
```
