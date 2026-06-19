# Part 2 — `LLAT_manifold_experiments`: idealized-experiment repo

## Context

`/wk2/pc/AI_models/RegionalCouple_AI` was split into two clean repos. **Part 1**
(`/wk2/pc/AI_models/claude_RegionalCouple_AI`) is the operational typhoon-forecast
mirror — done. **Part 2** is this plan: a separate, modular repo for **idealized
experiments** that probe the gap between the LLAT (DLAMPty / coupled FCNv2) regional
model's *manifold* and the *physical world*.

The experiments today are scattered:
- `RegionalCouple_AI/idealized_exp/` — IC-injection experiments
  (`LLAT_add_heat_onestep.py` one-time Gaussian heating bump on upper-T;
  `idealize_vortex` IC overwrite) + PV/θ, divergence, wind-profile plotters.
- `/wk2/pc/AI_models/semi-linear/` — the perturbation-difference method (control vs
  perturbation) with `exp_heating`, `exp_continuous_heating`, `exp_wind_init`,
  `exp_moisture_init`, `exp_pacific_heating`. This is the **default technique** and
  must be first-class in the new repo.

Goal: one repo, `LLAT_manifold_experiments`, with a thin **Core Base** (reusing
Part 1's `regional_couple` package) plus a **perturbation-evolution driver**, a
**hook/injection system with dynamic static-variable masking**, config-driven
experiment definitions classified by physics, self-documenting outputs, and advanced
PV/divergence/wave diagnostics left out of Part 1. First end-to-end slice:
**diabatic heating** (IC bump + per-step continuous forcing), fully exercising the
hook API.

### Decisions (confirmed with user)
- **Repo name:** `LLAT_manifold_experiments`.
- **Core Base reuse (editable install):** add a `pyproject.toml` to **Part 1's root**
  (`claude_RegionalCouple_AI/`), then `pip install -e ../claude_RegionalCouple_AI`
  in the current env. This repo imports `regional_couple` for model loaders, `io`,
  config patterns; reuses Part 1's `vendor/` + `weights/` via config paths. **No
  copying of model code.** Adding `pyproject.toml` is packaging-only and does not
  touch Part 1's operational logic.
- **Driver:** this repo owns its own driver; Part 1 stays untouched/operational.
- **First intervention:** diabatic heating, end-to-end. Other categories get
  directory + README/config templates + registered stubs only.
- **Terminology:** the 6h operator `S` is **not** strictly "semi-linear" (see docs
  section). Internal modules avoid implying it is.
- Run environment: `pangu_env` (per project memory).

## Repo architecture (Base + Overrides)

```
LLAT_manifold_experiments/
├── README.md                       # manifold-vs-physics thesis; index of experiments
├── pyproject.toml / requirements   # depends on regional_couple (editable)
├── config/
│   ├── paths.yaml                  # IC roots, weights (reuse Part1), output roots
│   └── model_layout.yaml           # variable indices; default static-lock set; per-var names
├── docs/
│   ├── perturbation_method.md      # ← co-write; rigorous derivation (see docs section)
│   └── PLAN_repo2.md               # this plan
├── src/llat_manifold/
│   ├── config.py                   # mirror Part1 config.py (yaml + repo-relative resolve)
│   ├── layout.py                   # var-index helpers; default lock set; active_lock(perturbation)
│   ├── operators.py                # super_operator S: FCNv2 1 step + DLAMPty 2 substeps + couple
│   ├── driver.py                   # perturbation-evolution loop: snapshot + continuous; calls hooks
│   ├── perturbations/
│   │   ├── base.py                 # Perturbation ABC: apply_ic / apply_step / claimed_static_vars
│   │   ├── heating.py              # Gaussian diabatic heating (IC bump + continuous forcing)
│   │   ├── vortex.py sst.py terrain.py moisture.py wind.py   # vortex=port; rest=registered stubs
│   │   └── registry.py             # name -> Perturbation, built from config
│   ├── io.py                       # reuse regional_couple.io.arrays; output-dir resolver + config stamp
│   └── diagnostics/
│       ├── pv.py  divergence.py  wind_profile.py  response.py
│       ├── waves.py                # gravity-wave radiation; Rossby-radius adjustment (see §3)
│       └── modal.py                # reserved: EOF / higher-moment modal analysis
├── experiments/
│   ├── diabatic_heating/           # ← FIRST, fully implemented
│   │   ├── README.md               # physics: balance adjustment, PV generation by heating
│   │   ├── configs/{base, ic_bump_Deep_10K, continuous_5K_7d, snapshot_5K_iter20}.yaml
│   │   └── outputs/ -> symlink to big-storage root
│   ├── sst_modification/ terrain/ thermo_restriction/ idealized_vortex/   # scaffolds
└── scripts/
    └── run_experiment.py           # --config <yaml>; resolve perturbation+mode, run, stamp config
```

## The core operator and perturbation evolution (the technical heart)

Re-implemented cleanly from the two reference scripts.

**1. Super operator `S`** (`operators.py`): FCNv2 `predict_one_step` (6h) + DLAMPty
`predict_one_step` ×2 (0→3h, 3→6h) + `transfer_FCNV2_DLAMPty_with_radius(radius=7.5)`.
`changing_additional_information` is called in **continuous** mode (time advances) and
skipped in **snapshot** mode (time frozen). Model loaders come from
`regional_couple.inference.models`.

**2. Finite-time perturbation evolution** (`driver.py`): maintain control state `I`
and accumulated perturbation `δ`; each step synthesize `A = I + δ`, integrate both
(`I' = S(I)`, `A' = S(A)`), peel `δ = A' − I'`. Two modes:
- `snapshot`: time frozen, iterate N times at one valid time (power-iteration-style
  amplification of the fastest-growing finite-time mode). Port of
  `run_snapshot_semilinear.py`.
- `continuous`: time-marching, forcing added each step (`amp/heating_steps`). Port of
  `exp_continuous_heating/run_continuous_heating.py`. **Fix the latent bug**: `A = I +
  δ` must be synthesized *before* `np.ascontiguousarray`/locking (current script
  references `A_fcn` before assignment).

## Hook / injection system + dynamic static masking

`Perturbation` ABC (`perturbations/base.py`):
- `apply_ic(upper, surface, fcn, ctx)` — modify t=0 fields (SST, terrain, vortex, IC
  heating bump). Attaches where Part 1 does `IC_from_xarray_to_npy`.
- `apply_step(δ_upper, δ_surface, δ_fcn, ctx)` — per-step forcing in the loop
  (continuous heating); `ctx` carries `fore_i`, TC center, lead time.
- `claimed_static_vars() -> list` — **dynamic masking**: declares which static
  variables this experiment *takes over*, so the driver releases them from the
  default lock.

**Static-variable locking with dynamic masking** (`layout.py` + `model_layout.yaml`):
- `model_layout.yaml` defines the **default lock set**: DLAMPty-surface indices
  **9–17** (`sst_filled, f, solar, hgt, landmask, diurnal_sin, diurnal_cos, doy_sin,
  doy_cos`) + `[-1,-2]` (lat/lon), with names mapped to indices.
- `layout.active_lock_indices(perturbation)` = default set **minus**
  `perturbation.claimed_static_vars()`. Heating claims nothing (full lock). The SST
  experiment claims `sst_filled`; the terrain experiment claims `hgt`/`landmask` — so
  those vars evolve/stay-modified instead of being reset to control.
- The driver resets only the *active* lock indices after each DLAMPty step and zeros
  only those indices in δ.

**Injection-target indices** (in `model_layout.yaml`): FCNv2 channel `4`=2t; DLAMPty
surface `2`=t2m; DLAMPty upper order `(u,v,t,q,z,w)` → T=index `2`. Heating can target
surface-t2m (difference-method style) or the upper-T column (onestep style); both via
config.

## Config management & I/O

- `config.py` mirrors Part 1's yaml loader (repo-relative resolve, `lru_cache`).
- One yaml per experiment: `perturbation.type` + params, `mode`
  (`snapshot`/`continuous`/`forward`), `tc_id`, `init_time`, `fore_hour`/`iterations`,
  optional `claims` override.
- **Self-documenting outputs:** `run_experiment.py` writes the resolved config + git
  hash to `config_used.yaml` beside the `.npz`.
- **Naming (not valid-time-strict):** time features are handled inside the model and
  ideal runs often use axisymmetric/idealized vortices, so no rigid valid-time format
  is enforced. For traceability, names carry experiment tag + params + init time, e.g.
  `delta_continuous_7d_5K_init2025091700_lead<HHH>hr.npz`.
- **Storage:** reuse `regional_couple.io.arrays` (npz). Default to saving **δ
  (anomalies)** for perturbation runs; full state only when configured.
  `experiments/*/outputs/` symlinked to big storage.

## Diagnostics (the tools left out of Part 1) — physical focus

Port and make config-aware (auto-annotate titles with experiment params):
- `pv.py` ← `idealized_exp/plot_PV2*.py` (PV-θ cross-sections, tangential).
- `divergence.py` ← `plot_div*.py`; `wind_profile.py` ← `idealized_exp/wind_r/`;
  `response.py` ← Heating/RatePrecip-vs-ΔP curves + `plot_vorticity.py`.
- `waves.py` — **the physics emphasis**: resolution prevents resolving polygonal-eyewall
  wavenumbers directly, so the focus is the *outward-propagating consequences*:
  (1) **gravity-wave radiation/propagation** (e.g. radius–time Hovmöller of divergence
  or surface-pressure perturbation showing outward phase speed), and (2) **thermal
  diffusion & balance adjustment scaled by the Rossby radius of deformation** (compare
  the perturbation's horizontal spreading against the local Rossby radius to show
  geostrophic/gradient-wind adjustment).
- `modal.py` — **reserved space** for EOF / higher-moment modal analysis of the δ
  fields (no full implementation in the first slice).

## docs/perturbation_method.md — co-write, rigorous derivation

Written *with* the user; no rigid template, but the physics/math must be derived
step-by-step with no logical leaps. Must cover:
1. Definition of `S` as the composition FCNv2(6h) ∘ [DLAMPty(3h)]² ∘ couple.
2. **Terminology correction (critical):** `S` is **not** a strict *semi-linear* /
   tangent-linear operator. True semi-linearization linearizes around a state for a
   single (infinitesimal/one-step) increment. Because `S` spans a finite 6h window
   through FCNv2 + two DLAMPty steps + coupling, the trajectory is intrinsically
   **nonlinear**. Therefore `δ = S(I+δ) − S(I)` is a **finite-time nonlinear
   perturbation evolution**, not a tangent-linear response — derive and state this
   distinction explicitly (write the tangent-linear `M = ∂S/∂x|_I` and show
   `S(I+δ) − S(I) = Mδ + O(‖δ‖²)`, with the higher-order terms retained here).
3. Snapshot (frozen-time iteration) vs continuous (time-marching) — the mathematical
   difference (presence/absence of `changing_additional_information`; fixed vs advancing
   valid time) and what each isolates.
4. Why locking static variables to the control prevents dipole errors (the static
   channels are not prognostic; letting them differ between I and A injects a spurious
   static dipole into δ). Then how dynamic masking relaxes this for SST/terrain runs.

## Implementation order (first vertical slice = diabatic heating)

1. Add `pyproject.toml` to Part 1 root; `pip install -e`; smoke-test
   `import regional_couple`.
2. Scaffold this repo; `config.py`, `config/model_layout.yaml`, `layout.py` (default
   lock set + `active_lock_indices`).
3. `operators.py` (S, loaders from `regional_couple.inference.models`).
4. `driver.py` snapshot + continuous modes with hook calls + dynamic lock; fix the
   A-synthesis bug.
5. `perturbations/{base,heating,registry}.py`.
6. `experiments/diabatic_heating/` README + 3 configs (ic_bump / continuous / snapshot).
7. `scripts/run_experiment.py` + config stamping.
8. Diagnostics: `pv.py` wired to a δ output; first cut of `waves.py` (divergence
   radius–time). `modal.py` left as reserved stub.
9. Scaffold the other categories (README + config template + registered stub) incl.
   SST claiming `sst_filled` to demonstrate dynamic masking.
10. Draft `docs/perturbation_method.md` for the user to refine.

## Verification

- **Import/install:** in `pangu_env`, `python -c "import regional_couple,
  llat_manifold"` resolves with no `/wk2/yungyun` or `/wk171` literals in this repo
  (grep → none); models load from Part 1 `weights/`.
- **Lock + dynamic-mask unit tests:** after `S`, active-locked static indices equal
  the control and δ is exactly 0 there; with an SST perturbation claiming
  `sst_filled`, that index is *not* reset and δ there is non-zero.
- **Numerical parity:** run `continuous_5K_7d` a few steps and diff δ against the
  legacy `semi-linear/.../delta_continuous_7d_lead*.npz` (unchanged after refactor +
  bugfix).
- **End-to-end heating slice:** run all 3 diabatic_heating configs short; confirm
  `.npz` + `config_used.yaml` written and one PV-θ plot renders with experiment params
  in its title.
- **Regression vs onestep:** `ic_bump_Deep_10K` reproduces
  `LLAT_add_heat_onestep.py`'s t=0 Gaussian heating field exactly.
```
