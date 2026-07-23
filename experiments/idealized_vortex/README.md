# Experiment: Idealized vortex — ring barotropic instability

## Physical design

Does the LLAT manifold contain **barotropic instability of an annular vorticity
ring** (eyewall-ring breakdown into mesovortices)? Real eyewall rings are 2–3
grid points at 0.25° — unresolvable — so we exploit the scale-invariance of
barotropic dynamics and **enlarge the ring** (core diameter 8–32 grid points,
keep-V scaling: V fixed ≈ 35 m/s, ζ_ring ∝ 1/r, growth slows accordingly).

The quantitative target is the Hendricks et al. (2009, JAS) γ–δ phase space:
the fastest-growing azimuthal wavenumber m depends only on the **hollowness
γ = ζ_eye/ζ_ring** and the **thickness δ = r1/r2** — thick rings → m = 2–4,
thin rings → higher m. We measure m and its growth rate σ from the run and
compare against that phase diagram. References: Schubert et al. (1999),
Kossin & Schubert (2001), Hendricks et al. (2009); methodology of idealized
perturbations in a DL model: Hakim & Masanam (2024, AIES).

## Setup (all code isolated in `src/llat_manifold/idealized_vortex/` — the
shared driver/perturbations/registry are untouched)

1. **Quiescent background** (`scripts/make_quiescent_background.py`): the real
   analysis IC flattened — winds→0, thermo→domain-mean profiles, uniform ocean
   (hgt=landmask=0, mean SST), f/solar/time-encodings/lat-lon kept (β kept).
2. **Balanced ring IC** (`ring.py` + `balance.py`): three-region smoothed ζ(r)
   (Schubert-style), V(r) by integration with an outer taper (circulation
   confined inside the 20° domain), typhoon-like vertical decay F(p),
   gradient-wind Φ′ per level + hydrostatic warm-core T′ + msl′/sp′; small
   white-noise u/v seed breaks axisymmetry. z channel is geopotential (m²s⁻²).
3. **Snapshot semi-linear loop** (`driver_vortex.py`): the shared frozen-time
   re-centering recursion (u′ᵢ = M(ū+u′ᵢ₋₁) − ū; ring lives inside u′, exactly
   the ITCZ-breakdown scheme), plus `background_npz` loading and an optional
   8-px `boundary_relax` frame pinning δ to the background each iteration.
4. **Azimuthal diagnostics** (`azimuthal.py`): ζ′(850) → polar (r,θ) → FFT →
   A_m(r, iter); growth curves + e-folding σ per day + dominant m + ζ′ maps.

## Run

```bash
python scripts/make_quiescent_background.py --tc-id 202518W --init 2025091700
python scripts/run_vortex_experiment.py --config experiments/idealized_vortex/configs/base.yaml
python scripts/plot_ic_vs_ragasa.py --background outputs/idealized_vortex/backgrounds/quiescent_202518W_2025091700.npz
python -m llat_manifold.idealized_vortex.azimuthal outputs/idealized_vortex/ring_snapshot/<run>
python tests/test_vortex_offline.py     # offline (no models)
```

Configs: `base.yaml` (γ=0.2, δ=0.7, rmw=2.5°, vmax=35, N=80 iters ≈ 10 days);
`delta_*.yaml` (δ-sweep, wavenumber selection); `rmw_*deg.yaml` (scale sweep
1–4°, the resolution/scale-matching question). IC validation figures land in
`outputs/idealized_vortex/ic_check/` (ζ/V_t/θ r–p sections vs RAGASA + the
gradient-wind balance check).

## Legacy scaffold note

The original plan (port `inference_two_way_FCNv2_idealized_vortex_all.py` +
`{sfc,upper}_idealize_all.npy` into `VortexPerturbation.apply_ic`) is
superseded by the analytic balanced ring above; `perturbations/vortex.py`
remains an untouched scaffold.
