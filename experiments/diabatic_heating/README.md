# Experiment: Diabatic heating

## Spatial setup at a glance

The heating is an axisymmetric Gaussian centred on the storm. The horizontal footprint
(and the Deep/Shallow/Stratiform vertical weighting) looks like:

![Gaussian heating footprint](assets/gaussian_contours.png)

## Physical design

Inject a localized, axisymmetric **diabatic heating** anomaly near the TC core and
watch how the LLAT (DLAMPty / coupled FCNv2) model responds. Heating is the leading
energy source of a tropical cyclone (latent heat release in convection), so this is
the most direct probe of whether the learned model obeys the dynamics that *should*
follow from heating.

Three configs (all init from `202518W`, `2025091700`):

| config | mode | injection | what it isolates |
| --- | --- | --- | --- |
| `ic_bump_Deep_10K` | `forward` | one-shot upper-T bump (Deep profile, +10 K) | immediate balance adjustment to an imposed warm core |
| `continuous_5K_7d` | `continuous` | per-step surface-T forcing, 7 days | sustained-forcing response & track of the growing anomaly |
| `snapshot_5K_iter20` | `snapshot` | per-iteration forcing, frozen time | the fastest-growing finite-time mode at one valid time |

Vertical heating profiles (`heat_type`): **Deep** (single half-sine, mid-tropospheric
peak — deep convection), **Stratiform** (full sine — heating aloft / cooling below,
mature-stage anvil), **Shallow** (low-level peak — shallow convection).

## Why we expect a particular response (physics to check)

- **Hydrostatic balance:** a warm anomaly lowers the column thickness gradient aloft
  and raises geopotential above the heating; the surface should respond with a
  pressure fall under the heating. Check that $\Delta\theta$ and $\Delta z$ are hydrostatically
  consistent. `hydrostatic.py` quantifies the **non-hydrostatic degree** two ways:
  *Strategy 1* forms the vertical-momentum residual $\varepsilon = |Dw/Dt|/g$ (steady-state,
  spherical, with the $-(u^2+v^2)/A$ curvature term; $\partial w/\partial t$ neglected in `snapshot`
  mode) — $\varepsilon \approx 10^{-4}$–$10^{-3}$ flags a hydrostatic column, $\varepsilon \to 10^{-1}$ a non-hydrostatic core;
  *Strategy 2* checks the isobaric hydrostatic equation $\partial\Phi/\partial p = -R_d T/p$ as a $z \leftrightarrow T$
  consistency test. Both draw a pressure-on-y profile (domain + eyewall) **and** an
  azimuthal-mean radius–height map; the evolution suite also animates the maps.
- **Gradient-wind / geostrophic adjustment:** the mass perturbation cannot stay
  unbalanced. Within the Rossby radius of deformation it radiates gravity waves and
  settles toward gradient-wind balance; outside it, a balanced vortex response. The
  `waves.py` radius–time diagnostic is built to expose exactly this.
- **PV generation:** diabatic heating generates potential vorticity below the heating
  maximum and destroys it above ($\partial(\mathrm{PV})/\partial t \propto \partial\dot\theta/\partial z$ along the absolute-vorticity
  vector). A physically faithful model should build a low-level PV tower under
  sustained heating. Compare `pv.py` cross-sections of control vs perturbed.
- **Integrated kinetic energy response:** `ike.py` reuses the Part-1 IKE core to ask how
  much the storm's IKE changes under the perturbation — $\Delta\mathrm{IKE} = \mathrm{IKE}(\bar u + \delta) - \mathrm{IKE}(\bar u)$ split
  into total / inner ($r < 200$ km) / outer ($r \ge 200$ km), plotted vs iteration (one summary
  curve per run, same IKE definition as the operational dashboard).
  The **cross-run** version (`src/fig_I1_ike_ladder.py`, one figure over the whole intensity
  ladder × moist / q-locked / δq-only) is the one that carries the result:
  > 結果與圖文整理(中文研究筆記):[`findings_IKE_response.md`](findings_IKE_response.md) ——
  > 動能響應比 PV 響應**晚兩天**、是**強度開關式**的、且鎖 q 之後只剩 **3%**(ΔPV 還剩 38–59%);
  > 純 δq 不加熱產生的動能是「加熱但鎖 q」的**十倍**。

## How it is implemented

`perturbation.type: heating` → `llat_manifold.perturbations.heating.HeatingPerturbation`.
`injection: ic` adds the bump to the upper-T column at t=0 (`apply_ic`); `per_step`
adds a TC-following Gaussian to the surface-T perturbation each step (`apply_step`).
Heating claims **no** static variables, so the full static-variable lock applies.

## Run

```bash
conda activate pangu_env
python scripts/run_experiment.py --config experiments/diabatic_heating/configs/continuous_5K_7d.yaml
```

The run folder is named with the init time, e.g.
`outputs/diabatic_heating/heating_moist/continuous_5K_7d_init2025091700/`, and contains:

```
data/           δ/state bundles (*.npz)
plots/          PV_Theta_tengential.png   (PV–θ + heating profile + tangential wind)
                wind_circulation.png      (|v| shading + u–w secondary-circulation quiver)
                div_Theta_uv.png          (divergence–θ + heating profile)
                wind_balance_profile.png  (LLAT vs gradient vs geostrophic wind)
                fields.png                (Part 1 5-row 2D field grid)
                hydrostatic_eps_profile.png / _xsection.png      (Strategy 1: ε=|Dw/Dt|/g non-hydrostatic check)
                hydrostatic_thermo_profile.png / _xsection.png   (Strategy 2: ∂Φ/∂P vs −RT/P z↔T consistency)
                ike_perturbation.png      (ΔIKE total/inner/outer vs iteration; one curve per run)
                hovmoller_msl.png         (radius–time gravity-wave Hovmöller; continuous runs)
                evolution/                per-iteration frames + GIFs (incl. hydrostatic_eps/_thermo) + growth_curves.png
config_used.yaml
README.md       auto-generated: data source, run info, inline figures
```

The standard plot suite runs automatically after the model run (disable with
`--no-plots`). To re-render a single figure by hand:

```bash
RUN=outputs/diabatic_heating/heating_moist/continuous_5K_7d_init2025091700
python -m llat_manifold.diagnostics.pv           $RUN/data/<bundle>.npz --out $RUN/plots/PV.png
python -m llat_manifold.diagnostics.pv           $RUN/data/<bundle>.npz --wind --out $RUN/plots/wind_circ.png
python -m llat_manifold.diagnostics.divergence   $RUN/data/<bundle>.npz --out $RUN/plots/div.png
python -m llat_manifold.diagnostics.wind_profile $RUN/data/<bundle>.npz --out $RUN/plots/wind_bal.png
python -m llat_manifold.diagnostics.fields       $RUN/data/<bundle>.npz --out $RUN/plots/fields.png
python -m llat_manifold.diagnostics.waves        $RUN/data --var msl     --out $RUN/plots/hov.png
# Non-hydrostatic checks (writes *_profile.png + *_xsection.png for each strategy):
python -m llat_manifold.diagnostics.hydrostatic  $RUN/data/<bundle>.npz --baseline $RUN/data/background.npz \
       --out-dir $RUN/plots --strategy both
# Per-iteration evolution frames + GIFs (e.g. just the two hydrostatic maps):
python -m llat_manifold.diagnostics.evolution    $RUN --diag hydro_eps,hydro_thermo
# Perturbation IKE response (ΔIKE total/inner/outer vs iteration; one figure per run):
python -m llat_manifold.diagnostics.ike          $RUN --out $RUN/plots/ike_perturbation.png
```

For semi-linear ($\delta$) bundles, pass the matching control to get physical $\Delta\mathrm{PV}$:
`--baseline <control_bundle>.npz --add-to-baseline`. The hydrostatic checks instead take
`--baseline <background>.npz` to reconstruct the absolute state $\bar u + \delta$ (snapshot mode).

## Outputs layout: one folder per experiment family

`outputs/diabatic_heating/` groups runs one level below the category by **family**
(full index + conventions: [`outputs/diabatic_heating/README.md`](../../outputs/diabatic_heating/README.md)):

```
outputs/diabatic_heating/
  figures/          cross-run comparison figures (never at run roots)
  heating_moist/    sweep_* + tseries_*      heating, q free (baseline)
  heating_qlock/    sweepq_* + tseriesq_*    heating with δq≡0
  dq_measured/      sweepdq_* + tseriesdq_*  δq-only, measured scaling
  dq_latent/        sweepdql_*               δq-only, latent-equivalent
  snapshot/         snapshot_*               frozen-time power iterations
```

**Adding a new family** (e.g. the planned T-locked / lock-w / layered-δq suites) is
one command: `run_amp_sweep.py` takes `--pert {heating,moisture}`, `--lock VAR...`
(any of u/v/t/q/z/w) and `--dq-scaling {measured,latent}`; the four standard combos
pick their folder automatically, and any *new* combination requires an explicit
`--tag-prefix` **and** `--family` (prefixes must stay disjoint — that is what keeps
`response.py` patterns from crossing families, since its search recurses one level).
Single-config runs set `family:` in their yaml. Example:

```bash
# δq-only with the temperature response denied -> outputs/diabatic_heating/dq_tlock/
python scripts/run_amp_sweep.py --amps 2 5 --pert moisture --lock t \
       --tag-prefix sweepdqtl --family dq_tlock
```

## Amplitude sweep + ΔPV-vs-theory (forcing–response)

Two cross-run figures probe the $\Delta\mathrm{PV}$ response quantitatively (`diagnostics/response.py`):

* **Amplitude sweep** — continuous-mode runs at `amp_K` $= 0.5, 1.0, \dots, 10.0$ K (0.5 K
  spacing), both init times, 24 h with the full `amp_K` spread over the first 8 steps
  (`amp_mode: spread`, `forcing_steps: 8`). A $\Delta\mathrm{PV}$ dipole scalar
  ($\max \Delta\mathrm{PV}$ over 700–1000 hPa / $\min \Delta\mathrm{PV}$ over 200–500 hPa,
  heated core $r \le 2\sigma$) at a fixed lead is plotted vs `amp_K` against a
  linear reference through the origin (slope pinned at 0.5 K) — departure from that line
  is the $O(\|\delta\|^2)$ manifold-curvature signal of `docs/perturbation_method.md` §5. Each
  point carries a small `(pressure level, radius)` tag locating its extremum.
* **ΔPV series vs theory** — 5-day runs at fixed `amp_K` $= 5$ K (24 h injection), model
  $\Delta\mathrm{PV}$ extrema per iteration vs the accumulated diabatic-source expectation.
  The Ertel-PV source at leading order (Haynes–McIntyre) is a **rate** equation in the
  material heating rate $\dot\theta = \mathrm{D}\theta/\mathrm{D}t$:

  $$
  \frac{\mathrm{D}\,\mathrm{PV}}{\mathrm{D}t}
  \;=\;
  \mathrm{PV}\,\frac{\partial\dot\theta}{\partial\theta}
  \quad\Longrightarrow\quad
  \Delta\mathrm{PV}_{\mathrm{th}}(n)
  \;=\;
  \sum_{i \,\le\, \min(n,\,8)}
  \mathrm{PV}_{\mathrm{ctrl}}(i)\,
  \frac{\partial\,\Delta\theta_{\mathrm{step}} / \partial p}{\partial\theta_{\mathrm{ctrl}}(i) / \partial p},
  $$

  accumulated per forced step on the **control** state ($\dot\theta\,\Delta t =
  \Delta\theta_{\mathrm{step}}$ per 3 h step, so $\Delta t$ cancels;
  $\Delta\theta_{\mathrm{step}} = \Delta T_{\mathrm{step}}(p_0/p)^{R_d/c_p}$). The accumulated
  3-D theory field gets the same dipole reduction as the model $\Delta\mathrm{PV}$ at every
  lead. After the forcing stops the source vanishes and the theory line is **flat** —
  materially, PV is conserved — so the model's later drift is itself the conservation check.
  Model above theory = the manifold generates/holds more $\Delta\mathrm{PV}$ than the injected
  heating supports; below = it has dissipated or exported it. Caveats: control-state
  (semi-linear leading order) estimate; the frozen Eulerian field ignores advection of the
  generated PV, so during forcing the model reduction sits naturally below it. x-axis
  is iteration $n$ in nominal hours (model steps are nominal 3 h, not strict physical time).

```bash
# batch runner (loads DLAMPty once; complete runs are skipped, so refining the sweep
# later — e.g. 0.1 K spacing over a curved interval — is an incremental call):
python scripts/run_amp_sweep.py --amps-range 5 10 0.5 --steps 8                        # 22 sweep runs
python scripts/run_amp_sweep.py --amps 5.0 --steps 40 --forcing-steps 8 --tag-prefix tseries  # 2 time-series runs
# figures:
python -m llat_manifold.diagnostics.response sweep outputs/diabatic_heating --lead 24 \
       --out outputs/diabatic_heating/figures/pv_amplitude_sweep_lead024h.png
python -m llat_manifold.diagnostics.response timeseries outputs/diabatic_heating/heating_moist/tseries_5K_120h_init2025092000 \
       --out outputs/diabatic_heating/heating_moist/tseries_5K_120h_init2025092000/plots/pv_timeseries_vs_theory.png
# 6×3 structure panel (columns = 0.5/2/4/6/8/10 K; rows = upper-min-layer map,
# low-max-layer map, azimuthal-mean r–z). --per-K plots ΔPV/amp_K: if the response
# were semi-linear all six columns would look identical:
python -m llat_manifold.diagnostics.response maps outputs/diabatic_heating --init 2025092000 \
       --out outputs/diabatic_heating/figures/pv_sweep_maps_strong_abs.png
python -m llat_manifold.diagnostics.response maps outputs/diabatic_heating --init 2025092000 --per-K \
       --out outputs/diabatic_heating/figures/pv_sweep_maps_strong_perK.png
```

## Moisture-locked runs (`sweepq_*` / `tseriesq_*`): is heating→PV bound to moisture?

> 結果與圖文整理(中文研究筆記):[`findings_moisture_binding_zh.md`](findings_moisture_binding_zh.md)

**The question.** In the physical atmosphere, the PV generated by an imposed $\dot\theta$ is a
*dry* process — absolute vorticity times a stability change; humidity need not vary at
all. But the LLAT manifold learned its dynamics from data in which warm anomalies and
moisture anomalies almost always co-occur (convective heating *is* latent heating). So
inside the model, "thermodynamics" may not exist as an independent axis: the manifold
may route part — or most — of its heating response *through the moisture channel*,
because that is the only neighbourhood of state space it has ever seen. These runs
measure that binding.

**The intervention.** Identical heating experiments, but with `lock_upper_vars: [q]`:
the perturbed trajectory's specific humidity is pinned back to the control's at every
step, i.e. **$\delta q \equiv 0$ — the moisture field is held at its unperturbed values and is not
allowed to keep iterating inside the semi-linear evolution**. The heating still goes
into $T$ each step; only the moisture co-evolution is denied. (Driver support:
`driver._upper_lock_indices`; needs a control trajectory, so continuous/snapshot only.)

**Reading the comparison.** For every moist run (`sweep_*`, `tseries_*`) there is a
one-to-one q-locked twin (`sweepq_*`, `tseriesq_*`) in its **own run folder — nothing
is overwritten**, since the pair *is* the experiment:

* q-locked $\Delta\mathrm{PV} \approx$ moist $\Delta\mathrm{PV}$ → the PV response is genuinely
  dry-dynamical; the manifold separates thermodynamics from moisture.
* q-locked $\Delta\mathrm{PV}$ collapses (or deforms) → the response was moisture-mediated: the
  manifold has fused heating and moistening into one entangled direction, and cutting
  the q wire severs part of the "dry" PV pathway too. The gap between the curves is
  the moisture-mediated share, and how it varies with amplitude/vortex strength maps
  where the binding is tightest.

```bash
# runs (new folders; --lock-q auto-switches the tag prefix so moist runs are kept):
python scripts/run_amp_sweep.py --amps-range 0.5 10 0.5 --steps 8 --lock-q             # sweepq_*
python scripts/run_amp_sweep.py --amps 5.0 --steps 40 --forcing-steps 8 --tag-prefix tseries --lock-q  # tseriesq_*
# q-locked versions of the standard figures:
python -m llat_manifold.diagnostics.response sweep outputs/diabatic_heating --pattern "sweepq_*" --out ...
python -m llat_manifold.diagnostics.response maps  outputs/diabatic_heating --init 2025092000 --prefix sweepq --out ...
# the binding probes (moist solid vs q-locked dashed; the gap = moisture-mediated share):
python -m llat_manifold.diagnostics.response compare-sweep outputs/diabatic_heating \
       --out outputs/diabatic_heating/figures/pv_sweep_qlock_comparison.png
python -m llat_manifold.diagnostics.response compare-tseries \
       outputs/diabatic_heating/heating_moist/tseries_5K_120h_init2025092000 \
       outputs/diabatic_heating/heating_qlock/tseriesq_5K_120h_init2025092000 \
       --out outputs/diabatic_heating/figures/pv_tseries_qlock_comparison_strong.png
```

### The reverse probe (`sweepdq_*` / `tseriesdq_*` / `sweepdql_*`): δq-only forcing

The q-lock runs cut the moisture wire; the reverse probe drives it alone. Inject a
specific-humidity anomaly with the same spatial design (vertical profile × centred
Gaussian) but **no temperature bump**: $\dot\theta = 0$, so a dry-dynamical model
should produce $\Delta\mathrm{PV} \approx 0$. Any PV that appears anyway is
q-channel-routed response — the two-sided evidence for the binding.

Two amplitude scalings, both parameterised by a nominal `amp_K` (shared x-axis with
the heating sweep):

* **measured** (primary, `sweepdq_*`): at nominal `amp_K`, inject the δq the *moist
  amp_K heating run itself* grew by nominal hour 24 — per-init 13-level profile and
  `gkg_per_K` measured from `sweep_5K_*` and stored in
  [`configs/dq_measured.yaml`](configs/dq_measured.yaml) (≈ 0.24 / 0.19 g/kg per K for
  weak/strong; peak at 700 hPa; scale is ~linear in amp, ±10 %, so linear scaling from
  the 5 K measurement is used).
* **latent-equivalent** (full sweep, `sweepdql_*`/`tseriesdql_*`): $\Delta q = (c_p/L_v)\,
  \Delta T \approx 0.402$ g/kg per K — the moisture whose complete condensation would
  release `amp_K` of heating — with the Deep vertical profile.

```bash
# runs:
python scripts/run_amp_sweep.py --amps-range 0.5 10 0.5 --steps 8 --pert moisture   # sweepdq_*
python scripts/run_amp_sweep.py --amps 5.0 --steps 40 --forcing-steps 8 --pert moisture \
       --tag-prefix tseriesdq                                                       # tseriesdq_*
python scripts/run_amp_sweep.py --amps 2 5 8 --steps 8 --pert moisture --dq-scaling latent  # sweepdql_*
# figures — heating vs δq-only on the same nominal-K axis:
python -m llat_manifold.diagnostics.response compare-sweep outputs/diabatic_heating \
       --pattern-b "sweepdq_*" --label-a "heating (ΔT)" --label-b "δq-only (measured δq)" \
       --title "Reverse probe — heating vs δq-only forcing" \
       --out outputs/diabatic_heating/figures/pv_sweep_dq_comparison.png
# equivalence-ratio view (ΔPV(B)/ΔPV(A) vs amplitude, log axis, ±25% band; reusable
# for any two sweeps — defaults compare latent-equivalent δq against heating):
python -m llat_manifold.diagnostics.response ratio-sweep outputs/diabatic_heating \
       --out outputs/diabatic_heating/figures/pv_sweep_dql_equivalence_ratio.png
# energy budget: sensible/latent/total core column energy vs iteration (one panel per run)
# and the moisture Hovmöller (radius–time of Lv·∫δq dm — where re-moistening comes from):
python -m llat_manifold.diagnostics.response energy <run_dir>... --ncols 3 \
       --out outputs/diabatic_heating/figures/pv_energy_partition_tseries.png
python -m llat_manifold.diagnostics.response qhov <run_dir>... \
       --out outputs/diabatic_heating/figures/pv_moisture_hovmoller_strong.png
python -m llat_manifold.diagnostics.response compare-tseries \
       outputs/diabatic_heating/heating_moist/tseries_5K_120h_init2025092000 \
       outputs/diabatic_heating/dq_measured/tseriesdq_5K_120h_init2025092000 \
       --label-a "heating (ΔT)" --label-b "δq-only" \
       --title-prefix "Reverse probe — heating vs δq-only:" \
       --caption "δq-only has θ̇=0: any ΔPV is q-channel-routed response" \
       --out outputs/diabatic_heating/figures/pv_tseries_dq_comparison_strong.png
# δq-only runs also work with sweep/timeseries/maps (no theory line — θ̇=0):
python -m llat_manifold.diagnostics.response sweep outputs/diabatic_heating \
       --pattern "sweepdq_*" --out outputs/diabatic_heating/figures/pv_amplitude_sweep_dq_lead024h.png
python -m llat_manifold.diagnostics.response maps outputs/diabatic_heating \
       --init 2025092000 --prefix sweepdq --out outputs/diabatic_heating/figures/pv_sweep_maps_dq_strong_abs.png
```

## Committed runs (outputs index)

The run folders live at the repo root under `outputs/diabatic_heating/`. (The
`experiments/diabatic_heating/outputs` path is a symlink and does not expand on GitHub —
use the links below.) Each run keeps its `README.md`, `config_used.yaml` and `plots/*.png`;
the heavy `data/*.npz` bundles are git-ignored and regenerated on demand.

### `snapshot_5K_iter20` — init `2025091700`

- 📄 [Run README](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025091700/README.md)
  · ⚙️ [config_used.yaml](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025091700/config_used.yaml)
- 🖼 Figures:
  [PV–θ + tangential wind](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025091700/plots/PV_Theta_tengential.png)
  · [wind circulation](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025091700/plots/wind_circulation.png)
  · [divergence–θ](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025091700/plots/div_Theta_uv.png)
  · [wind balance profile](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025091700/plots/wind_balance_profile.png)
  · [fields](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025091700/plots/fields.png)

### `snapshot_5K_iter20` — init `2025092000`

- 📄 [Run README](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025092000/README.md)
  · ⚙️ [config_used.yaml](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025092000/config_used.yaml)
- 🖼 Figures:
  [PV–θ + tangential wind](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025092000/plots/PV_Theta_tengential.png)
  · [wind circulation](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025092000/plots/wind_circulation.png)
  · [divergence–θ](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025092000/plots/div_Theta_uv.png)
  · [wind balance profile](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025092000/plots/wind_balance_profile.png)
  · [fields](../../outputs/diabatic_heating/snapshot/snapshot_5K_iter20_init2025092000/plots/fields.png)
