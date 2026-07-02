# `llat_manifold.diagnostics` — how each diagnostic is computed

This folder turns the model-output bundles of a run into figures. This README is the
**map**: for every diagnostic it says *what is computed* (the formula), *what it reads /
writes*, and *which `.py` (and function) to open*. When a plot looks wrong, start here,
jump to the linked source, and check the formula against the data model below.

## Data model (read this first)

Every diagnostic consumes **`.npz` bundles** loaded by
[`io.load_delta_bundle`](../io.py) → `(dlampty_upper, dlampty_sfc)`:

| array | shape | axis meaning | channels (last axis) |
| --- | --- | --- | --- |
| `dlampty_upper` | `(13, 81, 81, 6)` | level, lat(row, N→S), lon(col, W→E), var | `u,v,t,q,z,w` (z = geopotential; ÷g → height m) |
| `dlampty_sfc` | `(81, 81, 20)` | lat, lon, var | `u10,v10,t2m,…,f`(10)…; `lon`=−2, `lat`=−1 |

Pressure levels (hPa, top→bottom): `50,100,150,200,250,300,400,500,600,700,850,925,1000`.
Channel names/indices live in [`layout.py`](../layout.py) + [`config/model_layout.yaml`](../../../config/model_layout.yaml);
**always** index by name (`layout.upper_index("w")`), never a hard-coded integer.

**δ vs absolute state.** Perturbation runs store δ bundles plus a control/background:
the physical field is `ū + δ`. The suite reconstructs it
([`suite._absolute_bundle`](suite.py)); diagnostics that need physical (nonlinear)
fields take a `baseline_path` and add it. Per mode:
`snapshot`→ `data/background.npz`; `continuous`→ matching `control_*.npz`;
`forward`→ bundles are already absolute (no baseline).

Shared field extraction: [`_idealized.fields_from_bundle`](_idealized.py) returns
`(u, v, t, z[m], w, f, lat_2d, lon_2d, p_hpa)` and is the canonical place that converts
geopotential→height (÷ `G=9.80665`) — reuse it instead of re-deriving.

---

## Diagnostics index

| diagnostic | source · key function | computes | output PNG(s) |
| --- | --- | --- | --- |
| Non-hydrostatic ε | [hydrostatic.py](hydrostatic.py) · `plot_nonhydrostatic_epsilon` | `ε=|Dw/Dt|/g` | `hydrostatic_eps_profile.png`, `…_xsection.png` |
| Hydrostatic z↔T | [hydrostatic.py](hydrostatic.py) · `plot_hydrostatic_thermo` | `∂Φ/∂P` vs `−RT/P` | `hydrostatic_thermo_profile.png`, `…_xsection.png` |
| Wind balance | [wind_profile.py](wind_profile.py) · `plot_wind_balance_profile` | LLAT vs gradient vs geostrophic Vt | `wind_balance_profile.png` |
| PV–θ | [pv.py](pv.py) · `plot_pv_theta_cross_section` | Ertel PV + isentropes + Vt | `PV_Theta_tengential.png` |
| Wind circulation | [pv.py](pv.py) · `plot_wind_circulation_cross_section` | \|v\| + (u,−w) secondary flow | `wind_circulation.png` |
| Divergence–θ | [divergence.py](divergence.py) · `plot_div_theta_cross_section` | spherical ∇·(u,v) + isentropes | `div_Theta_uv.png` |
| Perturbation IKE | [ike.py](ike.py) · `plot_ike_perturbation` | `ΔIKE` total/inner/outer vs iter | `ike_perturbation.png` |
| 2D fields | [fields.py](fields.py) · `plot_fields` | Part-1 5-row field grid | `fields.png` |
| Wave Hovmöller | [waves.py](waves.py) · `hovmoller` | radius–time of δ field + L_R | `hovmoller_msl.png` |
| Response (scaffold) | [response.py](response.py) · `central_pressure_drop` | scalar Δp_min extractor | — (sweep assembles) |
| Modal (reserved) | [modal.py](modal.py) · `eof` | EOF/SVD of δ ensemble | — (NotImplemented) |

### Non-hydrostatic ε — [hydrostatic.py](hydrostatic.py)
**Strategy 1** — the genuine dynamics test.

$$\varepsilon = \frac{\quad \left| \dfrac{Dw}{Dt} \right| \quad}{g}$$

The non-dimensional non-hydrostatic parameter: the vertical acceleration measured against
gravity. From the vertical momentum equation $\dfrac{Dw}{Dt} = -\dfrac{1}{\rho}\dfrac{\partial P}{\partial z} - g$,
so $\varepsilon \sim 10^{-4}\text{–}10^{-3}$ means a hydrostatic column and $\varepsilon \to 10^{-1}$
a non-hydrostatic core. We form $Dw/Dt$ as a single-snapshot, steady-state, **spherical**
material derivative including the curvature/metric term ($a$ = Earth radius):

$$\frac{Dw}{Dt} \;\approx\; u\frac{\partial w}{\partial x} + v\frac{\partial w}{\partial y} + w\frac{\partial w}{\partial z} \;-\; \frac{u^2+v^2}{a}$$

$\partial w/\partial t$ is neglected (snapshot mode freezes valid time). The vertical gradient
is pointwise, `np.gradient(w,axis=0)/np.gradient(z,axis=0)`, to survive the non-uniform level
spacing. See `_dwdt_fields`. Outputs a pressure-on-$y$ profile (domain + eyewall + max,
advective-only vs +metric) and an azimuthal-mean radius–height map; `which=` picks figure(s).

### Hydrostatic z↔T — [hydrostatic.py](hydrostatic.py)
**Strategy 2** — a coordinate/thermo *consistency* check (not dynamics).

$$\frac{\partial \Phi}{\partial P} = -\alpha = -\frac{RT}{P}, \qquad \Phi = g\,z$$

The isobaric hydrostatic equation. We compare LHS `np.gradient(Φ,p_pa,axis=0)` against RHS
$-RT/P$ via the fractional residual

$$\text{resid} = \frac{\quad \dfrac{\partial \Phi}{\partial P} + \dfrac{RT}{P} \quad}{\left| \dfrac{RT}{P} \right|}$$

A tiny residual mainly confirms the model kept ERA5's hydrostatically-built isobaric heights
consistent with $T$ (hence "consistency check", not a dynamics test). Two-panel profile
(LHS/RHS + residual) and an azimuthal-mean residual $r$–$z$ map.

### Wind balance — [wind_profile.py](wind_profile.py)
The model's own azimuthal-mean tangential wind,

$$v_t = -\,u\sin\theta + v\cos\theta,$$

compared against the two *balanced* winds it should approach.

**Geostrophic wind** — Coriolis balances the pressure-gradient force alone (no curvature):

$$v_g = \frac{1}{f}\,\frac{\partial \Phi}{\partial r} \qquad\text{(isobaric)}, \qquad\qquad v_g = \frac{1}{\rho\,f}\,\frac{\partial P}{\partial r} \qquad\text{(surface, } \rho = \tfrac{P}{RT}).$$

**Gradient wind** — adds the centripetal (curvature) term, the proper TC balance:

$$\frac{v_{gr}^{\,2}}{r} + f\,v_{gr} = \frac{\partial \Phi}{\partial r} \qquad\Longrightarrow\qquad v_{gr} = -\frac{f r}{2} + \sqrt{\;\frac{f^{2} r^{2}}{4} + r\,\frac{\partial \Phi}{\partial r}\;}$$

(the physical positive root; the surface form replaces $r\,\partial\Phi/\partial r$ with
$\tfrac{r}{\rho}\,\partial P/\partial r$). Code: `_winds_isobaric` / `_winds_surface`, from the
azimuthal-mean geopotential $\Phi$ (or pressure $P$). Plotting $v_t$ at sfc/850/500 hPa against
$v_g$ and $v_{gr}$ shows how close the model vortex sits to gradient-wind balance — and where
(typically the eyewall) it departs from the geostrophic approximation.

### PV–θ & wind circulation — [pv.py](pv.py)
Ertel potential vorticity on isobaric levels over the 0.25° spherical grid:

$$\mathrm{PV} = -g\left[\left(\frac{\partial v}{\partial p}\frac{\partial \theta}{\partial x} - \frac{\partial u}{\partial p}\frac{\partial \theta}{\partial y}\right) + (\zeta + f)\frac{\partial \theta}{\partial p}\right], \qquad \theta = T\left(\frac{p_0}{p}\right)^{R/c_p}$$

with relative vorticity $\zeta = \partial v/\partial x - \partial u/\partial y$
([`_idealized.calculate_pv_spherical`](_idealized.py), in PVU). Drawn on a radius–height W–E
cross-section with isentropes ($\theta$) and tangential-wind contours; the companion plot
shades $|v|$ with the in-plane $(u, -w)$ secondary-circulation quiver. **PV is nonlinear** in
$(u,v,t)$: for δ bundles pass `baseline_path=…, add_to_baseline=True` for physical Δ-PV.

### Divergence–θ — [divergence.py](divergence.py)

$$\delta = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

Spherical horizontal divergence ([`_idealized.calculate_divergence`](_idealized.py),
$10^{-5}\,\text{s}^{-1}$) on the W–E cross-section + isentropes — the in/out-flow signature
of the secondary circulation.

### Perturbation IKE — [ike.py](ike.py)

$$\Delta \mathrm{IKE}_k = \mathrm{IKE}(\bar u + \delta_k) - \mathrm{IKE}(\bar u), \qquad \mathrm{IKE}(r) = \tfrac{1}{2}\,\rho\,\bar v_t^{\,2}\,\big(2\pi r\,H\,\Delta r\big)\ \ (\rho=H=1)$$

$$\mathrm{IKE}_{\text{inner}} = \!\!\sum_{r<200\,\text{km}}\!\! \mathrm{IKE}(r), \qquad \mathrm{IKE}_{\text{outer}} = \!\!\sum_{r\ge 200\,\text{km}}\!\! \mathrm{IKE}(r), \qquad \mathrm{IKE}_{\text{total}} = \sum_r \mathrm{IKE}(r)$$

Per-iteration change in storm integrated kinetic energy, total / inner / outer, with
$x=$ iteration and $y=\Delta\mathrm{IKE}$ [TJ] — **one summary curve per run** (not an
evolution movie). $\mathrm{IKE}$ is the *verbatim* Part-1 core
[`regional_couple.diagnostics.ike`](../../../../claude_RegionalCouple_AI/src/regional_couple/diagnostics/ike.py):
10 m wind → storm-centred polar → azimuthal-mean tangential profile $\bar v_t$ → the integral
above over 5 km annuli (→ TJ), the same definition as the operational dashboard's IKE panels.
`_ike_profile` uses the data's **own** 1D lat/lon axes (see Gotchas #1).

### 2D fields — [fields.py](fields.py)
No single formula — a 5-row 2D field grid (10 m wind, precip, vorticity, $\omega$, TCWV)
rendered by Part-1's field-comparison plotter via `bundle_to_xarray`. The vendored grid owns
its own save (no returned Figure).

### Wave Hovmöller — [waves.py](waves.py)

$$L_R = \frac{N H}{f}$$

Radius–time (Hovmöller) of an azimuthal-mean δ surface field (default `msl`): an
outward-tilting phase line gives the radiating gravity wave's apparent radial phase speed,
and the Rossby radius of deformation $L_R$ (`rossby_radius_km`) is overlaid to separate the
gravity-wave-dominated ($r<L_R$) from the balanced ($r>L_R$) response. Needs ≥3 leads →
continuous/forward only.

### Response / Modal — [response.py](response.py) · [modal.py](modal.py)

$$\Delta p_{\min} = \min\big(\delta\,\mathrm{MSL}\big)$$

`response.central_pressure_drop` extracts this scalar intensification proxy per run; the
amplitude **sweep** assembles response-vs-forcing curves. `modal.eof` is **reserved**
(raises `NotImplementedError`) — EOF/SVD of the δ power-iteration sequence toward its leading
finite-time mode.

---

## Orchestration

- **[suite.py](suite.py) · `standard_plots(run_dir)`** — the per-run entry point
  (`scripts/run_experiment.py` calls it). Picks the representative bundle (last
  iter/lead), resolves the baseline, builds the absolute bundle, and `_try`-runs each
  diagnostic into `plots/` (failures are caught + logged, never abort the run). **Add a
  new diagnostic here** with one `_try(...)` line.
- **[evolution.py](evolution.py) · `evolution_plots(run_dir)`** — walks **every** bundle
  to make per-iteration frames + a GIF per diagnostic (in `plots/evolution/<diag>/`) plus
  `growth_curves.png`. Diagnostics are registered in the `_DIAGS` dict; absolute-state
  plotters get `which="xsection"` to return the map figure for stamping.
- **[\_\_init\_\_.py](__init__.py)** — `load_stamp` (reads `config_used.yaml`) and
  `experiment_title` (the 2-line figure title). 
- **[_idealized.py](_idealized.py)** — shared scientific core (PV, divergence,
  `fields_from_bundle`, `radius_height`, θ, heating-profile panel). Constants
  `G,RD,CP,A_EARTH` live here; reuse them so Strategy-1/2 share one `g`.

## Run a single diagnostic by hand

Each module is a CLI (`python -m llat_manifold.diagnostics.<mod> …`). Per-**bundle**
plotters take a bundle path; per-**run** plotters (`ike`, `evolution`) take the run dir:

```bash
RUN=outputs/diabatic_heating/snapshot_10K_iter20_init2025092000
B=$RUN/data
# per-bundle (δ → add background for the absolute/nonlinear field):
python -m llat_manifold.diagnostics.hydrostatic $B/delta_…_iter020.npz --baseline $B/background.npz \
       --out-dir $RUN/plots --strategy both
python -m llat_manifold.diagnostics.pv          $B/delta_…_iter020.npz --baseline $B/background.npz --add-to-baseline
# per-run:
python -m llat_manifold.diagnostics.ike         $RUN --out $RUN/plots/ike_perturbation.png
python -m llat_manifold.diagnostics.evolution   $RUN --diag hydro_eps,hydro_thermo
```

> Run with the **pangu_env** interpreter directly —
> `/home/pc/.conda/envs/pangu_env/bin/python`. In this setup `conda activate pangu_env`
> does **not** switch `python` (it stays on `sun`), so `python -m …` would use the wrong env.

## Gotchas / lessons (things that have actually broken)

1. **IKE grid axes.** `_ike_profile` must use the data's own 1D axes
   (`lat2d[:,0]`, `lon2d[0,:]`), **not** a reconstructed `np.arange`-based grid — float
   overshoot can yield an 82-point axis for some storm centres and crash the polar
   transform (`IndexError: boolean index did not match…`). Fixed in [ike.py](ike.py).
2. **README image links.** The run README is regenerated by
   [`io.generate_run_readme`](../io.py), which only relativizes figure paths
   (`plots/…`) when handed an **absolute** `run_dir`. Always pass
   `Path(run_dir).resolve()`; a relative path produces full-path image links.
3. **`w` units.** Treated as vertical velocity in **m/s** (the ε band 10⁻⁶–10⁻⁴
   confirms it). [hydrostatic.py](hydrostatic.py) prints `w`'s min/max/mean at runtime so a
   unit surprise (Pa/s ω) is visible.
4. **Nonlinear fields need the baseline.** PV and IKE are nonlinear; running them on a raw
   δ bundle (no baseline) gives physically meaningless numbers and degenerate lat/lon.
