# outputs/diabatic_heating — run index

Runs are grouped one level below the category by **experiment family**; every run
folder keeps the standard layout (`data/` + `plots/` + `config_used.yaml` +
auto-`README.md`). Cross-run comparison figures live in [`figures/`](figures/),
never at run roots.

| family | runs | what it is |
| --- | --- | --- |
| [`heating_moist/`](heating_moist/) | `sweep_*` ×40, `tseries_*` ×2 | heating, q free (the baseline suite) |
| [`heating_qlock/`](heating_qlock/) | `sweepq_*` ×40, `tseriesq_*` ×2 | heating with δq≡0 (moisture-denial) |
| [`dq_measured/`](dq_measured/) | `sweepdq_*` ×40, `tseriesdq_*` ×2 | δq-only, measured scaling (reverse probe) |
| [`dq_latent/`](dq_latent/) | `sweepdql_*` ×40, `tseriesdql_*` ×2 | δq-only, latent-equivalent scaling (cp/Lv per K) |
| [`dq_tlock/`](dq_tlock/) | `sweepdqtl_*` ×40, `tseriesdqtl_*` ×2 | δq-only with δT≡0 (routing test) |
| [`heating_wlock/`](heating_wlock/) | `sweepw_*` ×40, `tseriesw_*` ×2 | heating with δw≡0 (lock control) |
| [`heating_zlock/`](heating_zlock/) | `sweepz_*` ×40, `tseriesz_*` ×2 | heating with δz≡0 (lock control) |
| [`dq_bl/`](dq_bl/) | `sweepdqbl_*` ×40, `tseriesdqbl_*` ×2 | δq-only, boundary layer 850–1000 hPa (equal column) |
| [`dq_ft/`](dq_ft/) | `sweepdqft_*` ×40, `tseriesdqft_*` ×2 | δq-only, free troposphere 400–700 hPa (equal column) |
| [`dq_offcore/`](dq_offcore/) | `sweepdqoff_*` ×40 | δq-only injected +7°E off-vortex (state-dependence proxy) |
| [`dq_uvlock/`](dq_uvlock/) | `sweepdquv_*` ×40, `tseriesdquv_*` ×2 | δq-only with δu=δv≡0 (dynamic-path closure check) |
| [`snapshot/`](snapshot/) | `snapshot_*` ×3 | frozen-time power-iteration runs |

Science summary (中文):
[`experiments/diabatic_heating/findings_moisture_binding_zh.md`](../../experiments/diabatic_heating/findings_moisture_binding_zh.md)

## Conventions (follow these when adding a family)

1. **One family = one folder + one disjoint tag prefix.** Tag prefixes must not be
   a prefix of each other up to `_` (`sweep` / `sweepq` / `sweepdq` / `sweepdql`
   are disjoint), so `response.py` glob patterns never cross families even though
   its search recurses one level.
2. New families come from `scripts/run_amp_sweep.py` — the four standard combos
   pick their folder automatically; any new pert/lock combo **requires**
   `--tag-prefix` and `--family` (the script refuses to guess).
3. Single-config runs (`scripts/run_experiment.py`) set `family:` in their yaml.
4. Cross-run figures go to `figures/` with the family in the filename
   (`pv_sweep_maps_<family-ish>_<init-ish>_<variant>.png`).
5. Nothing here is ever overwritten by a new experiment — twin/variant runs get a
   new family + prefix; the moist/locked/dq **pairs are the experiment**.
