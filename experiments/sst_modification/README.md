# Experiment: SST modification

## Physical design

Change the sea-surface temperature the model "sees" and observe the TC response. SST
controls the air–sea enthalpy flux that fuels the storm, so a uniform or patchy ΔSST
tests whether the model reproduces the expected thermodynamic sensitivity (warmer SST
→ stronger potential intensity → deeper storm).

## Dynamic static-variable masking (key mechanism)

`sst_filled` is a **static** surface channel: by default the perturbation driver
resets it to the control after every step (so a perturbation cannot secretly drift the
boundary conditions). This experiment must *keep* its warmed SST, so
`SSTPerturbation.claimed_static_vars()` returns `["sst_filled"]`. The driver then
removes that index from the active lock set
(`layout.active_lock_indices`), leaving SST free to differ from the control. This is
the reference example of dynamic masking — see `docs/perturbation_method.md` §4.

## Status

First implementation applies a uniform ΔSST over the DLAMPty sub-domain at t=0
(`apply_ic`). Extend with spatial masks (warm-pool patches, cold wakes) as needed.

## Run

```bash
python scripts/run_experiment.py --config experiments/sst_modification/configs/warm_2K.yaml
```
