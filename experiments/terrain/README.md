# Experiment: Terrain modification (scaffold)

## Physical design

Remove or alter orography (`hgt`) and the land/sea mask (`landmask`) to isolate the
terrain's effect on track and structure (e.g. Taiwan/Luzon interaction, frictional
spin-down, topographic channelling). Comparing with-terrain vs flattened runs probes
whether the model encodes genuine terrain dynamics or a learned climatological bias.

## Dynamic static-variable masking

`hgt` and `landmask` are static channels; this experiment claims both
(`TerrainPerturbation.claimed_static_vars() -> ["hgt", "landmask"]`) so the modified
orography persists through the run.

## Status

Scaffold. `TerrainPerturbation.apply_ic` raises `NotImplementedError`; implement the
orography edit (e.g. zero `hgt` in a box, set `landmask` to ocean) when building this
category out.
