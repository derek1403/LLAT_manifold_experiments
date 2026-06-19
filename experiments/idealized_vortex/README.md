# Experiment: Idealized vortex (scaffold)

## Physical design

Replace the realistic initial condition with a clean, **axisymmetric** vortex (and
optionally a quiescent environment) so the storm's evolution is not contaminated by
real synoptic asymmetries. This is the cleanest setting to study intrinsic structure
changes — eyewall replacement, polygonal asymmetries and their outward radiation,
balanced spin-up — on the model's own manifold.

## Status

Scaffold. Port `inference_two_way_FCNv2_idealized_vortex_all.py` and the
`idealized_exp/initial_data/idealize_vortex/{sfc,upper}_idealize_all.npy` arrays into
`VortexPerturbation.apply_ic` (overwrite the upper/surface channels from the idealized
arrays). Best paired with `forward` mode (clean trajectory) or `snapshot` mode (mode
structure).
