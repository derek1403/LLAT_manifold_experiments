#!/usr/bin/env bash
# Intensity-ladder heating runs: the 0921 / 0922 vortices dropped into 0920's fixed
# environment (SST / f / radiation / lat-lon / clock), so only the vortex differs from
# the existing 0920 (STRONG) set. The IC of each member is
# axisym_<tc>_<init>_env<ENV>.npz, selected with run_amp_sweep.py --ic axisym
# --env-from <ENV>; build those first with:
#
#   python scripts/make_axisymmetric_ic.py --tc-id 202518W \
#       --inits 2025092100 2025092200 --env-from 2025092000
#
#   bash scripts/run_ladder_batch.sh [stage]
#     stage 1  heating_moist (sweep + 5-day tseries)   — must finish before stage 2
#     stage 2  measure the delta-q scaling             — writes dq_measured_axisym.yaml
#     stage 3  heating_qlock + dq_measured sweeps       (the H0 / H2 core probes)
#     (no argument = all three, in order)
#
# "Core families only" (heating_moist, heating_qlock, dq_measured): the test-the-waters
# subset that answers whether the moisture-binding story survives at genuine typhoon
# intensity, before committing the other 8 families. Runs already holding a complete
# bundle set are skipped, so this is re-runnable and resumable.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/pc/.conda/envs/pangu_env/bin/python}
ENV_INIT=${ENV_INIT:-2025092000}                 # the fixed reference environment
INITS=${INITS:-"2025092100 2025092200"}          # the vortices (their own atmospheres)
# dq scaling is measured per member AND must keep the existing 0917/0920 entries, so
# the measure step re-reads all four (idempotent for the two that already exist).
DQ_INITS=${DQ_INITS:-"2025091700 2025092000 2025092100 2025092200"}

SWEEP="$PY scripts/run_amp_sweep.py --ic axisym --env-from $ENV_INIT --inits $INITS"
RANGE="--amps-range 0.5 10 0.5"                   # 20 amplitudes x 2 inits = 40 runs
TS="--amps 5.0 --steps 40 --forcing-steps 8"     # the 5-day series, 2 runs
STAGE="${1:-all}"

run() {   # run <label> <args...>
  local label=$1; shift
  echo "=============================================================="
  echo "[ladder $(date +%H:%M:%S)] $label"
  echo "=============================================================="
  # shellcheck disable=SC2086
  $SWEEP "$@" 2>&1 | grep -Ev "Warning|numeric\.py|^ *dd = |^ *dy = |optical_depth|ufunc\(|getfattr" || true
}

stage1() {
  run "heating_moist sweep"   $RANGE
  run "heating_moist tseries" $TS --tag-prefix tseries
}

stage2() {
  echo "[ladder] measuring the axisymmetric delta-q scaling (all four inits)"
  # shellcheck disable=SC2086
  $PY scripts/measure_dq_scaling.py --ic axisym --inits $DQ_INITS || exit 1
}

stage3() {
  run "heating_qlock sweep"   $RANGE --lock-q
  run "dq_measured sweep"     $RANGE --pert moisture
}

# The other 8 families (everything except the core heating_moist / heating_qlock /
# dq_measured), so the ladder members carry the full family set the 0917/0920 axisym
# runs already have and every figure renders for them too. Same tags/families as
# run_axisym_batch.sh stage3; only --env-from differs. dq_* families read the δq table
# stage 2 wrote (now with 0921/0922 entries); dq_latent uses the cp/Lv profile instead.
stage4() {
  # --- heating with one channel pinned to control (H1) ---
  run "heating_wlock sweep"   $RANGE --lock w --tag-prefix sweepw   --family heating_wlock
  run "heating_wlock tseries" $TS    --lock w --tag-prefix tseriesw --family heating_wlock
  run "heating_zlock sweep"   $RANGE --lock z --tag-prefix sweepz   --family heating_zlock
  run "heating_zlock tseries" $TS    --lock z --tag-prefix tseriesz --family heating_zlock

  # --- δq only: energy-equivalence (H3) ---
  run "dq_latent sweep"       $RANGE --pert moisture --dq-scaling latent
  run "dq_latent tseries"     $TS    --pert moisture --dq-scaling latent --tag-prefix tseriesdql

  # --- δq only with the response channel pinned (H5 routing) ---
  run "dq_tlock sweep"        $RANGE --pert moisture --lock t   --tag-prefix sweepdqtl   --family dq_tlock
  run "dq_tlock tseries"      $TS    --pert moisture --lock t   --tag-prefix tseriesdqtl --family dq_tlock
  run "dq_uvlock sweep"       $RANGE --pert moisture --lock u v --tag-prefix sweepdquv   --family dq_uvlock
  run "dq_uvlock tseries"     $TS    --pert moisture --lock u v --tag-prefix tseriesdquv --family dq_uvlock

  # --- which layer's moisture the model reads (M1) and whether it needs a vortex (M3) ---
  run "dq_bl sweep"           $RANGE --pert moisture --dq-layer bl
  run "dq_bl tseries"         $TS    --pert moisture --dq-layer bl --tag-prefix tseriesdqbl --family dq_bl
  run "dq_ft sweep"           $RANGE --pert moisture --dq-layer ft
  run "dq_ft tseries"         $TS    --pert moisture --dq-layer ft --tag-prefix tseriesdqft --family dq_ft
  run "dq_offcore sweep"      $RANGE --pert moisture --dq-offset 0 28   # 7 deg east ~ 745 km
}

case "$STAGE" in
  1) stage1 ;;
  2) stage2 ;;
  3) stage3 ;;
  4) stage4 ;;
  all) stage1 && stage2 && stage3 && stage4 ;;
  *) echo "usage: $0 [1|2|3|4|all]" >&2; exit 2 ;;
esac

echo "=============================================================="
echo "[ladder $(date +%H:%M:%S)] stage '$STAGE' finished. Ladder inventory:"
for fam in heating_moist heating_qlock dq_measured heating_wlock heating_zlock \
           dq_latent dq_tlock dq_uvlock dq_bl dq_ft dq_offcore; do
  d="outputs/diabatic_heating_axisym/$fam"
  line="  $(printf '%-16s' "$fam")"
  for init in $INITS; do
    n=$(find "$d" -maxdepth 1 -mindepth 1 -type d -name "*init${init}" 2>/dev/null | wc -l)
    line+=" init${init}:$(printf '%2d' "$n")"
  done
  echo "$line"
done
