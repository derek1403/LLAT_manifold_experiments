#!/usr/bin/env bash
# Replicate the full diabatic-heating run set on the axisymmetric idealized-vortex IC.
#
#   bash scripts/run_axisym_batch.sh [stage]
#     stage 1  heating_moist only          (42 runs)  — must finish before stage 2
#     stage 2  measure the delta-q scaling (0 runs)   — writes dq_measured_axisym.yaml
#     stage 3  the remaining 10 families   (398 runs)
#     (no argument = all three, in order)
#
# The staging is not cosmetic: the "measured" delta-q families inject the moisture the
# moist runs themselves grew, so that number has to be read off the axisymmetric moist
# runs before any dq_* family can start.
#
# Every family keeps the tag prefix and family name it has on the real IC; only the
# output category differs (outputs/diabatic_heating_axisym/), so every downstream tag
# glob and figure script works on either set with just --ic.
#
# Runs already holding a complete set of bundles are skipped, so this is re-runnable
# and resumable after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/pc/.conda/envs/pangu_env/bin/python}
SWEEP="$PY scripts/run_amp_sweep.py --ic axisym"
RANGE="--amps-range 0.5 10 0.5"          # 20 amplitudes x 2 inits = 40 runs
TS="--amps 5.0 --steps 40 --forcing-steps 8"   # the 5-day series, 2 runs
STAGE="${1:-all}"

run() {   # run <label> <args...>
  local label=$1; shift
  echo "=============================================================="
  echo "[batch $(date +%H:%M:%S)] $label"
  echo "=============================================================="
  # shellcheck disable=SC2086
  $SWEEP "$@" 2>&1 | grep -Ev "Warning|numeric\.py|^ *dd = |^ *dy = |optical_depth|ufunc\(" || true
}

stage1() {
  run "heating_moist sweep"   $RANGE
  run "heating_moist tseries" $TS --tag-prefix tseries
}

stage2() {
  echo "[batch] measuring the axisymmetric delta-q scaling"
  $PY scripts/measure_dq_scaling.py --ic axisym || exit 1
}

stage3() {
  # --- heating with one channel pinned to the control (H0/H1 controls) ---
  run "heating_qlock sweep"   $RANGE --lock-q
  run "heating_qlock tseries" $TS --lock-q --tag-prefix tseries
  run "heating_wlock sweep"   $RANGE --lock w --tag-prefix sweepw   --family heating_wlock
  run "heating_wlock tseries" $TS    --lock w --tag-prefix tseriesw --family heating_wlock
  run "heating_zlock sweep"   $RANGE --lock z --tag-prefix sweepz   --family heating_zlock
  run "heating_zlock tseries" $TS    --lock z --tag-prefix tseriesz --family heating_zlock

  # --- delta-q only, the reverse probe (H2/H3) ---
  run "dq_measured sweep"     $RANGE --pert moisture
  run "dq_measured tseries"   $TS    --pert moisture --tag-prefix tseriesdq
  run "dq_latent sweep"       $RANGE --pert moisture --dq-scaling latent
  run "dq_latent tseries"     $TS    --pert moisture --dq-scaling latent --tag-prefix tseriesdql

  # --- routing: delta-q only with the response channel pinned (H5) ---
  run "dq_tlock sweep"        $RANGE --pert moisture --lock t   --tag-prefix sweepdqtl   --family dq_tlock
  run "dq_tlock tseries"      $TS    --pert moisture --lock t   --tag-prefix tseriesdqtl --family dq_tlock
  run "dq_uvlock sweep"       $RANGE --pert moisture --lock u v --tag-prefix sweepdquv   --family dq_uvlock
  run "dq_uvlock tseries"     $TS    --pert moisture --lock u v --tag-prefix tseriesdquv --family dq_uvlock

  # --- where the model reads moisture from (M1) and whether it needs a vortex (M3) ---
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
  all) stage1 && stage2 && stage3 ;;
  *) echo "usage: $0 [1|2|3|all]" >&2; exit 2 ;;
esac

echo "=============================================================="
echo "[batch $(date +%H:%M:%S)] stage '$STAGE' finished. Inventory:"
for f in outputs/diabatic_heating_axisym/*/; do
  printf '  %-18s %3d runs\n' "$(basename "$f")" "$(find "$f" -maxdepth 1 -mindepth 1 -type d | wc -l)"
done
