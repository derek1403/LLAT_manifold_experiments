#!/usr/bin/env bash
# Local-Rossby-radius intensity sweep: identical heating into vortices that differ
# only in intensity.
#
#   bash scripts/run_lr_sweep.sh [stage]
#     stage 1  the 5 K moist + q-locked pairs over all vmax   (12 runs)
#     stage 2  the 2 K moist branch (amplitude linearity)     ( 6 runs)
#     (no argument = both)
#
# Why: the weak-0917 / strong-0920 pair differs in SST, shear, humidity, translation
# and (for 0920) land + terrain, so it cannot isolate the effect of the vortex's own
# inertial stability. Here the environment is one fixed quiescent background and the
# only thing that changes between members is vmax, hence L_R = N·H/√(ξη):
#
#     vmax  0 → 50 m/s   gives   L_R_core  3491 → 391 km
#
# which brackets the real pair (weak 1346 km, strong 364 km). The heating is byte-for-byte
# the family the rest of the study uses: Deep profile, σ = 5, amp spread over 8 steps.
#
# Two branches per vortex:
#   heating_moist  — q free, the model's full moist response;
#   heating_qlock  — δq ≡ 0, which removes the moisture feedback and leaves the
#                    (nearly) dry balanced adjustment the L_R argument is a theory of.
# If the dry branch follows L_R and the moist branch does not, that difference is
# itself the result.
#
# 16 steps = 48 h: 24 h of forcing plus 24 h of free spreading, so a confinement
# radius can be read both while the anomaly is being built and after it is left alone.
#
# Runs already holding a complete set of bundles are skipped, so this is re-runnable.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/pc/.conda/envs/pangu_env/bin/python}
CAT=diabatic_heating_lrsweep
IC_DIR=outputs/vortex_intensity_ic
VMAX=${VMAX:-"0 10 20 30 40 50"}
RMW=${RMW:-1}
STEPS="--steps 16 --forcing-steps 8"
STAGE="${1:-all}"

run() {   # run <vmax> <amp> <label> <extra args...>
  local v=$1 amp=$2 label=$3; shift 3
  local ic="$IC_DIR/vortex_v${v}_rmw${RMW}.npz"
  if [[ ! -f $ic ]]; then
    echo "[lr] MISSING $ic — build it first:"
    echo "    $PY scripts/make_vortex_intensity_ic.py"
    exit 1
  fi
  echo "=============================================================="
  echo "[lr $(date +%H:%M:%S)] vmax=${v} m/s  amp=${amp} K  ${label}"
  echo "=============================================================="
  # shellcheck disable=SC2086
  $PY scripts/run_amp_sweep.py \
      --ic-npz "$ic" --category "$CAT" \
      --family "v${v}_rmw${RMW}_${label}" \
      --amps "$amp" $STEPS --inits 2025091700 "$@" \
      2>&1 | grep -Ev "Warning|numeric\.py|^ *dd = |^ *dy = |optical_depth|ufunc\(" || true
}

stage1() {
  for v in $VMAX; do
    run "$v" 5.0 moist
    run "$v" 5.0 qlock --lock-q
  done
}

stage2() {
  for v in $VMAX; do
    run "$v" 2.0 moist
  done
}

case "$STAGE" in
  1) stage1 ;;
  2) stage2 ;;
  all) stage1 && stage2 ;;
  *) echo "usage: $0 [1|2|all]" >&2; exit 2 ;;
esac

echo "=============================================================="
echo "[lr $(date +%H:%M:%S)] stage '$STAGE' finished. Inventory:"
for f in outputs/$CAT/*/; do
  printf '  %-24s %2d run(s)\n' "$(basename "$f")" \
    "$(find "$f" -maxdepth 1 -mindepth 1 -type d | wc -l)"
done
