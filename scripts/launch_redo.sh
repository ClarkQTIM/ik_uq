#!/usr/bin/env bash
# Redo on the paper's own images and split (paper_labels.csv from src/paper_labels.py), paper recipe.
#
#   Replication (the paper's labels, two per-pathogen outputs, scored per pathogen):
#     A_paper  sigmoid / paper_soft      C_paper  sigmoid / paper_hard
#     B_paper  BD / paper_soft           D_paper  BD / paper_hard          (BD: ICMLA KL, as published)
#   Corrected (one two-class label from all four tests, scored as binary):
#     A_2c     sigmoid / twoclass_soft   C_2c     sigmoid / twoclass_hard
#     B_2c     BD / twoclass_soft        D_2c     BD / twoclass_hard
#   Test-supervised baselines (supplement): SM sigmoid / smear, CU sigmoid / culture
#
# Trains on the paper's full training split (no validation set unless the table has one); the test
# split is divided 20/80 into calibration/evaluation for conformal. Runs are shared round-robin
# between the GPUs in DEVICES, one at a time per GPU. Survives logging out.
#
#   OUT_ROOT=results/paper_split scripts/launch_redo.sh
#   DRY_RUN=1 OUT_ROOT=results/paper_split scripts/launch_redo.sh                 # print the plan
#   OUT_ROOT=... DEVICES="cuda:1" SETUPS="A_paper C_paper" scripts/launch_redo.sh  # a subset
set -eu
cd "$(dirname "$0")/.."
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT, e.g. results/paper_split}
LABELS=${LABELS:-results/labels/paper_labels.csv}
DEVICES=${DEVICES:-"cuda:0 cuda:1"}
SEED=${SEED:-42}
SETUPS=${SETUPS:-"A_paper C_paper B_paper D_paper A_2c C_2c B_2c D_2c SM CU"}
[ -f "$LABELS" ] || { echo "No label table at $LABELS: run src/paper_labels.py first."; exit 1; }
LABELS=$(realpath "$LABELS")
mkdir -p "$OUT_ROOT"
OUT_ROOT=$(realpath "$OUT_ROOT")
TAG=$(date +%Y-%m-%d_%H-%M)
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$OUT_ROOT/logs"

spec() {  # NAME -> "head label"
  case $1 in
    A_paper) echo "sigmoid paper_soft" ;;      C_paper) echo "sigmoid paper_hard" ;;
    B_paper) echo "beta_dirichlet paper_soft" ;; D_paper) echo "beta_dirichlet paper_hard" ;;
    A_2c)    echo "sigmoid twoclass_soft" ;;   C_2c)    echo "sigmoid twoclass_hard" ;;
    B_2c)    echo "beta_dirichlet twoclass_soft" ;; D_2c) echo "beta_dirichlet twoclass_hard" ;;
    SM)      echo "sigmoid smear" ;;           CU)      echo "sigmoid culture" ;;
    *)       echo "Unknown setup $1" >&2; exit 1 ;;
  esac
}

read -r -a devs <<< "$DEVICES"
declare -A chain
n=0
for s in $SETUPS; do
  d=${devs[$((n % ${#devs[@]}))]}
  chain[$d]+="$(printf '%s\x1f' "$s $(spec "$s") --labels_csv $LABELS --seed $SEED")"
  n=$((n + 1))
done
for d in "${devs[@]}"; do
  IFS=$'\x1f' read -r -a cells <<< "${chain[$d]:-}"
  [ ${#cells[@]} -eq 0 ] && continue
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$d (${#cells[@]} runs):"; printf '    %s\n' "${cells[@]}"; continue
  fi
  log="$OUT_ROOT/logs/redo_${TAG}_${d/:/}.log"
  nohup scripts/run_chain.sh "$d" "$OUT_ROOT" "$TAG" "${cells[@]}" > "$log" 2>&1 &
  echo "Started ${#cells[@]} runs on $d -> $log"
done
echo "Tag $TAG, $n runs. Follow with:"
echo "  grep -a -h -o \"=== .*\\|epoch [0-9]*/[0-9]* .*\\|[A-Za-z]*Error.*\" $OUT_ROOT/logs/redo_${TAG}_*.log | tail"
