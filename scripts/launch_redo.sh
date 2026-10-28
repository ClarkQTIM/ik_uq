#!/usr/bin/env bash
# Redo on the paper's own images and split (paper_labels.csv from src/paper_labels.py), paper recipe.
# Two analyses (28 Oct 2026), two independent outputs (bacterial, fungal), Travis's posteriors as soft labels:
#   LCA analysis, all images; hard = posterior > 0.5 (Travis's rule); scored against those hard labels
#     A_lca  sigmoid / soft      C_lca  sigmoid / hard
#     B_lca  BD / soft           D_lca  BD / hard                   (BD: ICMLA KL, as published)
#   Culture analysis, only images with a culture (MADURAI, SCUT); hard = the cultures; scored against cultures
#     A_cul  sigmoid / soft      C_cul  sigmoid / culture
#     B_cul  BD / soft           D_cul  BD / culture
#   The paper's test-supervised baselines (supplement): SM sigmoid / smear, CU sigmoid / culture (BCE)
# Every run's test_metrics.json scores each output against both references where it can.
#
# Trains on the paper's training split (validation loss logged when the table has the validation file); the
# test split is divided 20/80 into calibration/evaluation for conformal. Runs are shared round-robin between
# the GPUs in DEVICES, one at a time per GPU. Survives logging out.
#
#   OUT_ROOT=results/paper_split scripts/launch_redo.sh
#   DRY_RUN=1 OUT_ROOT=results/paper_split scripts/launch_redo.sh                 # print the plan
#   OUT_ROOT=... DEVICES="cuda:1" SETUPS="A_cul C_cul" scripts/launch_redo.sh      # a subset
set -eu
cd "$(dirname "$0")/.."
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT, e.g. results/paper_split}
LABELS=${LABELS:-results/labels/paper_labels.csv}
DEVICES=${DEVICES:-"cuda:0 cuda:1"}
SEED=${SEED:-42}
SETUPS=${SETUPS:-"A_cul C_cul B_cul D_cul A_lca C_lca B_lca D_lca SM CU"}
[ -f "$LABELS" ] || { echo "No label table at $LABELS: run src/paper_labels.py first."; exit 1; }
LABELS=$(realpath "$LABELS")
mkdir -p "$OUT_ROOT"
OUT_ROOT=$(realpath "$OUT_ROOT")
TAG=$(date +%Y-%m-%d_%H-%M)
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$OUT_ROOT/logs"

spec() {  # NAME -> "head label [options]"
  case $1 in
    A_lca) echo "sigmoid paper_soft" ;;                         C_lca) echo "sigmoid paper_hard" ;;
    B_lca) echo "beta_dirichlet paper_soft" ;;                  D_lca) echo "beta_dirichlet paper_hard" ;;
    A_cul) echo "sigmoid paper_soft --subset culture" ;;        C_cul) echo "sigmoid culture_hard --subset culture" ;;
    B_cul) echo "beta_dirichlet paper_soft --subset culture" ;; D_cul) echo "beta_dirichlet culture_hard --subset culture" ;;
    SM)    echo "sigmoid smear" ;;                              CU)    echo "sigmoid culture" ;;
    *)     echo "Unknown setup $1" >&2; exit 1 ;;
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
