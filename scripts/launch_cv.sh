#!/usr/bin/env bash
# 5-fold participant-grouped cross-validation, paper recipe, every setup.
#
#   A    sigmoid / soft                  C    sigmoid / hard
#   B    BD / soft, ICMLA KL (published) D    BD / hard, ICMLA KL
#   BS1  BD / soft, Sensoy KL weight 1   DS1  BD / hard, Sensoy KL weight 1
#   SM   sigmoid / smear (Gram, KOH)     CU   sigmoid / culture (blood agar, PDA)
#        test-supervised baselines (supplement: BCE; smear 11 epochs, culture 22 from layer 559)
#
# Runs go fold by fold (all setups of fold 0 first), shared round-robin between the GPUs in
# DEVICES, one chain per GPU, one run at a time per GPU. Survives logging out.
#
#   OUT_ROOT=/scratch90/chris_/ik_uq_cv scripts/launch_cv.sh
#   OUT_ROOT=... DEVICES="cuda:1" SETUPS="A C" FOLDS="0 1" scripts/launch_cv.sh   # a subset
#   DRY_RUN=1 OUT_ROOT=... scripts/launch_cv.sh                                     # print the plan only
set -eu
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT, e.g. /scratch90/chris_/ik_uq_cv}
DEVICES=${DEVICES:-"cuda:0 cuda:1"}
N_FOLDS=${N_FOLDS:-5}
FOLDS=${FOLDS:-$(seq -s " " 0 $((N_FOLDS - 1)))}
SEED=${SEED:-42}
SETUPS=${SETUPS:-"A C B D BS1 DS1 SM CU"}
cd "$(dirname "$0")/.."
TAG=$(date +%Y-%m-%d_%H-%M)
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$OUT_ROOT/logs"

spec() {  # NAME -> "head label [options]"
  case $1 in
    A)   echo "sigmoid soft" ;;
    C)   echo "sigmoid hard" ;;
    B)   echo "beta_dirichlet soft" ;;
    D)   echo "beta_dirichlet hard" ;;
    BS1) echo "beta_dirichlet soft --kl_schedule sensoy --kl_max 1 --kl_form sensoy" ;;
    DS1) echo "beta_dirichlet hard --kl_schedule sensoy --kl_max 1 --kl_form sensoy" ;;
    SM)  echo "sigmoid smear" ;;
    CU)  echo "sigmoid culture" ;;
    *)   echo "Unknown setup $1" >&2; exit 1 ;;
  esac
}

read -r -a devs <<< "$DEVICES"
declare -A chain
n=0
for k in $FOLDS; do
  for s in $SETUPS; do
    d=${devs[$((n % ${#devs[@]}))]}
    chain[$d]+="$(printf '%s\x1f' "${s}_f${k} $(spec "$s") --seed $SEED --n_folds $N_FOLDS --fold $k")"
    n=$((n + 1))
  done
done

for d in "${devs[@]}"; do
  IFS=$'\x1f' read -r -a cells <<< "${chain[$d]:-}"
  [ ${#cells[@]} -eq 0 ] && continue
  log="$OUT_ROOT/logs/cv_${TAG}_${d/:/}.log"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$d (${#cells[@]} runs):"; printf '    %s\n' "${cells[@]}"; continue
  fi
  nohup scripts/run_chain.sh "$d" "$OUT_ROOT" "$TAG" "${cells[@]}" > "$log" 2>&1 &
  echo "Started ${#cells[@]} runs on $d -> $log"
done
echo "Tag $TAG, $n runs in total. Follow with:"
echo "  grep -a -h -o \"=== .*\\|[A-Za-z]*Error.*\" $OUT_ROOT/logs/cv_${TAG}_*.log"
