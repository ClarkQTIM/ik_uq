#!/usr/bin/env bash
# KL sweep for the BD head on the redo (paper's images, split, labels), chosen on the validation file.
# For each analysis (culture: --subset culture, hard = cultures; LCA: all images, hard = posterior > 0.5) and each
# label (soft, hard), five KL settings:
#   pub    the published ICMLA schedule (default)
#   S0.01  Sensoy form, weight 0.01      S0.1  Sensoy form, weight 0.1      S1  Sensoy form, weight 1
#   E1     ICMLA eq. 5 form, weight 1
# Weight ramps as weight * min(1, epoch / 10). 2 analyses x 2 labels x 5 settings = 20 runs.
# Each run writes val_metrics.json (AUROC and how well evidence and entropy flag errors, on validation);
# src/select_kl.py picks the settings. The chosen BD runs are the main BD runs: launch_redo.sh then only needs
# SETUPS="A_cul C_cul A_lca C_lca SM CU".
#
#   OUT_ROOT=results/provisional_kl scripts/launch_kl_redo.sh
#   DRY_RUN=1 OUT_ROOT=results/provisional_kl scripts/launch_kl_redo.sh
#   OUT_ROOT=... ANALYSES="cul" KLS="pub S1" scripts/launch_kl_redo.sh       # a subset
set -eu
cd "$(dirname "$0")/.."
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT, e.g. results/provisional_kl}
LABELS=${LABELS:-results/labels/paper_labels.csv}
DEVICES=${DEVICES:-"cuda:0 cuda:1"}
SEED=${SEED:-42}
ANALYSES=${ANALYSES:-"cul lca"}
KLS=${KLS:-"pub S0.01 S0.1 S1 E1"}
[ -f "$LABELS" ] || { echo "No label table at $LABELS: run src/paper_labels.py first."; exit 1; }
LABELS=$(realpath "$LABELS")
mkdir -p "$OUT_ROOT"
OUT_ROOT=$(realpath "$OUT_ROOT")
TAG=$(date +%Y-%m-%d_%H-%M)
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$OUT_ROOT/logs"

kl_opts() {
  case $1 in
    pub)   echo "" ;;
    S0.01) echo "--kl_schedule sensoy --kl_max 0.01 --kl_form sensoy" ;;
    S0.1)  echo "--kl_schedule sensoy --kl_max 0.1 --kl_form sensoy" ;;
    S1)    echo "--kl_schedule sensoy --kl_max 1 --kl_form sensoy" ;;
    E1)    echo "--kl_schedule sensoy --kl_max 1 --kl_form icmla_eq5" ;;
    *)     echo "Unknown KL setting $1" >&2; exit 1 ;;
  esac
}

cells=()
for an in $ANALYSES; do
  case $an in
    cul) sub="--subset culture"; hard="culture_hard" ;;
    lca) sub=""; hard="paper_hard" ;;
    *)   echo "Unknown analysis $an" >&2; exit 1 ;;
  esac
  for kl in $KLS; do
    cells+=("B_${an}_kl${kl} beta_dirichlet paper_soft $sub $(kl_opts "$kl") --labels_csv $LABELS --seed $SEED")
    cells+=("D_${an}_kl${kl} beta_dirichlet $hard $sub $(kl_opts "$kl") --labels_csv $LABELS --seed $SEED")
  done
done

read -r -a devs <<< "$DEVICES"
declare -A chain
for i in "${!cells[@]}"; do
  d=${devs[$((i % ${#devs[@]}))]}
  chain[$d]+="$(printf '%s\x1f' "${cells[$i]}")"
done
for d in "${devs[@]}"; do
  IFS=$'\x1f' read -r -a cs <<< "${chain[$d]:-}"
  [ ${#cs[@]} -eq 0 ] && continue
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$d (${#cs[@]} runs):"; printf '    %s\n' "${cs[@]}"; continue
  fi
  log="$OUT_ROOT/logs/kl_${TAG}_${d/:/}.log"
  nohup scripts/run_chain.sh "$d" "$OUT_ROOT" "$TAG" "${cs[@]}" > "$log" 2>&1 &
  echo "Started ${#cs[@]} runs on $d -> $log"
done
echo "Tag $TAG, ${#cells[@]} runs. When done: python src/select_kl.py --runs $OUT_ROOT"
