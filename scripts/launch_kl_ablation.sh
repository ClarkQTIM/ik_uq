#!/usr/bin/env bash
# KL ablation for the Beta-Dirichlet cells (B = soft labels, D = hard labels), paper recipe.
# The ICMLA-schedule arm already exists (the B and D runs of 2026-10-21). New arms, in order:
#   S1    Sensoy form, kl_max 1    (Sensoy et al.'s own setting)
#   S0.1  Sensoy form, kl_max 0.1
#   E1    ICMLA eq. 5 form, kl_max 1
#   S0.01 Sensoy form, kl_max 0.01
# Weight ramps as kl_max * min(1, epoch / 10). Each arm runs B then D, so arms finish in pairs.
# All on one GPU, one after another (8 runs). Survives logging out.
#
#   OUT_ROOT=/scratch90/chris_/ik_uq_runs DEVICE=cuda:1 scripts/launch_kl_ablation.sh
set -eu
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT, e.g. /scratch90/chris_/ik_uq_runs}
DEVICE=${DEVICE:-cuda:1}
SEED=${SEED:-42}
cd "$(dirname "$0")/.."
TAG=$(date +%Y-%m-%d_%H-%M)
mkdir -p "$OUT_ROOT/logs"
S="--seed $SEED --kl_schedule sensoy"
nohup scripts/run_chain.sh "$DEVICE" "$OUT_ROOT" "$TAG" \
    "B_klS1 beta_dirichlet soft $S --kl_max 1 --kl_form sensoy" \
    "D_klS1 beta_dirichlet hard $S --kl_max 1 --kl_form sensoy" \
    "B_klS0.1 beta_dirichlet soft $S --kl_max 0.1 --kl_form sensoy" \
    "D_klS0.1 beta_dirichlet hard $S --kl_max 0.1 --kl_form sensoy" \
    "B_klE1 beta_dirichlet soft $S --kl_max 1 --kl_form icmla_eq5" \
    "D_klE1 beta_dirichlet hard $S --kl_max 1 --kl_form icmla_eq5" \
    "B_klS0.01 beta_dirichlet soft $S --kl_max 0.01 --kl_form sensoy" \
    "D_klS0.01 beta_dirichlet hard $S --kl_max 0.01 --kl_form sensoy" \
    > "$OUT_ROOT/logs/kl_ablation_$TAG.log" 2>&1 &
echo "Started KL ablation (tag $TAG) on $DEVICE. Follow with:"
echo "  grep -a -o \"=== .*\\|epoch [0-9]*/15 .*\\|[A-Za-z]*Error.*\" $OUT_ROOT/logs/kl_ablation_$TAG.log"
