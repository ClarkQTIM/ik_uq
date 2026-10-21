#!/usr/bin/env bash
# v2, paper recipe, all four cells.
#   cuda:0 runs A (sigmoid, soft) then C (sigmoid, hard)
#   cuda:1 runs B (BD, soft)      then D (BD, hard)
# so A and B finish first. Survives logging out (nohup).
#
#   OUT_ROOT=/scratch90/chris_/ik_uq_runs scripts/launch_paper_2x2.sh
# If batch 32 runs out of GPU memory:
#   EXTRA="--batch 16 --accum 2" OUT_ROOT=... scripts/launch_paper_2x2.sh
set -eu
OUT_ROOT=${OUT_ROOT:?Set OUT_ROOT to a folder outside runs/, e.g. /scratch90/chris_/ik_uq_runs}
cd "$(dirname "$0")/.."
TAG=$(date +%Y-%m-%d_%H-%M)
mkdir -p "$OUT_ROOT/logs"
nohup scripts/run_chain.sh cuda:0 "$OUT_ROOT" "$TAG" "A sigmoid soft" "C sigmoid hard" \
    > "$OUT_ROOT/logs/gpu0_$TAG.log" 2>&1 &
nohup scripts/run_chain.sh cuda:1 "$OUT_ROOT" "$TAG" "B beta_dirichlet soft" "D beta_dirichlet hard" \
    > "$OUT_ROOT/logs/gpu1_$TAG.log" 2>&1 &
echo "Started (tag $TAG). Follow with:"
echo "  tail -f $OUT_ROOT/logs/gpu0_$TAG.log"
echo "  tail -f $OUT_ROOT/logs/gpu1_$TAG.log"
