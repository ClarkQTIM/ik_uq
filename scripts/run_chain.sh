#!/usr/bin/env bash
# Run two cells one after the other on one GPU.
# Usage: scripts/run_chain.sh DEVICE OUT_ROOT TAG "CELL1" "CELL2"
#   where each CELL is "<name> <head> <label_type>", e.g. "A sigmoid soft"
# Extra model.py options can be passed in the EXTRA environment variable.
set -u
DEVICE=$1; OUT_ROOT=$2; TAG=$3; shift 3
cd "$(dirname "$0")/.."
for cell in "$@"; do
  read -r NAME HEAD LABEL <<< "$cell"
  echo "=== $(date '+%F %T')  Cell $NAME: $HEAD / $LABEL on $DEVICE"
  python src/model.py --recipe paper --head "$HEAD" --label_type "$LABEL" \
      --device "$DEVICE" --out_root "$OUT_ROOT" --run_name "${NAME}_${HEAD}_${LABEL}_paper_${TAG}" \
      ${CSV:+--csv "$CSV"} ${IMG_ROOT:+--img_root "$IMG_ROOT"} ${EXTRA:-}
  rc=$?
  echo "=== $(date '+%F %T')  Cell $NAME finished with exit code $rc"
done
