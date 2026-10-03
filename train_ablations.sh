#!/usr/bin/env bash
# All variants use the full training split and the same schedule per dataset.
set -euo pipefail
cd "$(dirname "$0")"
dataset="${1:?Usage: bash train_ablations.sh voc|coco NEW_RUN_ROOT}"
run_root="${2:?Provide a new run root}"
case "$dataset" in voc|coco) ;; *) echo 'Dataset must be voc or coco' >&2; exit 2 ;; esac
for variant in D0_no_dino B0_dinov2_control B1_partial_labels A1_anchored_graph AB1_combined; do
    bash "train_${dataset}.sh" "${run_root}/${variant}" "$variant"
done
