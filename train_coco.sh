#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
research_python="${RESEARCH_PYTHON:-python}"
"$research_python" tools/launch.py train --config "research/configs/coco/${2:-B1_partial_labels}.json" --run-dir "${1:?Usage: bash train_coco.sh NEW_RUN_DIRECTORY [CONFIG_NAME]}"
