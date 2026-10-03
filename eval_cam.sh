#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
research_python="${RESEARCH_PYTHON:-python}"
args=(cam --run "${1:?Provide a training run directory}" --out "${2:?Provide a new CAM directory}")
if [[ $# -ge 3 ]]; then args+=(--ids "$3"); fi
"$research_python" tools/launch.py "${args[@]}"
