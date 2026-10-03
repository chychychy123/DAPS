#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
research_python="${RESEARCH_PYTHON:-python}"
"$research_python" tools/launch.py eval --run "${1:?Provide a COCO training run directory}" --out "${2:?Provide a new evaluation directory}" --recipe "${3:-I0}"
