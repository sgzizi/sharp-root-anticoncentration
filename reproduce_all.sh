#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ARTIFACT_DIR"

python -m pytest
python experiments/dimension_free.py
python experiments/rbf_graph_ssl.py
python experiments/common_offset_demo.py
