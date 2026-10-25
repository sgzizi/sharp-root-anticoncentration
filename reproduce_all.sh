#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ARTIFACT_DIR"
export PYTHONPATH="$ARTIFACT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest
python experiments/dimension_free.py
python experiments/rbf_graph_ssl.py
python experiments/common_offset_demo.py
# Maintenance annotation; functional content is unchanged.
