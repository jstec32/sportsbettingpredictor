#!/bin/bash
# Scan markets for betting opportunities
# Usage: ./check-opportunities.sh [min-edge] [min-confidence]
# Example: ./check-opportunities.sh 0.03 0.7

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

python scripts/scan_markets.py \
  --min-edge ${1:-0.03} \
  --min-confidence ${2:-0.7} \
  --output opportunities.json
