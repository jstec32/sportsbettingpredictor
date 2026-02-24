#!/bin/bash
# Backtest models against historical data
# Usage: ./backtest.sh [sport] [season]
# Example: ./backtest.sh nfl 2024

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

python scripts/backtest.py \
  --sport ${1:-nfl} \
  --season ${2:-2024} \
  --models all \
  --output reports/backtest_$(date +%Y%m%d).json
