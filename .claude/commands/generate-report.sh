#!/bin/bash
# Generate performance and risk reports
# Usage: ./generate-report.sh [period]
# Example: ./generate-report.sh week
# Periods: day, week, month, season, all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Create reports directory if it doesn't exist
mkdir -p reports

python scripts/generate_report.py \
  --period ${1:-week} \
  --metrics roi,sharpe,clv,drawdown \
  --output reports/report_$(date +%Y%m%d).json
