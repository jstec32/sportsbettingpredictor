#!/bin/bash
# Verify data quality and integrity
# Usage: ./verify-data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

python scripts/verify_data.py --check-all --alert-on-issues
