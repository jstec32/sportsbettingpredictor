#!/bin/bash
# Train and validate prediction models
# Usage: ./train-models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Training models..."
python scripts/train.py --update-features --validate --save

echo "Training complete. Models saved to models/ directory."
