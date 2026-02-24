.PHONY: install dev-install test lint format typecheck clean setup-db verify all sync-schedule capture-odds settle-scores

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

# Install production dependencies
install:
	$(PIP) install -r requirements.txt

# Install with development dependencies
dev-install:
	python3.12 -m venv .venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

# Run all tests
test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run unit tests only
test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

# Run integration tests only
test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

# Run linting
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

# Format code
format:
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check --fix .

# Run type checking
typecheck:
	$(PYTHON) -m mypy src/

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Setup database
setup-db:
	$(PYTHON) scripts/setup_db.py

# Backfill historical data
backfill:
	$(PYTHON) scripts/backfill_data.py

# Settle completed games: update scheduled→final with scores (run daily at 2 AM)
# Prerequisite for real-odds backtesting — without this the training set doesn't grow
settle-scores:
	$(PYTHON) scripts/settle_scores.py --league nba --days 2

# Sync upcoming game schedule from ESPN (run daily)
# Populates 'scheduled' game rows so capture-odds can match games by team name
sync-schedule:
	$(PYTHON) scripts/sync_schedule.py --league nba --days 7

# Capture current odds from The Odds API and store with game_id
# Always syncs schedule first so team-name matching works
# Cron (every 30 min, 6pm-midnight ET on game days):
#   */30 18-23 * * * cd /path/to/SportsBettingPredictor && make capture-odds >> logs/odds.log 2>&1
capture-odds: sync-schedule
	$(PYTHON) scripts/capture_odds.py --sport nba

# Train models
train:
	$(PYTHON) scripts/train.py

# Run backtest
backtest:
	$(PYTHON) scripts/backtest.py

# Scan for opportunities
scan:
	$(PYTHON) scripts/scan_markets.py

# Verify data quality
verify-data:
	$(PYTHON) scripts/verify_data.py

# Generate report
report:
	$(PYTHON) scripts/generate_report.py

# Run all verification checks
verify: lint typecheck test
	@echo "All checks passed!"

# Full setup
all: dev-install setup-db verify
	@echo "Setup complete!"
