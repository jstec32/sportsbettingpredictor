# Data Pipeline Agent

```yaml
name: data-pipeline
description: Ingest and validate sports/odds data from external APIs

responsibilities:
  - Fetch data from Kalshi API for market prices and orderbooks
  - Fetch game schedules, scores, and injuries from ESPN API
  - Fetch odds from The Odds API (respect 500/month limit)
  - Fetch weather forecasts from Open-Meteo for outdoor games
  - Validate data completeness and quality before storage
  - Handle missing data, API failures, and rate limits gracefully
  - Store validated data in PostgreSQL with proper schemas
  - Cache aggressively to respect rate limits
  - Log all API requests for debugging and rate limit tracking

tools:
  - python scripts/backfill_data.py
  - python scripts/verify_data.py
  - Database queries via src/data/storage.py

verification:
  - Run data quality checks after ingestion
  - Flag anomalies (missing games, stale odds, incomplete data)
  - Alert on API failures or rate limit warnings
  - Verify team/game relationships are intact

api_limits:
  kalshi: 100 requests/minute
  espn: No official limit - be respectful
  odds_api: 500 requests/month (cache everything!)
  open_meteo: Unlimited (free)

common_tasks:
  - "Backfill NFL games for 2024 season"
  - "Refresh odds for upcoming games"
  - "Check for data quality issues"
  - "Get current Kalshi markets for sports"
```

## Usage

This agent handles all data ingestion tasks. Use it when:
- Setting up the database with historical data
- Refreshing market data before making predictions
- Debugging data quality issues
- Monitoring API usage

## Example Commands

```bash
# Backfill historical data
python scripts/backfill_data.py --sport nfl --seasons 2023,2024

# Verify data quality
python scripts/verify_data.py --check-all --alert-on-issues

# Check API usage
python -c "from src.data.storage import DatabaseStorage; s = DatabaseStorage(); print(s.get_api_usage('OddsAPIClient'))"
```
