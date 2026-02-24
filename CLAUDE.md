# CLAUDE.md - Sports Betting Predictor

## Project Overview

This is a Kalshi sports betting prediction system for NFL and NBA markets. It uses Python 3.11, PostgreSQL, and free-tier APIs to collect data, engineer features, train models, detect edges, and manage paper trading positions.

## Quick Start Commands

```bash
# Setup
make dev-install          # Install all dependencies
make setup-db             # Initialize database

# Daily Workflow
make verify-data          # Check data quality
make scan                 # Find betting opportunities
make report               # Generate performance report

# Development
make test                 # Run tests
make lint                 # Check code style
make typecheck            # Run mypy
```

## Architecture

### Data Flow
1. **Ingestion** (`src/data/`): APIs → PostgreSQL
2. **Features** (`src/features/`): Raw data → Predictive features
3. **Models** (`src/models/`): Features → Win probabilities
4. **Edge Detection** (`src/market/`): Model probs vs market odds → Edges
5. **Position Sizing** (`src/execution/`): Edges → Bet sizes (Kelly)
6. **Risk Management** (`src/risk/`): Portfolio limits → Alerts

### Key Tables
- `teams`, `games`: Core sports data
- `kalshi_markets`, `odds_history`: Market data
- `predictions`, `bets`: Model outputs and trades
- `portfolio_snapshots`: Performance tracking

### API Clients
- **KalshiClient**: Market data and trading
- **ESPNClient**: Scores, schedules, injuries
- **OddsAPIClient**: Sportsbook odds (500/month free)
- **WeatherClient**: Open-Meteo (unlimited, free)

## Important Constraints

### Rate Limits
- The Odds API: 500 requests/month - CACHE AGGRESSIVELY
- Kalshi: 100 requests/minute
- ESPN: Unofficial API - be respectful, add delays

### Risk Limits (config/risk_limits.yaml)
- Max single bet: 5% of bankroll
- Max total exposure: 20% of bankroll
- Min edge threshold: 3%
- Kelly fraction: 0.25 (quarter Kelly)
- Halt trading at 20% drawdown

### Code Standards
- Type hints on all functions
- Async/await for API calls
- Pydantic models for data validation
- loguru for logging
- pytest for testing

## Common Tasks

### Adding a New Data Source
1. Create client in `src/data/new_client.py`
2. Inherit from `BaseAPIClient`
3. Add to `src/data/__init__.py`
4. Add schema to `scripts/schema.sql`
5. Write tests in `tests/unit/test_new_client.py`

### Adding a New Model
1. Create model in `src/models/new_model.py`
2. Inherit from `BaseModel`
3. Implement `train()` and `predict_proba()`
4. Add to ensemble config in `config/models.yaml`
5. Write tests

### Adding a New Feature
1. Add to appropriate module in `src/features/`
2. Ensure no look-ahead bias
3. Handle missing data gracefully
4. Add to feature documentation

## File Locations

| What | Where |
|------|-------|
| API clients | `src/data/` |
| Feature engineering | `src/features/` |
| Prediction models | `src/models/` |
| Edge calculation | `src/market/edge_calculator.py` |
| Position sizing | `src/execution/position_sizer.py` |
| Risk limits | `config/risk_limits.yaml` |
| Database schema | `scripts/schema.sql` |
| Subagent specs | `.claude/agents/` |
| Slash commands | `.claude/commands/` |

## Environment Variables

Required in `.env`:
- `DATABASE_URL`: PostgreSQL connection string
- `ODDS_API_KEY`: The Odds API key

Optional:
- `KALSHI_EMAIL`, `KALSHI_PASSWORD`: For live trading
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `ENVIRONMENT`: development, production

## Debugging Tips

1. **API issues**: Check `api_requests` table for errors
2. **Missing data**: Run `make verify-data`
3. **Model performance**: Check `reports/` for backtests
4. **Database**: Use `psql` with DATABASE_URL

## Subagents Available

Use `/agent <name>` to invoke:
- `data-pipeline`: Data ingestion and validation
- `feature-engineer`: Build predictive features
- `model-trainer`: Train and evaluate models
- `edge-detector`: Find profitable opportunities
- `position-sizer`: Calculate optimal bet sizes
- `paper-trader`: Execute paper trades
- `risk-monitor`: Monitor portfolio and alerts
