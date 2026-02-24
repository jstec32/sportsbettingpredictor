delta.com# Sports Betting Predictor

A Kalshi sports betting prediction agent for NFL & NBA markets. Built with Python 3.11, PostgreSQL, and free-tier APIs.

## Features

- **Data Pipeline**: Ingest data from Kalshi, ESPN, The Odds API, and Open-Meteo
- **Feature Engineering**: Team metrics, injury adjustments, situational factors, weather impact
- **Prediction Models**: Elo ratings, logistic regression, ensemble methods
- **Edge Detection**: Find profitable betting opportunities with positive expected value
- **Position Sizing**: Kelly criterion-based bet sizing with configurable risk limits
- **Paper Trading**: Test strategies without real money
- **Risk Management**: Portfolio tracking, exposure limits, drawdown alerts

## Project Structure

```
SportsBettingPredictor/
├── .claude/              # Claude Code integration (agents, commands)
├── config/               # YAML configuration files
├── data/                 # Local data (gitignored)
├── docs/                 # Documentation
├── logs/                 # Log files (gitignored)
├── reports/              # Generated reports (gitignored)
├── scripts/              # CLI scripts for common tasks
├── src/                  # Main source code
│   ├── data/             # API clients and data ingestion
│   ├── features/         # Feature engineering
│   ├── models/           # Prediction models
│   ├── market/           # Market analysis
│   ├── execution/        # Trade execution
│   ├── risk/             # Risk management
│   └── utils/            # Utilities
└── tests/                # Test suite
```

## Prerequisites

- Python 3.11+
- PostgreSQL 14+
- API Keys:
  - The Odds API (free tier: 500 requests/month)
  - Kalshi account (optional for Phase 0)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/joshsteckler/SportsBettingPredictor.git
cd SportsBettingPredictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
make dev-install
```

4. Copy environment variables:
```bash
cp .env .env
# Edit .env with your API keys and database credentials
```

5. Setup the database:
```bash
make setup-db
```

## Usage

### Data Ingestion
```bash
python scripts/backfill_data.py --sport nfl --seasons 2023,2024
```

### Train Models
```bash
make train
```

### Run Backtest
```bash
make backtest
```

### Scan for Opportunities
```bash
make scan
```

### Generate Report
```bash
make report
```

## Configuration

Configuration files are in the `config/` directory:

- `database.yaml`: Database connection settings
- `models.yaml`: Model hyperparameters
- `risk_limits.yaml`: Risk management parameters

## Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Development

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make typecheck

# All checks
make verify
```

## API Rate Limits

| API | Limit | Notes |
|-----|-------|-------|
| The Odds API | 500/month | Free tier |
| ESPN | No official limit | Unofficial API, be respectful |
| Open-Meteo | Unlimited | Free, no key required |
| Kalshi | 100/min | With authentication |

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Sports betting involves risk of financial loss. Always bet responsibly and never bet more than you can afford to lose. The authors are not responsible for any financial losses incurred from using this software.
