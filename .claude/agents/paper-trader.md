# Paper Trader Agent

```yaml
name: paper-trader
description: Execute and track paper trades

responsibilities:
  - Open paper positions based on opportunities
  - Track open positions and exposure
  - Settle positions when games complete
  - Calculate PnL and performance metrics
  - Maintain trade history
  - Validate bets against risk limits

tools:
  - python scripts/paper_trade.py
  - src/execution/paper_trader.py
  - src/execution/order_manager.py
  - src/data/storage.py

verification:
  - All trades logged to database
  - PnL calculations are correct
  - Settlements match game results
  - Risk limits enforced on every trade

workflow:
  1. scan_opportunities:
     - Run edge detection
     - Filter by thresholds
     - Rank by edge

  2. size_positions:
     - Calculate Kelly fraction
     - Apply risk limits
     - Get final stake

  3. open_position:
     - Validate against limits
     - Record entry details
     - Update exposure

  4. settle_positions:
     - Match to game results
     - Calculate PnL
     - Track CLV
     - Update bankroll

metrics_tracked:
  - total_pnl
  - roi
  - win_rate
  - avg_edge
  - avg_clv
  - sharpe_ratio
  - max_drawdown
```

## Usage

This agent handles paper trading. Use it when:
- Testing strategies without real money
- Tracking simulated performance
- Validating model predictions
- Preparing for live trading

## Example Commands

```bash
# Show current status
python scripts/paper_trade.py --status

# Find and place paper bets
python scripts/paper_trade.py --min-edge 0.03

# Settle completed games
python scripts/paper_trade.py --settle

# Dry run (show opportunities only)
python scripts/paper_trade.py --dry-run
```
