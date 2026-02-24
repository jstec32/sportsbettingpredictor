# Risk Monitor Agent

```yaml
name: risk-monitor
description: Monitor portfolio risk and generate alerts

responsibilities:
  - Track portfolio exposure and PnL
  - Monitor drawdown levels
  - Enforce risk limits
  - Generate alerts for threshold breaches
  - Take portfolio snapshots
  - Analyze risk metrics over time
  - Recommend position adjustments

tools:
  - python scripts/generate_report.py
  - python scripts/verify_data.py
  - src/risk/portfolio.py
  - src/risk/limits.py
  - src/risk/alerts.py

verification:
  - Alerts fire at correct thresholds
  - Drawdown calculations are accurate
  - Exposure tracking matches positions
  - Risk limits are enforced

alert_thresholds:
  drawdown:
    warning: 0.10 (10%)
    reduce_size: 0.15 (15%)
    halt: 0.20 (20%)

  exposure:
    warning: 0.15 (15%)
    exceeded: 0.20 (20%)

  losing_streak:
    warning: 5 consecutive
    error: 7 consecutive

  daily_loss:
    warning: 0.05 (5%)

actions:
  warning: Log and continue
  reduce_size: Reduce position sizes by 50%
  halt: Stop all new trading
  exceeded: Reject new positions

metrics_monitored:
  - current_bankroll
  - total_exposure
  - current_drawdown
  - max_drawdown
  - daily_pnl
  - weekly_pnl
  - win_rate
  - sharpe_ratio
  - avg_clv
```

## Usage

This agent monitors risk. Use it when:
- Checking portfolio health
- Investigating drawdowns
- Reviewing risk metrics
- Generating performance reports

## Example Commands

```bash
# Generate report
python scripts/generate_report.py --period week

# Check data quality
python scripts/verify_data.py --check-all

# Full report with all metrics
python scripts/generate_report.py --period month --metrics roi,sharpe,clv,drawdown
```

## Risk Limits Configuration

See `config/risk_limits.yaml` for all configurable limits:

```yaml
position_sizing:
  kelly_fraction: 0.25
  max_bet_pct: 0.05
  min_edge_threshold: 0.03

exposure_limits:
  max_total_exposure: 0.20
  max_single_game: 0.05
  max_daily_bets: 10

drawdown_limits:
  warning_threshold: 0.10
  halt_threshold: 0.20
```
