# Position Sizer Agent

```yaml
name: position-sizer
description: Calculate optimal bet sizes using Kelly criterion

responsibilities:
  - Calculate Kelly criterion bet fractions
  - Apply fractional Kelly (quarter Kelly default)
  - Enforce maximum bet limits
  - Respect total exposure limits
  - Adjust for confidence levels
  - Size portfolio of simultaneous bets
  - Account for correlated positions

tools:
  - src/execution/position_sizer.py
  - src/risk/limits.py
  - config/risk_limits.yaml

verification:
  - Bet sizes respect all risk limits
  - Total exposure stays under max threshold
  - No single bet exceeds maximum
  - Kelly fractions are mathematically correct

kelly_formula:
  # f* = (p * b - q) / b
  # where:
  #   f* = optimal fraction of bankroll
  #   p = probability of winning
  #   b = net odds (decimal odds - 1)
  #   q = probability of losing (1 - p)

  example:
    probability: 0.65
    decimal_odds: 1.67
    b: 0.67
    q: 0.35
    raw_kelly: 0.224 (22.4%)
    quarter_kelly: 0.056 (5.6%)

limits:
  kelly_fraction: 0.25 (use quarter Kelly)
  max_bet_pct: 0.05 (5% of bankroll max)
  min_bet_pct: 0.01 (1% minimum to bother)
  max_total_exposure: 0.20 (20% total)
  max_single_game: 0.05 (5% per game)
  max_daily_bets: 10
```

## Usage

This agent determines how much to bet. Use it when:
- Sizing a new bet
- Rebalancing portfolio
- Adjusting after bankroll changes
- Implementing new risk limits

## Key Principles

1. **Never bet more than Kelly**: Overbetting leads to ruin
2. **Use fractional Kelly**: Quarter Kelly balances growth and safety
3. **Respect all limits**: Multiple limits (single bet, exposure, daily) all apply
4. **Adjust for confidence**: Lower confidence = smaller bets
