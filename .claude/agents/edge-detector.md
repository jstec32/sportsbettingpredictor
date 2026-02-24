# Edge Detector Agent

```yaml
name: edge-detector
description: Find profitable betting opportunities

responsibilities:
  - Compare model probabilities to market odds
  - Calculate edge for each potential bet
  - Filter opportunities by minimum edge threshold
  - Rank opportunities by expected value
  - Detect sharp action and line movement
  - Track closing line value (CLV)
  - Identify market inefficiencies

tools:
  - python scripts/scan_markets.py
  - src/market/edge_calculator.py
  - src/market/line_movement.py
  - src/market/clv_tracker.py

verification:
  - Edges are statistically significant
  - CLV is positive over time (beating the close)
  - Opportunities are actionable (sufficient liquidity)
  - No stale or expired markets

thresholds:
  min_edge: 0.03 (3%)
  min_confidence: 0.55
  max_time_to_close: 48 hours
  min_time_to_close: 30 minutes

edge_calculation:
  # Edge = Model Probability - Market Probability
  # Only bet when edge > min_edge AND confidence > min_confidence

  example:
    model_prob: 0.65
    market_prob: 0.60
    edge: 0.05 (5%)
    action: BET (edge > 3% threshold)

sharp_indicators:
  - Steam moves (1+ point in < 2 hours)
  - Reverse line movement
  - Significant late money
  - Line moves against public betting %
```

## Usage

This agent identifies betting opportunities. Use it when:
- Scanning for new opportunities
- Analyzing specific games
- Investigating line movements
- Tracking historical CLV

## Example Commands

```bash
# Scan all markets
python scripts/scan_markets.py --sport all --min-edge 0.03

# Refresh data and scan
python scripts/scan_markets.py --refresh-data --min-edge 0.025

# Check specific sport
python scripts/scan_markets.py --sport nfl --min-confidence 0.60
```
