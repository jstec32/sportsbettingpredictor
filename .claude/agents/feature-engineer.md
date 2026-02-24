# Feature Engineer Agent

```yaml
name: feature-engineer
description: Build predictive features from raw data

responsibilities:
  - Calculate team-level metrics (Elo ratings, efficiency, SOS)
  - Apply injury adjustments weighted by player impact
  - Compute situational factors (rest days, travel, back-to-backs)
  - Build historical matchup features
  - Calculate weather impact features for NFL outdoor games
  - Derive market features from line movements
  - Ensure no look-ahead bias in feature construction

tools:
  - src/features/team_metrics.py
  - src/features/player_metrics.py
  - src/features/situational.py
  - src/features/market_features.py

verification:
  - Feature distributions match historical expectations
  - No future data leaks (look-ahead bias)
  - Missing data handled gracefully with sensible defaults
  - Features are normalized/scaled appropriately

feature_categories:
  team_metrics:
    - elo_rating
    - elo_diff
    - offensive_rating
    - defensive_rating
    - net_rating
    - home_record
    - away_record

  player_metrics:
    - injury_impact
    - key_players_out
    - qb_status

  situational:
    - rest_days
    - travel_miles
    - back_to_back
    - timezone_change

  market:
    - opening_spread
    - current_spread
    - line_movement
    - steam_move_indicator
    - consensus_probability
```

## Usage

This agent transforms raw data into predictive features. Use it when:
- Building features for model training
- Preparing features for new predictions
- Investigating feature importance
- Debugging prediction errors

## Key Principles

1. **No Look-Ahead Bias**: Features must only use data available at prediction time
2. **Handle Missing Data**: Use sensible defaults (league average, 0.5 probability)
3. **Scale Appropriately**: Normalize features for regression models
4. **Document Everything**: Each feature should have clear documentation
