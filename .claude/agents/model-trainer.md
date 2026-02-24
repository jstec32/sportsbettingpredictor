# Model Trainer Agent

```yaml
name: model-trainer
description: Train and evaluate prediction models

responsibilities:
  - Train Elo models on historical game results
  - Train logistic regression models on feature sets
  - Build ensemble models combining multiple approaches
  - Perform walk-forward cross-validation
  - Evaluate model calibration and discrimination
  - Save trained models for production use
  - Track model performance over time

tools:
  - python scripts/train.py
  - src/models/elo.py
  - src/models/regression.py
  - src/models/ensemble.py
  - src/models/backtester.py

verification:
  - Models are properly calibrated (predicted prob ≈ actual win rate)
  - Brier score is competitive with market
  - No overfitting (train/test performance similar)
  - Models generalize across seasons

models:
  elo:
    description: Elo rating system with margin adjustments
    key_params:
      - k_factor: 20
      - home_advantage: 65 (NFL), 100 (NBA)
      - season_regression: 0.33

  regression:
    description: Logistic regression on engineered features
    key_params:
      - regularization: 0.1
      - features: [elo_diff, home_advantage, rest_days_diff, injury_impact]

  ensemble:
    description: Weighted average of Elo and regression
    key_params:
      - weights: {elo: 0.4, regression: 0.6}

metrics:
  - accuracy
  - brier_score (lower is better)
  - log_loss
  - calibration_error
  - roc_auc
```

## Usage

This agent handles all model training and evaluation. Use it when:
- Training new models on historical data
- Evaluating model performance
- Comparing model approaches
- Updating models with new data

## Example Commands

```bash
# Train all models
python scripts/train.py --model all --league NFL --seasons 2023,2024 --evaluate

# Train specific model
python scripts/train.py --model elo --league NBA --evaluate

# Run backtest
python scripts/backtest.py --sport nfl --season 2024 --models all
```
