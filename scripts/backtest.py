#!/usr/bin/env python3
"""Run backtests on prediction models."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.models import BaseModel, EloModel, XGBoostModel, EnsembleModel
from src.models.backtester import Backtester, BacktestResult
from src.data.storage import DatabaseStorage
from src.features.situational import SituationalFeatureCalculator
from src.utils.logging_config import setup_logging

load_dotenv()


@click.command()
@click.option(
    "--sport",
    type=click.Choice(["nfl", "nba"]),
    default="nba",
    help="Sport to backtest",
)
@click.option(
    "--season",
    type=int,
    default=2026,
    help="Season to backtest",
)
@click.option(
    "--models",
    default="all",
    help="Models to test (comma-separated or 'all')",
)
@click.option(
    "--min-edge",
    type=float,
    default=0.03,
    help="Minimum edge threshold for simulated bets",
)
@click.option(
    "--train-pct",
    type=float,
    default=0.7,
    help="Percentage of data to use for training (0.0-1.0)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file for results",
)
@click.option(
    "--real-odds-only",
    is_flag=True,
    default=False,
    help="Only evaluate games that have real captured odds (skip synthetic)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    sport: str,
    season: int,
    models: str,
    min_edge: float,
    train_pct: float,
    output: str | None,
    real_odds_only: bool,
    verbose: bool,
) -> None:
    """Run backtests on prediction models."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    logger.info(f"Starting backtest for {sport.upper()} {season}...")

    storage = DatabaseStorage()
    league = sport.upper()

    # Load data with market probabilities
    data = load_backtest_data(storage, league, season)

    if data.empty:
        logger.error("No data available for backtest")
        return

    logger.info(f"Loaded {len(data)} games for backtest")
    logger.info(f"  Home win rate: {data['home_win'].mean():.1%}")

    if "has_real_odds" in data.columns:
        real_count = int(data["has_real_odds"].sum())
        logger.info(f"  Real odds coverage: {real_count}/{len(data)} games ({real_count/len(data):.1%})")

    if real_odds_only:
        if "has_real_odds" not in data.columns or data["has_real_odds"].sum() == 0:
            logger.error("--real-odds-only: no games with real odds found. Run capture-odds and settle-scores first.")
            return
        data = data[data["has_real_odds"]].copy()
        logger.info(f"  --real-odds-only: filtered to {len(data)} games with real odds")

    # Determine which models to test
    model_names = ["elo", "gradient_boosting", "ensemble"] if models == "all" else models.split(",")

    results = {}

    for model_name in model_names:
        logger.info(f"\nTesting {model_name}...")

        # Run walk-forward or simple backtest
        if "game_datetime" in data.columns and len(data) > 100:
            result = run_walkforward_backtest(data, model_name, league, min_edge)
        else:
            result = run_simple_backtest(data, model_name, league, train_pct, min_edge)

        if result is None:
            continue

        # Store results
        results[model_name] = result.to_dict()

        # Print summary
        print_backtest_results(model_name, result)

    # Compare models
    if len(results) > 1:
        print_model_comparison(results)

    # Save results
    save_results(results, league, season, output)

    logger.info("\nBacktest complete!")


def load_backtest_data(
    storage: DatabaseStorage,
    league: str,
    season: int,
) -> pd.DataFrame:
    """Load data for backtesting with market probabilities."""
    # Get games with scores
    games_query = """
        SELECT g.id, g.external_id, g.season, g.game_datetime,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.is_neutral_site,
               ht.name as home_team_name, at.name as away_team_name
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.league = :league
          AND g.status = 'final'
          AND g.season = :season
        ORDER BY g.game_datetime
    """

    try:
        result = storage.execute(games_query, {"league": league, "season": season})
        if not result:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row._mapping) for row in result])
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

        # Add situational and form features (no look-ahead)
        logger.info("  Computing situational features...")
        sit_calc = SituationalFeatureCalculator()
        sit_features = sit_calc.compute_features_for_games(df)
        df = pd.concat([df, sit_features], axis=1)

        # Get market probabilities for each game (earliest captured odds)
        odds_query = """
            SELECT DISTINCT ON (game_id)
                game_id, home_prob as market_home_prob, away_prob as market_away_prob,
                home_odds, away_odds
            FROM odds_history
            WHERE game_id IS NOT NULL
            ORDER BY game_id, captured_at ASC
        """
        odds_result = storage.execute(odds_query, {})

        if odds_result:
            odds_df = pd.DataFrame([dict(row._mapping) for row in odds_result])
            df = df.merge(odds_df, left_on="id", right_on="game_id", how="left")

        # Generate synthetic market probabilities for games without actual odds
        # This enables betting simulation for backtesting
        if "market_home_prob" not in df.columns:
            df["market_home_prob"] = None

        # Track which rows have real odds before filling synthetic
        df["has_real_odds"] = df["market_home_prob"].notna()
        real_count = int(df["has_real_odds"].sum())
        missing_odds = int(df["market_home_prob"].isna().sum())
        logger.info(f"  Market odds: {real_count} real, {missing_odds} synthetic")

        if missing_odds > 0:
            df = generate_synthetic_market_probs(df, league)

        return df

    except Exception as e:
        logger.error(f"Error loading backtest data: {e}")
        return pd.DataFrame()


def generate_synthetic_market_probs(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """Generate synthetic market probabilities for games without actual odds.

    Uses a simple Elo-based model to estimate what reasonable market odds would have been.
    This is a common approach when historical odds aren't available for backtesting.
    """
    # Process games chronologically to build up ratings
    df = df.sort_values("game_datetime").copy()

    # Initialize team ratings
    team_ratings: Dict[int, float] = {}
    initial_rating = 1500.0
    k_factor = 20.0

    # Home advantage in Elo points
    home_advantage_elo = 100.0 if league == "NBA" else 65.0

    # Set random seed for reproducibility
    np.random.seed(42)

    synthetic_probs = []
    for idx, row in df.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]

        # Get current ratings (or initialize)
        home_rating = team_ratings.get(home_id, initial_rating)
        away_rating = team_ratings.get(away_id, initial_rating)

        if pd.notna(row.get("market_home_prob")):
            # Keep existing market prob
            synthetic_probs.append(float(row["market_home_prob"]))
        else:
            # Generate synthetic market probability
            # Calculate expected home win probability
            elo_diff = home_rating - away_rating
            if not row["is_neutral_site"]:
                elo_diff += home_advantage_elo

            # Convert to probability
            base_prob = 1 / (1 + 10 ** (-elo_diff / 400))

            # Add small random noise to simulate market variance (±2%)
            noise = np.random.uniform(-0.02, 0.02)
            market_prob = float(np.clip(base_prob + noise, 0.15, 0.85))

            synthetic_probs.append(market_prob)

        # Update Elo ratings with the game result
        if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
            actual = 1.0 if home_score > away_score else 0.0

            # Expected score without home advantage for rating update
            expected = 1 / (1 + 10 ** (-(home_rating - away_rating) / 400))

            # Margin of victory multiplier
            mov = abs(home_score - away_score)
            mov_mult = np.log(max(mov, 1) + 1) * (2.2 / (((home_rating - away_rating) * 0.001) + 2.2))

            # Update ratings
            change = k_factor * mov_mult * (actual - expected)
            team_ratings[home_id] = home_rating + change
            team_ratings[away_id] = away_rating - change

    df["market_home_prob"] = synthetic_probs
    return df


def run_walkforward_backtest(
    data: pd.DataFrame,
    model_name: str,
    league: str,
    min_edge: float,
) -> Optional[BacktestResult]:
    """Run walk-forward backtesting."""
    logger.info("  Using walk-forward validation...")

    # Create model
    model = create_model(model_name)
    if model is None:
        return None

    # Configure backtester for shorter windows given our data
    config = {
        "train_window_days": 60,  # 2 months training
        "test_window_days": 7,    # 1 week test
        "step_days": 7,           # Step 1 week
        "min_samples": 30,        # Minimum training samples
    }

    backtester = Backtester(model, config)

    # Pass all available columns; backtester excludes result columns at predict time.
    # home_score/away_score are kept so the Elo model can update ratings during training.
    backtest_data = data.copy()

    try:
        result = backtester.run_walkforward(
            backtest_data,
            target_col="home_win",
            date_col="game_datetime",
            min_edge=min_edge,
        )
        return result
    except Exception as e:
        logger.error(f"  Walk-forward backtest failed: {e}")
        return run_simple_backtest(data, model_name, league, 0.7, min_edge)


def run_simple_backtest(
    data: pd.DataFrame,
    model_name: str,
    league: str,
    train_pct: float,
    min_edge: float,
) -> Optional[BacktestResult]:
    """Run simple train/test split backtest."""
    logger.info(f"  Using {train_pct:.0%}/{1-train_pct:.0%} train/test split...")

    # Sort by date
    if "game_datetime" in data.columns:
        data = data.sort_values("game_datetime")

    # Split
    split_idx = int(len(data) * train_pct)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()

    logger.info(f"  Train: {len(train_data)} games, Test: {len(test_data)} games")

    # Create and train model
    model = create_model(model_name)
    if model is None:
        return None

    # Include scores so Elo can update ratings; regression excludes them via _NON_FEATURE_COLS
    y_train = train_data["home_win"]
    X_train = train_data.drop(columns=["home_win", "market_home_prob"], errors="ignore").copy()

    try:
        model.train(X_train, y_train)
    except Exception as e:
        logger.error(f"  Training failed: {e}")
        return None

    # Prepare test data with market probs
    test_df = test_data.drop(columns=["market_home_prob"], errors="ignore").copy()
    if "market_home_prob" in test_data.columns:
        test_df["market_home_prob"] = test_data["market_home_prob"].values

    # Run backtest
    backtester = Backtester(model)

    try:
        result = backtester.run_simple(
            train_data.drop(columns=["market_home_prob"], errors="ignore"),
            test_df,
            target_col="home_win",
            min_edge=min_edge,
        )

        # Add betting simulation with actual odds
        if "market_home_prob" in test_data.columns:
            result = simulate_betting(model, test_data, result, min_edge)

        return result

    except Exception as e:
        logger.error(f"  Backtest failed: {e}")
        return None


def create_model(model_name: str) -> Optional[BaseModel]:
    """Create a new model instance."""
    if model_name == "elo":
        return EloModel()
    elif model_name == "regression":
        from src.models import LogisticRegressionModel
        return LogisticRegressionModel()
    elif model_name == "gradient_boosting":
        return XGBoostModel()
    elif model_name == "ensemble":
        elo = EloModel()
        gbm = XGBoostModel()
        return EnsembleModel(models=[elo, gbm])
    else:
        logger.error(f"Unknown model: {model_name}")
        return None


def simulate_betting(
    model: BaseModel,
    test_data: pd.DataFrame,
    result: BacktestResult,
    min_edge: float,
) -> BacktestResult:
    """Simulate actual betting with odds and track P&L."""
    feature_cols = ["home_team_id", "away_team_id", "is_neutral_site", "season"]

    X_test = test_data[feature_cols].copy()
    proba = model.predict_proba(X_test)
    home_probs = proba[:, 1]

    market_home_probs = test_data["market_home_prob"].values
    actuals = test_data["home_win"].values

    # Calculate edges
    home_edges = home_probs - market_home_probs
    away_edges = (1 - home_probs) - (1 - market_home_probs)

    # Simulate betting with Kelly sizing
    bankroll = 1000.0
    initial_bankroll = bankroll
    daily_returns = []
    bets = []

    for i in range(len(test_data)):
        if pd.isna(market_home_probs[i]):
            continue

        # Check for edge
        if home_edges[i] >= min_edge:
            # Bet on home
            decimal_odds = 1 / market_home_probs[i] if market_home_probs[i] > 0 else 2.0
            kelly = calculate_kelly(home_probs[i], decimal_odds)
            stake = bankroll * kelly * 0.25  # Quarter Kelly
            stake = min(stake, bankroll * 0.05)  # Max 5% per bet

            if stake > 0:
                won = actuals[i] == 1
                pnl = stake * (decimal_odds - 1) if won else -stake
                bankroll += pnl
                bets.append({
                    "side": "home",
                    "edge": home_edges[i],
                    "stake": stake,
                    "won": won,
                    "pnl": pnl,
                })

        elif away_edges[i] >= min_edge:
            # Bet on away
            market_away = 1 - market_home_probs[i]
            decimal_odds = 1 / market_away if market_away > 0 else 2.0
            away_prob = 1 - home_probs[i]
            kelly = calculate_kelly(away_prob, decimal_odds)
            stake = bankroll * kelly * 0.25
            stake = min(stake, bankroll * 0.05)

            if stake > 0:
                won = actuals[i] == 0
                pnl = stake * (decimal_odds - 1) if won else -stake
                bankroll += pnl
                bets.append({
                    "side": "away",
                    "edge": away_edges[i],
                    "stake": stake,
                    "won": won,
                    "pnl": pnl,
                })

    if bets:
        # Calculate betting metrics
        result.total_bets = len(bets)
        result.winning_bets = sum(1 for b in bets if b["won"])
        result.avg_edge = float(np.mean([b["edge"] for b in bets]))

        total_pnl = bankroll - initial_bankroll
        total_staked = sum(b["stake"] for b in bets)
        result.roi = total_pnl / total_staked if total_staked > 0 else 0

        # Calculate Sharpe ratio (annualized)
        pnl_series = [b["pnl"] for b in bets]
        if len(pnl_series) > 1:
            mean_pnl = np.mean(pnl_series)
            std_pnl = np.std(pnl_series)
            if std_pnl > 0:
                # Annualize assuming ~3 bets per day
                result.sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(365 * 3)

        # Calculate max drawdown
        cumulative = np.cumsum([0] + pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        result.max_drawdown = float(np.min(drawdowns) / initial_bankroll) if len(drawdowns) > 0 else 0

        result.daily_pnl = pnl_series

    return result


def calculate_kelly(prob: float, decimal_odds: float) -> float:
    """Calculate Kelly criterion fraction."""
    if decimal_odds <= 1:
        return 0.0

    q = 1 - prob
    b = decimal_odds - 1

    if b <= 0:
        return 0.0

    kelly = (prob * b - q) / b
    return max(0.0, kelly)


def print_backtest_results(model_name: str, result: BacktestResult) -> None:
    """Print formatted backtest results."""
    logger.info(f"\n  === {model_name.upper()} Results ===")
    logger.info(f"  Predictions: {result.n_predictions}")
    logger.info(f"  Accuracy:    {result.accuracy:.1%}")
    logger.info(f"  Brier Score: {result.brier_score:.4f}")
    logger.info(f"  Log Loss:    {result.log_loss:.4f}")
    logger.info(f"  Calibration: {result.calibration_error:.4f}")

    if result.total_bets > 0:
        logger.info(f"\n  --- Betting Simulation ---")
        logger.info(f"  Total Bets:  {result.total_bets}")
        logger.info(f"  Win Rate:    {result.winning_bets / result.total_bets:.1%}")
        logger.info(f"  Avg Edge:    {result.avg_edge:.1%}")
        logger.info(f"  ROI:         {result.roi:.1%}")
        if result.sharpe_ratio is not None:
            logger.info(f"  Sharpe:      {result.sharpe_ratio:.2f}")
        if result.max_drawdown is not None:
            logger.info(f"  Max DD:      {result.max_drawdown:.1%}")


def print_model_comparison(results: Dict[str, Dict]) -> None:
    """Print comparison of model results."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    headers = ["Model", "Accuracy", "Brier", "LogLoss", "ROI", "Bets"]
    row_format = "{:<12} {:>10} {:>8} {:>8} {:>8} {:>6}"

    logger.info(row_format.format(*headers))
    logger.info("-" * 60)

    for model_name, res in results.items():
        roi_str = f"{res.get('roi', 0):.1%}" if res.get('roi') else "N/A"
        bets = res.get('total_bets', 0)

        logger.info(row_format.format(
            model_name,
            f"{res['accuracy']:.1%}",
            f"{res['brier_score']:.4f}",
            f"{res['log_loss']:.4f}",
            roi_str,
            bets,
        ))


def save_results(
    results: Dict[str, Any],
    league: str,
    season: int,
    output: Optional[str],
) -> None:
    """Save backtest results to file."""
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"reports/backtest_{league.lower()}_{season}_{datetime.now():%Y%m%d_%H%M}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
