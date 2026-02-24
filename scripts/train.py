#!/usr/bin/env python3
"""Train prediction models."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.models import EloModel, LogisticRegressionModel, XGBoostModel, EnsembleModel
from src.data.storage import DatabaseStorage
from src.features.situational import SituationalFeatureCalculator
from src.utils.logging_config import setup_logging

load_dotenv()


@click.command()
@click.option(
    "--model",
    type=click.Choice(["elo", "regression", "gradient_boosting", "ensemble", "all"]),
    default="all",
    help="Model to train",
)
@click.option(
    "--league",
    type=click.Choice(["NFL", "NBA", "all"]),
    default="all",
    help="League to train on",
)
@click.option(
    "--seasons",
    default="2024,2025,2026",
    help="Comma-separated seasons for training",
)
@click.option(
    "--save-dir",
    type=click.Path(),
    default="models",
    help="Directory to save trained models",
)
@click.option(
    "--evaluate",
    is_flag=True,
    help="Evaluate models after training",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    model: str,
    league: str,
    seasons: str,
    save_dir: str,
    evaluate: bool,
    verbose: bool,
) -> None:
    """Train prediction models on historical data."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    logger.info("Starting model training...")

    # Parse seasons
    season_list = [int(s.strip()) for s in seasons.split(",")]

    # Determine leagues
    leagues = ["NFL", "NBA"] if league == "all" else [league]

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    storage = DatabaseStorage()

    for lg in leagues:
        logger.info(f"Training for {lg}...")

        # Load training data
        train_data = load_training_data(storage, lg, season_list)

        if train_data.empty:
            logger.warning(f"No training data for {lg}")
            continue

        logger.info(f"  Training samples: {len(train_data)}")

        # Prepare features and target
        X, y = prepare_features(train_data)

        # Train models
        models_to_train = []
        if model in ("elo", "all"):
            models_to_train.append(("elo", EloModel()))
        if model in ("regression", "all"):
            models_to_train.append(("regression", LogisticRegressionModel()))
        if model in ("gradient_boosting", "all"):
            if league == "NFL":
                logger.warning("GBM training on NFL — note: NBA is the primary focus")
            models_to_train.append(("gradient_boosting", XGBoostModel()))

        trained_models = []
        for model_name, model_instance in models_to_train:
            logger.info(f"  Training {model_name}...")
            model_instance.train(X, y)
            trained_models.append(model_instance)

            # Evaluate if requested
            if evaluate:
                metrics = model_instance.evaluate(X, y)
                logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")
                logger.info(f"    Brier Score: {metrics['brier_score']:.4f}")

            # Save model
            model_path = save_path / f"{lg.lower()}_{model_name}.pkl"
            model_instance.save(str(model_path))
            logger.info(f"    Saved to {model_path}")

        # Train ensemble if requested
        if model in ("ensemble", "all") and len(trained_models) >= 2:
            logger.info("  Training ensemble...")
            ensemble = EnsembleModel(models=trained_models)

            if evaluate:
                metrics = ensemble.evaluate(X, y)
                logger.info(f"    Ensemble Accuracy: {metrics['ensemble_accuracy']:.3f}")

            ensemble_path = save_path / f"{lg.lower()}_ensemble.pkl"
            ensemble.save(str(ensemble_path))
            logger.info(f"    Saved to {ensemble_path}")

    logger.info("Training complete!")


def load_training_data(
    storage: DatabaseStorage,
    league: str,
    seasons: list[int],
) -> pd.DataFrame:
    """Load training data from database."""
    # Query completed games
    query = """
        SELECT g.id, g.external_id, g.season, g.game_datetime,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.is_neutral_site,
               ht.name as home_team_name, at.name as away_team_name
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.league = :league
          AND g.status = 'final'
          AND g.season = ANY(:seasons)
        ORDER BY g.game_datetime
    """

    try:
        result = storage.execute(query, {"league": league, "seasons": seasons})
        if not result:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row._mapping) for row in result])

        # Add target variable
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

        return df

    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return pd.DataFrame()


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from game data.

    Computes situational and form features in addition to base columns.
    Passes score/result columns alongside so the Elo model can update ratings,
    but those columns are excluded from regression model training via _NON_FEATURE_COLS.
    """
    logger.info("  Computing situational features...")
    sit_calc = SituationalFeatureCalculator()
    sit_features = sit_calc.compute_features_for_games(df)

    X = df[["home_team_id", "away_team_id", "is_neutral_site", "season",
            "home_score", "away_score"]].copy()
    X = pd.concat([X, sit_features], axis=1)

    y = df["home_win"]
    return X, y


if __name__ == "__main__":
    main()
