"""Elo-based prediction model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseModel
from src.features.team_metrics import TeamMetricsCalculator
from src.utils.config import get_model_config


class EloModel(BaseModel):
    """Elo rating-based prediction model."""

    def __init__(
        self,
        model_id: str = "elo",
        version: str = "1.0.0",
        config: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Elo model.

        Args:
            model_id: Model identifier.
            version: Model version.
            config: Optional configuration overrides.
        """
        super().__init__(model_id, version)

        self._config = config or get_model_config().elo_config
        self._metrics = TeamMetricsCalculator()
        self._feature_names = ["elo_diff", "home_advantage", "is_neutral"]

    @property
    def initial_rating(self) -> float:
        """Initial Elo rating."""
        return self._config.get("initial_rating", 1500)

    @property
    def k_factor(self) -> float:
        """K-factor for updates."""
        return self._config.get("k_factor", 20)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Elo model by processing historical games.

        For Elo, "training" means processing games chronologically
        to establish current ratings.

        Args:
            X: Feature matrix with columns:
               - home_team_id
               - away_team_id
               - home_score
               - away_score
               - league
               - is_neutral
               - game_datetime
            y: Target variable (1 for home win, 0 for away win).
        """
        logger.info(f"Training Elo model on {len(X)} games...")

        # Reset ratings
        self._metrics._elo_ratings.clear()

        # Sort by datetime
        if "game_datetime" in X.columns:
            X = X.sort_values("game_datetime")

        # Process each game
        for idx, row in X.iterrows():
            home_team_id = row["home_team_id"]
            away_team_id = row["away_team_id"]
            home_score = row.get("home_score", 0)
            away_score = row.get("away_score", 0)
            league = row.get("league", "NFL")
            is_neutral = row.get("is_neutral", False)

            # Skip if no scores (game not played)
            if pd.isna(home_score) or pd.isna(away_score):
                continue

            self._metrics.process_game_result(
                home_team_id=int(home_team_id),
                away_team_id=int(away_team_id),
                home_score=int(home_score),
                away_score=int(away_score),
                league=str(league),
                is_neutral=bool(is_neutral),
            )

        self._is_trained = True
        logger.info(f"Elo model trained. {len(self._metrics._elo_ratings)} teams rated.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probabilities using Elo ratings.

        Args:
            X: Feature matrix with columns:
               - home_team_id
               - away_team_id
               - league
               - is_neutral (optional)

        Returns:
            Array of shape (n_samples, 2) with [away_prob, home_prob].
        """
        probas = []

        for idx, row in X.iterrows():
            home_team_id = int(row["home_team_id"])
            away_team_id = int(row["away_team_id"])
            league = str(row.get("league", "NFL"))
            is_neutral = bool(row.get("is_neutral", False))

            home_prob = self._metrics.calculate_win_probability(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                league=league,
                is_neutral=is_neutral,
            )

            probas.append([1 - home_prob, home_prob])

        return np.array(probas)

    def get_elo(self, team_id: int) -> float:
        """Get current Elo rating for a team.

        Args:
            team_id: Team ID.

        Returns:
            Current Elo rating.
        """
        return self._metrics.get_elo(team_id)

    def get_all_ratings(self) -> Dict[int, float]:
        """Get all current Elo ratings.

        Returns:
            Dictionary mapping team_id to rating.
        """
        return self._metrics.get_all_ratings()

    def regress_ratings(self, regression_factor: Optional[float] = None) -> None:
        """Regress all ratings toward the mean (for new season).

        Args:
            regression_factor: How much to regress (0-1).
        """
        for team_id in list(self._metrics._elo_ratings.keys()):
            self._metrics.regress_to_mean(team_id, regression_factor)

        logger.info("Ratings regressed to mean for new season")

    def get_features(
        self,
        home_team_id: int,
        away_team_id: int,
        league: str,
        is_neutral: bool = False,
    ) -> Dict[str, float]:
        """Get Elo-based features for a game.

        Args:
            home_team_id: Home team ID.
            away_team_id: Away team ID.
            league: League code.
            is_neutral: Whether game is at neutral site.

        Returns:
            Dictionary of Elo features.
        """
        return self._metrics.build_features(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league=league,
            is_neutral=is_neutral,
        )

    def expected_margin(
        self,
        home_team_id: int,
        away_team_id: int,
        league: str,
        is_neutral: bool = False,
    ) -> float:
        """Calculate expected point margin.

        Args:
            home_team_id: Home team ID.
            away_team_id: Away team ID.
            league: League code.
            is_neutral: Whether game is at neutral site.

        Returns:
            Expected point margin (positive = home favored).
        """
        home_prob = self._metrics.calculate_win_probability(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league=league,
            is_neutral=is_neutral,
        )

        # Convert probability to margin
        # Rough conversion: each point ~2.7% in probability
        return (home_prob - 0.5) / 0.027
