"""Logistic regression prediction model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel
from src.utils.config import get_model_config


class LogisticRegressionModel(BaseModel):
    """Logistic regression-based prediction model."""

    def __init__(
        self,
        model_id: str = "logistic_regression",
        version: str = "1.0.0",
        config: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the logistic regression model.

        Args:
            model_id: Model identifier.
            version: Model version.
            config: Optional configuration overrides.
        """
        super().__init__(model_id, version)

        self._config = config or get_model_config().regression_config
        self._feature_names = self._config.get(
            "features",
            [
                "elo_diff",
                "home_advantage",
                "rest_days_diff",
                "injury_impact",
            ],
        )

        regularization = self._config.get("regularization", 0.1)
        solver = self._config.get("solver", "lbfgs")
        max_iter = self._config.get("max_iter", 1000)

        self._model = LogisticRegression(
            C=1.0 / regularization if regularization > 0 else 1e6,
            solver=solver,
            max_iter=max_iter,
            random_state=42,
        )
        self._scaler = StandardScaler()

    # Columns that must never be used as input features (they leak outcomes or are non-predictive IDs)
    _NON_FEATURE_COLS: set[str] = {
        "home_score", "away_score", "home_win", "away_win",
        "game_datetime", "id", "external_id", "game_id",
        "home_team_name", "away_team_name", "league",
    }

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the logistic regression model.

        Args:
            X: Feature matrix.
            y: Target variable (1 for home win, 0 for away win).
        """
        logger.info(f"Training logistic regression on {len(X)} samples...")

        # Use specified features if available, otherwise use all non-result columns
        if all(f in X.columns for f in self._feature_names):
            X_train = X[self._feature_names].copy()
        else:
            feature_candidates = [c for c in X.columns if c not in self._NON_FEATURE_COLS]
            X_train = X[feature_candidates].copy()
            self._feature_names = feature_candidates

        # Handle missing values
        X_train = X_train.fillna(0)

        # Scale features
        X_scaled = self._scaler.fit_transform(X_train)

        # Train model
        self._model.fit(X_scaled, y)
        self._is_trained = True

        # Log feature importance
        if hasattr(self._model, "coef_"):
            importances = dict(zip(self._feature_names, self._model.coef_[0]))
            logger.info(f"Feature coefficients: {importances}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [away_prob, home_prob].
        """
        if not self._is_trained:
            # Return default 50/50 if not trained
            return np.full((len(X), 2), 0.5)

        # Use specified features
        if all(f in X.columns for f in self._feature_names):
            X_pred = X[self._feature_names].copy()
        else:
            X_pred = X.copy()

        # Handle missing values
        X_pred = X_pred.fillna(0)

        # Scale features
        X_scaled = self._scaler.transform(X_pred)

        # Predict probabilities
        return self._model.predict_proba(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model coefficients.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self._is_trained or not hasattr(self._model, "coef_"):
            return {}

        # Use absolute value of coefficients as importance
        importances = np.abs(self._model.coef_[0])
        return dict(zip(self._feature_names, importances))

    def get_coefficients(self) -> Dict[str, float]:
        """Get raw model coefficients.

        Returns:
            Dictionary mapping feature names to coefficients.
        """
        if not self._is_trained or not hasattr(self._model, "coef_"):
            return {}

        return dict(zip(self._feature_names, self._model.coef_[0]))

    @staticmethod
    def build_features(
        elo_diff: float,
        home_advantage: float,
        rest_days_diff: float = 0,
        injury_impact_diff: float = 0,
        travel_factor: float = 0,
        back_to_back_home: bool = False,
        back_to_back_away: bool = False,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Build feature dictionary for prediction.

        Args:
            elo_diff: Elo rating difference (home - away).
            home_advantage: Home advantage value.
            rest_days_diff: Rest days difference (home - away).
            injury_impact_diff: Injury impact difference.
            travel_factor: Away team travel factor.
            back_to_back_home: Home team on back-to-back.
            back_to_back_away: Away team on back-to-back.
            **kwargs: Additional features.

        Returns:
            Feature dictionary.
        """
        features = {
            "elo_diff": elo_diff,
            "home_advantage": home_advantage,
            "rest_days_diff": rest_days_diff,
            "injury_impact": injury_impact_diff,
            "travel_distance": travel_factor,
            "back_to_back": float(back_to_back_away) - float(back_to_back_home),
        }

        # Add any additional features
        features.update(kwargs)

        return features
