"""XGBoost gradient boosting prediction model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseModel
from src.utils.config import get_model_config


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting prediction model."""

    # Columns that must never be used as input features (leak outcomes or are non-predictive IDs)
    _NON_FEATURE_COLS: set[str] = {
        "home_score", "away_score", "home_win", "away_win",
        "game_datetime", "id", "external_id", "game_id",
        "home_team_name", "away_team_name", "league",
    }

    def __init__(
        self,
        model_id: str = "gradient_boosting",
        version: str = "1.0.0",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the XGBoost model.

        Args:
            model_id: Model identifier.
            version: Model version.
            config: Optional configuration overrides.
        """
        super().__init__(model_id, version)

        self._config = config or get_model_config().get_model_config("gradient_boosting")
        self._feature_names: list[str] = []

        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required. Install with: pip install xgboost>=2.0.0") from e

        self._model = xgb.XGBClassifier(
            n_estimators=self._config.get("n_estimators", 300),
            max_depth=self._config.get("max_depth", 4),
            learning_rate=self._config.get("learning_rate", 0.05),
            subsample=self._config.get("subsample", 0.8),
            colsample_bytree=self._config.get("colsample_bytree", 0.8),
            min_child_weight=self._config.get("min_child_weight", 5),
            gamma=self._config.get("gamma", 1.0),
            reg_alpha=self._config.get("reg_alpha", 0.1),
            reg_lambda=self._config.get("reg_lambda", 1.0),
            eval_metric="logloss",
            random_state=self._config.get("random_state", 42),
            verbosity=0,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost model.

        Args:
            X: Feature matrix.
            y: Target variable (1 for home win, 0 for away win).
        """
        logger.info(f"Training XGBoost on {len(X)} samples...")

        feature_candidates = [c for c in X.columns if c not in self._NON_FEATURE_COLS]
        X_train = X[feature_candidates].copy()
        self._feature_names = feature_candidates

        X_train = X_train.fillna(0)

        self._model.fit(X_train, y)
        self._is_trained = True

        # Log top-10 feature importances
        importances = self.get_feature_importance()
        if importances:
            top10 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
            logger.info("Top-10 feature importances:")
            for feat, score in top10:
                logger.info(f"  {feat}: {score:.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [away_prob, home_prob].
        """
        if not self._is_trained:
            return np.full((len(X), 2), 0.5)

        if self._feature_names and all(f in X.columns for f in self._feature_names):
            X_pred = X[self._feature_names].copy()
        else:
            feature_candidates = [c for c in X.columns if c not in self._NON_FEATURE_COLS]
            X_pred = X[feature_candidates].copy()

        X_pred = X_pred.fillna(0)

        return self._model.predict_proba(X_pred)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self._is_trained:
            return {}

        scores = self._model.feature_importances_
        return dict(zip(self._feature_names, scores))
