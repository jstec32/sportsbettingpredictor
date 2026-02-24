"""Base model interface for prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Prediction:
    """A single game prediction."""

    game_id: int
    model_id: str
    model_version: str
    prediction_type: str  # 'moneyline', 'spread', 'total'
    predicted_at: datetime
    home_win_prob: float
    away_win_prob: float
    predicted_spread: Optional[float] = None
    predicted_total: Optional[float] = None
    confidence: float = 0.5
    edge: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "game_id": self.game_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prediction_type": self.prediction_type,
            "predicted_at": self.predicted_at,
            "home_win_prob": self.home_win_prob,
            "away_win_prob": self.away_win_prob,
            "predicted_spread": self.predicted_spread,
            "predicted_total": self.predicted_total,
            "confidence": self.confidence,
            "edge": self.edge,
            "features_json": self.features,
        }


class BaseModel(ABC):
    """Abstract base class for prediction models."""

    def __init__(self, model_id: str, version: str = "1.0.0") -> None:
        """Initialize the model.

        Args:
            model_id: Unique identifier for the model.
            version: Model version string.
        """
        self.model_id = model_id
        self.version = version
        self._is_trained = False
        self._feature_names: List[str] = []

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    @property
    def feature_names(self) -> List[str]:
        """Get feature names used by the model."""
        return self._feature_names

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on historical data.

        Args:
            X: Feature matrix.
            y: Target variable (1 for home win, 0 for away win).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [away_prob, home_prob].
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict outcomes (0 for away win, 1 for home win).

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted outcomes.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def make_prediction(
        self,
        game_id: int,
        features: Dict[str, float],
        prediction_type: str = "moneyline",
    ) -> Prediction:
        """Make a prediction for a single game.

        Args:
            game_id: Game ID.
            features: Feature dictionary.
            prediction_type: Type of prediction.

        Returns:
            Prediction object.
        """
        X = pd.DataFrame([features])
        proba = self.predict_proba(X)[0]

        home_prob = proba[1]
        away_prob = proba[0]

        # Calculate confidence (distance from 50%)
        confidence = abs(home_prob - 0.5) * 2

        # Estimate spread from probability
        # Rough conversion: each point ~2.7% probability
        predicted_spread = None
        if home_prob != 0.5:
            predicted_spread = (home_prob - 0.5) / 0.027

        return Prediction(
            game_id=game_id,
            model_id=self.model_id,
            model_version=self.version,
            prediction_type=prediction_type,
            predicted_at=datetime.now(),
            home_win_prob=home_prob,
            away_win_prob=away_prob,
            predicted_spread=predicted_spread,
            confidence=confidence,
            features=features,
        )

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Feature matrix.
            y: True outcomes.

        Returns:
            Dictionary of evaluation metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )

        proba = self.predict_proba(X)
        predictions = self.predict(X)
        home_probs = proba[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "brier_score": brier_score_loss(y, home_probs),
            "log_loss": log_loss(y, home_probs),
        }

        # ROC AUC if we have both classes
        if len(np.unique(y)) > 1:
            metrics["roc_auc"] = roc_auc_score(y, home_probs)

        # Calibration metrics
        metrics["mean_predicted_prob"] = float(np.mean(home_probs))
        metrics["actual_win_rate"] = float(np.mean(y))

        return metrics

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model.
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk.

        Args:
            path: Path to model file.

        Returns:
            Loaded model instance.
        """
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
