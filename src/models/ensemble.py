"""Ensemble model combining multiple prediction models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseModel, Prediction
from src.utils.config import get_model_config


class EnsembleModel(BaseModel):
    """Ensemble model that combines predictions from multiple models."""

    def __init__(
        self,
        models: List[BaseModel],
        weights: Dict[str, float] | None = None,
        model_id: str = "ensemble",
        version: str = "1.0.0",
    ) -> None:
        """Initialize the ensemble model.

        Args:
            models: List of base models to combine.
            weights: Dictionary mapping model_id to weight.
                     If None, uses equal weights.
            model_id: Ensemble model identifier.
            version: Model version.
        """
        super().__init__(model_id, version)

        self.models = models
        self._model_map = {m.model_id: m for m in models}

        # Set weights
        if weights is None:
            # Equal weights
            self.weights = {m.model_id: 1.0 / len(models) for m in models}
        else:
            # Normalize provided weights
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}

        # Combine feature names from all models
        all_features = set()
        for model in models:
            all_features.update(model.feature_names)
        self._feature_names = list(all_features)

    @classmethod
    def from_config(
        cls,
        models: List[BaseModel],
        config: Dict[str, Any] | None = None,
    ) -> "EnsembleModel":
        """Create ensemble from configuration.

        Args:
            models: List of models to include.
            config: Optional configuration overrides.

        Returns:
            Configured EnsembleModel.
        """
        if config is None:
            config = get_model_config().ensemble_config

        weights = config.get("weights", {})
        return cls(models=models, weights=weights)

    @property
    def is_trained(self) -> bool:
        """Check if all component models are trained."""
        return all(m.is_trained for m in self.models)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all component models.

        Args:
            X: Feature matrix.
            y: Target variable.
        """
        logger.info(f"Training ensemble with {len(self.models)} models...")

        for model in self.models:
            logger.info(f"Training {model.model_id}...")
            model.train(X, y)

        self._is_trained = True
        logger.info("Ensemble training complete")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using weighted average.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [away_prob, home_prob].
        """
        n_samples = len(X)
        weighted_probs = np.zeros((n_samples, 2))

        for model in self.models:
            weight = self.weights.get(model.model_id, 0)
            if weight > 0:
                probs = model.predict_proba(X)
                weighted_probs += weight * probs

        return weighted_probs

    def predict_proba_with_details(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Predict probabilities and return individual model outputs.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (ensemble_probs, dict of model_id -> model_probs).
        """
        n_samples = len(X)
        weighted_probs = np.zeros((n_samples, 2))
        model_probs: Dict[str, np.ndarray] = {}

        for model in self.models:
            weight = self.weights.get(model.model_id, 0)
            probs = model.predict_proba(X)
            model_probs[model.model_id] = probs

            if weight > 0:
                weighted_probs += weight * probs

        return weighted_probs, model_probs

    def make_prediction(
        self,
        game_id: int,
        features: Dict[str, float],
        prediction_type: str = "moneyline",
    ) -> Prediction:
        """Make a prediction with ensemble confidence.

        Args:
            game_id: Game ID.
            features: Feature dictionary.
            prediction_type: Type of prediction.

        Returns:
            Prediction object with ensemble metadata.
        """
        X = pd.DataFrame([features])
        ensemble_probs, model_probs = self.predict_proba_with_details(X)

        home_prob = ensemble_probs[0, 1]
        away_prob = ensemble_probs[0, 0]

        # Calculate confidence based on model agreement
        home_probs_all = [probs[0, 1] for probs in model_probs.values()]
        std_dev = np.std(home_probs_all)

        # Lower std = higher agreement = higher confidence
        agreement_factor = max(0, 1 - std_dev * 2)
        prob_confidence = abs(home_prob - 0.5) * 2
        confidence = (agreement_factor + prob_confidence) / 2

        # Add model predictions to features
        detailed_features = features.copy()
        for model_id, probs in model_probs.items():
            detailed_features[f"{model_id}_home_prob"] = probs[0, 1]

        # Estimate spread
        predicted_spread = None
        if home_prob != 0.5:
            predicted_spread = (home_prob - 0.5) / 0.027

        return Prediction(
            game_id=game_id,
            model_id=self.model_id,
            model_version=self.version,
            prediction_type=prediction_type,
            predicted_at=pd.Timestamp.now(),
            home_win_prob=home_prob,
            away_win_prob=away_prob,
            predicted_spread=predicted_spread,
            confidence=confidence,
            features=detailed_features,
        )

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble and component models.

        Args:
            X: Feature matrix.
            y: True outcomes.

        Returns:
            Dictionary of metrics for ensemble and each model.
        """
        metrics = {}

        # Ensemble metrics
        ensemble_metrics = super().evaluate(X, y)
        for key, value in ensemble_metrics.items():
            metrics[f"ensemble_{key}"] = value

        # Individual model metrics
        for model in self.models:
            model_metrics = model.evaluate(X, y)
            for key, value in model_metrics.items():
                metrics[f"{model.model_id}_{key}"] = value

        return metrics

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update model weights.

        Args:
            weights: New weights (will be normalized).
        """
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        logger.info(f"Updated ensemble weights: {self.weights}")

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add a new model to the ensemble.

        Args:
            model: Model to add.
            weight: Initial weight for the model.
        """
        self.models.append(model)
        self._model_map[model.model_id] = model

        # Rebalance weights
        current_total = sum(self.weights.values())
        new_total = current_total + weight
        for model_id in self.weights:
            self.weights[model_id] *= current_total / new_total
        self.weights[model.model_id] = weight / new_total

        # Update feature names
        self._feature_names = list(set(self._feature_names) | set(model.feature_names))

        logger.info(f"Added {model.model_id} to ensemble with weight {weight / new_total:.3f}")

    def remove_model(self, model_id: str) -> None:
        """Remove a model from the ensemble.

        Args:
            model_id: ID of model to remove.
        """
        if model_id not in self._model_map:
            logger.warning(f"Model {model_id} not found in ensemble")
            return

        self.models = [m for m in self.models if m.model_id != model_id]
        del self._model_map[model_id]

        # Rebalance weights
        if model_id in self.weights:
            del self.weights[model_id]
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"Removed {model_id} from ensemble")
