"""Backtesting framework for prediction models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseModel
from src.utils.config import get_model_config


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    model_id: str
    start_date: datetime
    end_date: datetime
    n_predictions: int
    n_correct: int

    # Performance metrics
    accuracy: float
    brier_score: float
    log_loss: float
    calibration_error: float

    # Betting metrics
    roi: Optional[float] = None
    total_bets: int = 0
    winning_bets: int = 0
    avg_edge: Optional[float] = None
    avg_clv: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Details
    predictions_df: Optional[pd.DataFrame] = None
    daily_pnl: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "n_predictions": self.n_predictions,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "calibration_error": self.calibration_error,
            "roi": self.roi,
            "total_bets": self.total_bets,
            "winning_bets": self.winning_bets,
            "avg_edge": self.avg_edge,
            "avg_clv": self.avg_clv,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
        }


class Backtester:
    """Backtesting engine for evaluating models."""

    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the backtester.

        Args:
            model: Model to backtest.
            config: Optional configuration overrides.
        """
        self.model = model
        self._config = config or get_model_config().get_model_config("backtesting")

        self.train_window_days = self._config.get("train_window_days", 365)
        self.test_window_days = self._config.get("test_window_days", 30)
        self.step_days = self._config.get("step_days", 7)
        self.min_samples = self._config.get("min_samples", 100)

    def run_walkforward(
        self,
        data: pd.DataFrame,
        target_col: str = "home_win",
        date_col: str = "game_datetime",
        min_edge: float = 0.03,
    ) -> BacktestResult:
        """Run walk-forward backtesting.

        Args:
            data: DataFrame with features and target.
            target_col: Name of target column.
            date_col: Name of date column.
            min_edge: Minimum edge to count as a bet.

        Returns:
            BacktestResult object.
        """
        logger.info(f"Running walk-forward backtest for {self.model.model_id}")

        # Sort by date
        data = data.sort_values(date_col).copy()

        # Convert to datetime and normalize to timezone-naive UTC
        dates = pd.to_datetime(data[date_col], utc=True).dt.tz_localize(None)
        data[date_col] = dates

        start_date = dates.min()
        end_date = dates.max()

        # Results storage
        all_predictions = []
        all_actuals = []
        all_edges = []
        all_market_probs = []

        # Walk forward - convert to pandas Timestamp for proper comparison
        current_date = pd.Timestamp(start_date) + timedelta(days=self.train_window_days)

        end_ts = pd.Timestamp(end_date)
        while current_date + timedelta(days=self.test_window_days) <= end_ts:
            # Training data
            train_start = current_date - timedelta(days=self.train_window_days)
            train_mask = (dates >= train_start) & (dates < current_date)
            train_data = data[train_mask]

            if len(train_data) < self.min_samples:
                current_date += timedelta(days=self.step_days)
                continue

            # Test data
            test_end = current_date + timedelta(days=self.test_window_days)
            test_mask = (dates >= current_date) & (dates < test_end)
            test_data = data[test_mask]

            if len(test_data) == 0:
                current_date += timedelta(days=self.step_days)
                continue

            # Train model - exclude market_home_prob to prevent data leakage
            # Keep home_score/away_score for Elo training (it uses them to update ratings)
            train_exclude = [target_col, date_col, "market_home_prob"]
            X_train = train_data.drop(columns=train_exclude, errors="ignore")
            y_train = train_data[target_col]
            self.model.train(X_train, y_train)

            # Test model - exclude scores from prediction features (no look-ahead)
            test_exclude = [target_col, date_col, "market_home_prob", "home_score", "away_score"]
            X_test = test_data.drop(columns=test_exclude, errors="ignore")
            y_test = test_data[target_col]

            proba = self.model.predict_proba(X_test)
            home_probs = proba[:, 1]

            all_predictions.extend(home_probs)
            all_actuals.extend(y_test.values)

            # Calculate edge if market probabilities available
            if "market_home_prob" in test_data.columns:
                market_probs = test_data["market_home_prob"].values
                edges = home_probs - market_probs
                all_edges.extend(edges)
                all_market_probs.extend(market_probs)

            current_date += timedelta(days=self.step_days)

        # Calculate metrics
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)

        result = self._calculate_metrics(
            predictions=predictions,
            actuals=actuals,
            edges=np.array(all_edges) if all_edges else None,
            market_probs=np.array(all_market_probs) if all_market_probs else None,
            min_edge=min_edge,
        )

        result.model_id = self.model.model_id
        result.start_date = start_date.to_pydatetime() if hasattr(start_date, 'to_pydatetime') else start_date
        result.end_date = end_date.to_pydatetime() if hasattr(end_date, 'to_pydatetime') else end_date

        return result

    def run_simple(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str = "home_win",
        min_edge: float = 0.03,
    ) -> BacktestResult:
        """Run simple train/test split backtest.

        Args:
            train_data: Training data.
            test_data: Test data.
            target_col: Name of target column.
            min_edge: Minimum edge to count as a bet.

        Returns:
            BacktestResult object.
        """
        logger.info(f"Running simple backtest for {self.model.model_id}")

        # Train - keep scores for Elo model
        train_exclude = [target_col, "market_home_prob"]
        X_train = train_data.drop(columns=train_exclude, errors="ignore")
        y_train = train_data[target_col]
        self.model.train(X_train, y_train)

        # Test - exclude scores from prediction (no look-ahead)
        test_exclude = [target_col, "market_home_prob", "home_score", "away_score"]
        X_test = test_data.drop(columns=test_exclude, errors="ignore")
        y_test = test_data[target_col]

        proba = self.model.predict_proba(X_test)
        home_probs = proba[:, 1]

        # Calculate edge if market probabilities available
        edges = None
        market_probs = None
        if "market_home_prob" in test_data.columns:
            market_probs = test_data["market_home_prob"].values
            edges = home_probs - market_probs

        result = self._calculate_metrics(
            predictions=home_probs,
            actuals=y_test.values,
            edges=edges,
            market_probs=market_probs,
            min_edge=min_edge,
        )

        result.model_id = self.model.model_id

        return result

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        edges: Optional[np.ndarray] = None,
        market_probs: Optional[np.ndarray] = None,
        min_edge: float = 0.03,
    ) -> BacktestResult:
        """Calculate backtest metrics.

        Args:
            predictions: Predicted home win probabilities.
            actuals: Actual outcomes (1 = home win).
            edges: Edge values (model prob - market prob).
            market_probs: Market implied probabilities.
            min_edge: Minimum edge for betting simulation.

        Returns:
            BacktestResult object.
        """
        from sklearn.metrics import brier_score_loss, log_loss

        n_predictions = len(predictions)
        predicted_outcomes = (predictions > 0.5).astype(int)
        n_correct = np.sum(predicted_outcomes == actuals)

        # Core metrics
        accuracy = n_correct / n_predictions if n_predictions > 0 else 0
        brier = brier_score_loss(actuals, predictions)
        logloss = log_loss(actuals, predictions)

        # Calibration error
        calibration_error = abs(np.mean(predictions) - np.mean(actuals))

        result = BacktestResult(
            model_id="",
            start_date=datetime.now(),
            end_date=datetime.now(),
            n_predictions=n_predictions,
            n_correct=n_correct,
            accuracy=accuracy,
            brier_score=brier,
            log_loss=logloss,
            calibration_error=calibration_error,
        )

        # Betting simulation
        if edges is not None and market_probs is not None:
            bet_mask = np.abs(edges) >= min_edge
            if np.any(bet_mask):
                bet_edges = edges[bet_mask]
                bet_actuals = actuals[bet_mask]
                bet_predictions = predictions[bet_mask]
                bet_market = market_probs[bet_mask]

                # Determine bet direction
                bet_on_home = bet_edges > 0
                wins = (bet_on_home & (bet_actuals == 1)) | (~bet_on_home & (bet_actuals == 0))

                result.total_bets = len(bet_edges)
                result.winning_bets = int(np.sum(wins))
                result.avg_edge = float(np.mean(np.abs(bet_edges)))

                # Simple ROI calculation (assuming even odds for simplicity)
                result.roi = (
                    (result.winning_bets / result.total_bets - 0.5) * 2
                    if result.total_bets > 0
                    else 0
                )

                # CLV calculation
                clv_values = []
                for i in range(len(bet_market)):
                    if bet_on_home[i]:
                        clv = bet_market[i] - bet_predictions[i]
                    else:
                        clv = (1 - bet_market[i]) - (1 - bet_predictions[i])
                    clv_values.append(clv)

                result.avg_clv = float(np.mean(clv_values)) if clv_values else None

        return result

    def calibration_plot_data(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate data for calibration plot.

        Args:
            predictions: Predicted probabilities.
            actuals: Actual outcomes.
            n_bins: Number of bins.

        Returns:
            Tuple of (bin_centers, mean_predictions, mean_actuals).
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        mean_preds = []
        mean_acts = []

        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if np.any(mask):
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                mean_preds.append(np.mean(predictions[mask]))
                mean_acts.append(np.mean(actuals[mask]))

        return np.array(bin_centers), np.array(mean_preds), np.array(mean_acts)
