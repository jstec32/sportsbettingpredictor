"""Edge calculation for identifying profitable betting opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from src.models.base import Prediction
from src.utils.config import get_risk_config


@dataclass
class Opportunity:
    """A betting opportunity with positive expected value."""

    game_id: int
    market_type: str
    side: str  # 'home', 'away', 'over', 'under'
    model_prob: float
    market_prob: float
    edge: float
    confidence: float
    expected_value: float
    kelly_bet: float
    recommended_stake_pct: float
    prediction: Optional[Prediction] = None
    game_datetime: Optional[datetime] = None
    home_team: str = ""
    away_team: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "market_type": self.market_type,
            "side": self.side,
            "model_prob": self.model_prob,
            "market_prob": self.market_prob,
            "edge": self.edge,
            "confidence": self.confidence,
            "expected_value": self.expected_value,
            "kelly_bet": self.kelly_bet,
            "recommended_stake_pct": self.recommended_stake_pct,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "game_datetime": self.game_datetime.isoformat() if self.game_datetime else None,
        }


class EdgeCalculator:
    """Calculate betting edges from model predictions vs market odds."""

    def __init__(
        self,
        min_edge: Optional[float] = None,
        min_confidence: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
    ) -> None:
        """Initialize edge calculator.

        Args:
            min_edge: Minimum edge threshold to consider.
            min_confidence: Minimum confidence threshold.
            kelly_fraction: Fraction of Kelly to use for sizing.
        """
        config = get_risk_config()

        self.min_edge = min_edge or config.min_edge_threshold
        # Confidence = abs(home_prob - 0.5) * 2, so 0.08 means model predicts at least 54%/46%
        self.min_confidence = min_confidence or 0.08
        self.kelly_fraction = kelly_fraction or config.kelly_fraction

    def calculate_edge(
        self,
        model_prob: float,
        market_prob: float,
    ) -> float:
        """Calculate edge as the difference between model and market probability.

        Args:
            model_prob: Model's predicted probability.
            market_prob: Market implied probability.

        Returns:
            Edge value (positive = model sees value).
        """
        return float(model_prob) - float(market_prob)

    def calculate_expected_value(
        self,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """Calculate expected value of a bet.

        Args:
            model_prob: Model's predicted probability.
            decimal_odds: Decimal odds offered.

        Returns:
            Expected value per unit bet.
        """
        # Convert to float to handle Decimal types from database
        model_prob = float(model_prob)
        decimal_odds = float(decimal_odds)

        # EV = (prob * payout) - (1 - prob) * stake
        # For $1 bet at decimal odds d: EV = (p * d) - 1
        return (model_prob * decimal_odds) - 1

    def calculate_kelly(
        self,
        prob: float,
        decimal_odds: float,
    ) -> float:
        """Calculate Kelly criterion bet fraction.

        Args:
            prob: Probability of winning.
            decimal_odds: Decimal odds.

        Returns:
            Optimal bet fraction of bankroll.
        """
        # Convert to float to handle Decimal types from database
        prob = float(prob)
        decimal_odds = float(decimal_odds)

        if decimal_odds <= 1:
            return 0.0

        # Kelly formula: f = (p * odds - 1) / (odds - 1)
        # Equivalent to: f = (p * (d - 1) - q) / (d - 1) where q = 1 - p
        q = 1 - prob
        b = decimal_odds - 1  # Net odds (what you win per dollar bet)

        if b <= 0:
            return 0.0

        kelly = (prob * b - q) / b

        # Return 0 if negative (no edge)
        return max(0.0, kelly)

    def find_opportunities(
        self,
        predictions: List[Prediction],
        markets: List[Dict[str, Any]],
    ) -> List[Opportunity]:
        """Find betting opportunities from predictions and market data.

        Args:
            predictions: List of model predictions.
            markets: List of market data dictionaries.

        Returns:
            List of Opportunity objects sorted by edge.
        """
        opportunities = []

        # Create lookup by game_id
        market_lookup: Dict[int, Dict[str, Any]] = {}
        for market in markets:
            game_id = market.get("game_id")
            if game_id:
                market_lookup[game_id] = market

        for pred in predictions:
            market = market_lookup.get(pred.game_id)
            if not market:
                continue

            # Check home team opportunity
            home_market_prob = market.get("home_prob", 0.5)
            if home_market_prob > 0 and home_market_prob < 1:
                home_edge = self.calculate_edge(pred.home_win_prob, home_market_prob)

                if home_edge >= self.min_edge and pred.confidence >= self.min_confidence:
                    decimal_odds = 1 / home_market_prob
                    kelly = self.calculate_kelly(pred.home_win_prob, decimal_odds)
                    ev = self.calculate_expected_value(pred.home_win_prob, decimal_odds)

                    opportunities.append(
                        Opportunity(
                            game_id=pred.game_id,
                            market_type="moneyline",
                            side="home",
                            model_prob=pred.home_win_prob,
                            market_prob=home_market_prob,
                            edge=home_edge,
                            confidence=pred.confidence,
                            expected_value=ev,
                            kelly_bet=kelly,
                            recommended_stake_pct=kelly * self.kelly_fraction,
                            prediction=pred,
                            game_datetime=market.get("game_datetime"),
                            home_team=market.get("home_team", ""),
                            away_team=market.get("away_team", ""),
                        )
                    )

            # Check away team opportunity
            away_market_prob = market.get("away_prob", 0.5)
            if away_market_prob > 0 and away_market_prob < 1:
                away_edge = self.calculate_edge(pred.away_win_prob, away_market_prob)

                if away_edge >= self.min_edge and pred.confidence >= self.min_confidence:
                    decimal_odds = 1 / away_market_prob
                    kelly = self.calculate_kelly(pred.away_win_prob, decimal_odds)
                    ev = self.calculate_expected_value(pred.away_win_prob, decimal_odds)

                    opportunities.append(
                        Opportunity(
                            game_id=pred.game_id,
                            market_type="moneyline",
                            side="away",
                            model_prob=pred.away_win_prob,
                            market_prob=away_market_prob,
                            edge=away_edge,
                            confidence=pred.confidence,
                            expected_value=ev,
                            kelly_bet=kelly,
                            recommended_stake_pct=kelly * self.kelly_fraction,
                            prediction=pred,
                            game_datetime=market.get("game_datetime"),
                            home_team=market.get("home_team", ""),
                            away_team=market.get("away_team", ""),
                        )
                    )

        # Sort by edge (highest first)
        opportunities.sort(key=lambda x: x.edge, reverse=True)

        logger.info(f"Found {len(opportunities)} opportunities from {len(predictions)} predictions")

        return opportunities

    def filter_opportunities(
        self,
        opportunities: List[Opportunity],
        max_opportunities: Optional[int] = None,
        max_per_game: int = 1,
        min_ev: float = 0,
    ) -> List[Opportunity]:
        """Filter and limit opportunities.

        Args:
            opportunities: List of opportunities.
            max_opportunities: Maximum number to return.
            max_per_game: Maximum opportunities per game.
            min_ev: Minimum expected value.

        Returns:
            Filtered list of opportunities.
        """
        # Filter by minimum EV
        filtered = [o for o in opportunities if o.expected_value >= min_ev]

        # Limit per game
        game_counts: Dict[int, int] = {}
        limited = []
        for opp in filtered:
            count = game_counts.get(opp.game_id, 0)
            if count < max_per_game:
                limited.append(opp)
                game_counts[opp.game_id] = count + 1

        # Limit total
        if max_opportunities:
            limited = limited[:max_opportunities]

        return limited

    def rank_opportunities(
        self,
        opportunities: List[Opportunity],
        weights: Dict[str, float] | None = None,
    ) -> List[Opportunity]:
        """Rank opportunities by a weighted score.

        Args:
            opportunities: List of opportunities.
            weights: Weights for different factors.

        Returns:
            Opportunities sorted by score.
        """
        if weights is None:
            weights = {
                "edge": 0.4,
                "confidence": 0.3,
                "expected_value": 0.2,
                "kelly": 0.1,
            }

        def score(opp: Opportunity) -> float:
            return (
                weights.get("edge", 0) * opp.edge
                + weights.get("confidence", 0) * opp.confidence
                + weights.get("expected_value", 0) * opp.expected_value
                + weights.get("kelly", 0) * opp.kelly_bet
            )

        return sorted(opportunities, key=score, reverse=True)
