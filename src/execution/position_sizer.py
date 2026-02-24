"""Position sizing using Kelly criterion and risk management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.utils.config import get_risk_config


@dataclass
class PositionSize:
    """Calculated position size."""

    raw_kelly: float
    fractional_kelly: float
    position_pct: float
    position_size: float
    capped: bool = False
    cap_reason: Optional[str] = None


class PositionSizer:
    """Calculate optimal position sizes using Kelly criterion."""

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize position sizer.

        Args:
            config: Optional configuration override.
        """
        self._config = config or get_risk_config()

    @property
    def kelly_fraction(self) -> float:
        """Fraction of Kelly to use."""
        return self._config.kelly_fraction

    @property
    def max_bet_pct(self) -> float:
        """Maximum bet as percentage of bankroll."""
        return self._config.max_bet_pct

    @property
    def min_bet_pct(self) -> float:
        """Minimum bet as percentage of bankroll."""
        return self._config.min_bet_pct

    @property
    def min_edge(self) -> float:
        """Minimum edge required to bet."""
        return self._config.min_edge_threshold

    def kelly_criterion(
        self,
        prob: float,
        odds: float,
        fraction: Optional[float] = None,
    ) -> float:
        """Calculate Kelly criterion bet fraction.

        Args:
            prob: Probability of winning.
            odds: Decimal odds.
            fraction: Kelly fraction to use (default: configured).

        Returns:
            Optimal bet fraction of bankroll.
        """
        if odds <= 1:
            return 0

        # Kelly formula: f = (p * b - q) / b
        # where b = decimal_odds - 1, q = 1 - p
        b = odds - 1
        q = 1 - prob

        if b <= 0:
            return 0

        raw_kelly = (prob * b - q) / b

        if raw_kelly <= 0:
            return 0

        # Apply fractional Kelly
        frac = fraction or self.kelly_fraction
        return raw_kelly * frac

    def size_position(
        self,
        edge: float,
        confidence: float,
        bankroll: float,
        decimal_odds: float,
        current_exposure: float = 0,
    ) -> PositionSize:
        """Calculate position size considering all constraints.

        Args:
            edge: Model edge (model_prob - market_prob).
            confidence: Model confidence (0-1).
            bankroll: Current bankroll.
            decimal_odds: Decimal odds offered.
            current_exposure: Current total exposure.

        Returns:
            PositionSize object.
        """
        # Check minimum edge
        if edge < self.min_edge:
            return PositionSize(
                raw_kelly=0,
                fractional_kelly=0,
                position_pct=0,
                position_size=0,
                capped=True,
                cap_reason=f"Edge {edge:.2%} below minimum {self.min_edge:.2%}",
            )

        # Estimate probability from edge and market odds
        market_prob = 1 / decimal_odds
        model_prob = market_prob + edge

        # Ensure probability is valid
        model_prob = max(0.01, min(0.99, model_prob))

        # Calculate raw Kelly
        raw_kelly = self.kelly_criterion(model_prob, decimal_odds, fraction=1.0)

        # Apply fractional Kelly
        fractional_kelly = raw_kelly * self.kelly_fraction

        # Adjust for confidence
        position_pct = fractional_kelly * confidence

        # Apply caps
        capped = False
        cap_reason = None

        # Maximum single bet cap
        if position_pct > self.max_bet_pct:
            position_pct = self.max_bet_pct
            capped = True
            cap_reason = f"Capped at max bet {self.max_bet_pct:.1%}"

        # Minimum bet threshold
        if 0 < position_pct < self.min_bet_pct:
            position_pct = 0
            capped = True
            cap_reason = f"Below minimum bet {self.min_bet_pct:.1%}"

        # Check exposure limits
        max_exposure = self._config.max_total_exposure
        available_exposure = max_exposure - current_exposure / bankroll

        if position_pct > available_exposure:
            position_pct = max(0, available_exposure)
            capped = True
            cap_reason = f"Exposure limit reached ({max_exposure:.0%})"

        position_size = position_pct * bankroll

        return PositionSize(
            raw_kelly=raw_kelly,
            fractional_kelly=fractional_kelly,
            position_pct=position_pct,
            position_size=position_size,
            capped=capped,
            cap_reason=cap_reason,
        )

    def size_portfolio(
        self,
        opportunities: List[Dict[str, Any]],
        bankroll: float,
        max_positions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Size multiple positions respecting portfolio constraints.

        Args:
            opportunities: List of opportunity dicts with edge, confidence, odds.
            bankroll: Current bankroll.
            max_positions: Maximum number of positions.

        Returns:
            List of sized positions.
        """
        if max_positions is None:
            max_positions = self._config.max_daily_bets

        # Sort by edge
        sorted_opps = sorted(opportunities, key=lambda x: x.get("edge", 0), reverse=True)

        positions = []
        current_exposure = 0

        for opp in sorted_opps[:max_positions]:
            size = self.size_position(
                edge=opp.get("edge", 0),
                confidence=opp.get("confidence", 0.5),
                bankroll=bankroll,
                decimal_odds=opp.get("decimal_odds", 2.0),
                current_exposure=current_exposure,
            )

            if size.position_size > 0:
                opp_with_size = opp.copy()
                opp_with_size["position_size"] = size.position_size
                opp_with_size["position_pct"] = size.position_pct
                opp_with_size["kelly"] = size.fractional_kelly
                opp_with_size["capped"] = size.capped
                opp_with_size["cap_reason"] = size.cap_reason

                positions.append(opp_with_size)
                current_exposure += size.position_size

        return positions

    def calculate_bet_amount(
        self,
        kelly_pct: float,
        bankroll: float,
        confidence: float = 1.0,
    ) -> float:
        """Simple bet amount calculation.

        Args:
            kelly_pct: Kelly fraction to bet.
            bankroll: Current bankroll.
            confidence: Confidence multiplier.

        Returns:
            Bet amount.
        """
        bet_pct = kelly_pct * confidence

        # Apply caps
        bet_pct = min(bet_pct, self.max_bet_pct)

        if bet_pct < self.min_bet_pct:
            return 0

        return bet_pct * bankroll

    def expected_growth_rate(
        self,
        prob: float,
        odds: float,
        bet_fraction: float,
    ) -> float:
        """Calculate expected growth rate (log utility).

        The Kelly criterion maximizes this value.

        Args:
            prob: Win probability.
            odds: Decimal odds.
            bet_fraction: Fraction of bankroll to bet.

        Returns:
            Expected log growth rate.
        """
        if bet_fraction <= 0 or bet_fraction >= 1:
            return 0

        b = odds - 1  # Net odds
        q = 1 - prob

        # E[log(growth)] = p * log(1 + f*b) + q * log(1 - f)
        import math

        win_term = prob * math.log(1 + bet_fraction * b)
        lose_term = q * math.log(1 - bet_fraction)

        return win_term + lose_term
