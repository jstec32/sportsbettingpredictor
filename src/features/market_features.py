"""Market-derived features like line movement and consensus odds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class LineMovement:
    """Line movement data for a game."""

    game_id: int
    opening_line: float
    current_line: float
    movement: float
    movement_direction: str  # 'toward_home', 'toward_away', 'stable'
    steam_move: bool
    reverse_line_movement: bool


@dataclass
class MarketConsensus:
    """Market consensus data."""

    game_id: int
    consensus_home_prob: float
    consensus_spread: float
    consensus_total: float
    sharp_money_side: Optional[str]
    public_money_side: Optional[str]


class MarketFeatures:
    """Calculate market-derived features."""

    def __init__(self) -> None:
        """Initialize market features calculator."""
        pass

    def calculate_implied_probability(
        self,
        american_odds: float,
    ) -> float:
        """Convert American odds to implied probability.

        Args:
            american_odds: American-format odds.

        Returns:
            Implied probability (0 to 1).
        """
        if american_odds > 0:
            return 100.0 / (american_odds + 100.0)
        else:
            return abs(american_odds) / (abs(american_odds) + 100.0)

    def remove_vig(
        self,
        home_prob: float,
        away_prob: float,
    ) -> Tuple[float, float]:
        """Remove vig/juice from implied probabilities.

        Args:
            home_prob: Home implied probability.
            away_prob: Away implied probability.

        Returns:
            Tuple of (true_home_prob, true_away_prob).
        """
        total = home_prob + away_prob
        if total == 0:
            return 0.5, 0.5

        return home_prob / total, away_prob / total

    def spread_to_probability(
        self,
        spread: float,
        home_advantage: float = 3.0,
    ) -> float:
        """Convert point spread to win probability.

        Uses a simplified model where each point of spread
        corresponds to roughly 2.5-3% win probability change.

        Args:
            spread: Point spread (negative = home favored).
            home_advantage: Built-in home advantage in points.

        Returns:
            Home team win probability.
        """
        # Each point is worth ~2.7% in probability
        adjusted_spread = spread + home_advantage
        prob = 0.5 + (adjusted_spread * 0.027)
        return max(0.02, min(0.98, prob))

    def calculate_line_movement(
        self,
        odds_history: pd.DataFrame,
        game_id: int,
    ) -> Optional[LineMovement]:
        """Calculate line movement for a game.

        Args:
            odds_history: DataFrame of odds history.
            game_id: Game ID.

        Returns:
            LineMovement object or None.
        """
        game_odds = odds_history[odds_history["game_id"] == game_id]

        if len(game_odds) < 2:
            return None

        game_odds = game_odds.sort_values("captured_at")

        opening = game_odds.iloc[0]
        current = game_odds.iloc[-1]

        opening_spread = opening.get("spread_home", 0) or 0
        current_spread = current.get("spread_home", 0) or 0

        movement = current_spread - opening_spread

        # Determine direction
        if abs(movement) < 0.5:
            direction = "stable"
        elif movement < 0:
            direction = "toward_home"  # Home becoming more favored
        else:
            direction = "toward_away"

        # Detect steam move (sharp sudden movement)
        steam_move = False
        for i in range(1, len(game_odds)):
            time_diff = game_odds.iloc[i]["captured_at"] - game_odds.iloc[i - 1]["captured_at"]
            spread_diff = abs(
                (game_odds.iloc[i].get("spread_home", 0) or 0)
                - (game_odds.iloc[i - 1].get("spread_home", 0) or 0)
            )
            if isinstance(time_diff, timedelta) and time_diff < timedelta(hours=2):
                if spread_diff >= 1.0:
                    steam_move = True
                    break

        # Reverse line movement detection would require betting percentages
        # which we don't have from The Odds API
        reverse_line = False

        return LineMovement(
            game_id=game_id,
            opening_line=opening_spread,
            current_line=current_spread,
            movement=movement,
            movement_direction=direction,
            steam_move=steam_move,
            reverse_line_movement=reverse_line,
        )

    def calculate_consensus(
        self,
        bookmaker_odds: List[Dict[str, Any]],
        game_id: int,
    ) -> MarketConsensus:
        """Calculate consensus from multiple bookmakers.

        Args:
            bookmaker_odds: List of bookmaker odds data.
            game_id: Game ID.

        Returns:
            MarketConsensus object.
        """
        home_probs = []
        spreads = []
        totals = []

        for odds in bookmaker_odds:
            if odds.get("home_prob"):
                home_probs.append(odds["home_prob"])
            if odds.get("spread_home"):
                spreads.append(odds["spread_home"])
            if odds.get("total_line"):
                totals.append(odds["total_line"])

        return MarketConsensus(
            game_id=game_id,
            consensus_home_prob=sum(home_probs) / len(home_probs) if home_probs else 0.5,
            consensus_spread=sum(spreads) / len(spreads) if spreads else 0.0,
            consensus_total=sum(totals) / len(totals) if totals else 0.0,
            sharp_money_side=None,  # Would need betting flow data
            public_money_side=None,
        )

    def build_market_features(
        self,
        line_movement: Optional[LineMovement],
        consensus: Optional[MarketConsensus],
        kalshi_yes_price: Optional[float] = None,
    ) -> Dict[str, float]:
        """Build market feature dictionary.

        Args:
            line_movement: Line movement data.
            consensus: Market consensus data.
            kalshi_yes_price: Kalshi market yes price.

        Returns:
            Dictionary of market features.
        """
        features: Dict[str, float] = {}

        # Line movement features
        if line_movement:
            features["opening_spread"] = line_movement.opening_line
            features["current_spread"] = line_movement.current_line
            features["spread_movement"] = line_movement.movement
            features["spread_movement_abs"] = abs(line_movement.movement)
            features["toward_home"] = (
                1.0 if line_movement.movement_direction == "toward_home" else 0.0
            )
            features["toward_away"] = (
                1.0 if line_movement.movement_direction == "toward_away" else 0.0
            )
            features["steam_move"] = 1.0 if line_movement.steam_move else 0.0
            features["reverse_line"] = 1.0 if line_movement.reverse_line_movement else 0.0
        else:
            features["opening_spread"] = 0.0
            features["current_spread"] = 0.0
            features["spread_movement"] = 0.0
            features["spread_movement_abs"] = 0.0
            features["toward_home"] = 0.0
            features["toward_away"] = 0.0
            features["steam_move"] = 0.0
            features["reverse_line"] = 0.0

        # Consensus features
        if consensus:
            features["consensus_home_prob"] = consensus.consensus_home_prob
            features["consensus_spread"] = consensus.consensus_spread
            features["consensus_total"] = consensus.consensus_total
        else:
            features["consensus_home_prob"] = 0.5
            features["consensus_spread"] = 0.0
            features["consensus_total"] = 0.0

        # Kalshi features
        if kalshi_yes_price is not None:
            features["kalshi_home_prob"] = kalshi_yes_price
            # Market disagreement
            if consensus:
                features["kalshi_vs_consensus"] = kalshi_yes_price - consensus.consensus_home_prob
            else:
                features["kalshi_vs_consensus"] = 0.0
        else:
            features["kalshi_home_prob"] = 0.5
            features["kalshi_vs_consensus"] = 0.0

        return features

    def detect_sharp_action(
        self,
        line_movement: LineMovement,
        public_betting_pct: Optional[float] = None,
    ) -> bool:
        """Detect potential sharp betting action.

        Sharp action indicators:
        1. Steam moves (quick line moves)
        2. Reverse line movement (line moves opposite public betting)
        3. Late money moves

        Args:
            line_movement: Line movement data.
            public_betting_pct: Public betting percentage (if available).

        Returns:
            True if sharp action detected.
        """
        # Steam move is a strong indicator
        if line_movement.steam_move:
            return True

        # Reverse line movement
        if line_movement.reverse_line_movement:
            return True

        # Significant movement
        if abs(line_movement.movement) >= 2.0:
            return True

        return False

    def calculate_closing_line_value(
        self,
        bet_odds: float,
        closing_odds: float,
    ) -> float:
        """Calculate closing line value (CLV).

        CLV measures if you got a better price than the closing line.
        Positive CLV indicates you beat the market.

        Args:
            bet_odds: Odds at time of bet (implied probability).
            closing_odds: Closing odds (implied probability).

        Returns:
            CLV as percentage.
        """
        if closing_odds == 0:
            return 0.0

        # CLV = (closing_prob - bet_prob) / bet_prob
        return (closing_odds - bet_odds) / bet_odds
