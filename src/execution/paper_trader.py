"""Paper trading simulation for testing strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.utils.config import get_risk_config
from src.data.storage import DatabaseStorage
from src.market.edge_calculator import Opportunity


@dataclass
class PaperPosition:
    """A paper trading position."""

    position_id: int
    game_id: int
    market_type: str
    side: str
    entry_time: datetime
    stake: float
    odds_at_entry: float
    model_prob: float
    market_prob: float
    edge: float
    kelly_fraction: float

    # Settlement
    status: str = "open"  # open, won, lost, void
    exit_time: Optional[datetime] = None
    closing_odds: Optional[float] = None
    settlement_price: Optional[float] = None
    pnl: Optional[float] = None
    clv: Optional[float] = None

    def to_dict(self, Optional) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position_id": self.position_id,
            "game_id": self.game_id,
            "market_type": self.market_type,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
            "stake": self.stake,
            "odds_at_entry": self.odds_at_entry,
            "model_prob": self.model_prob,
            "market_prob": self.market_prob,
            "edge": self.edge,
            "kelly_fraction": self.kelly_fraction,
            "status": self.status,
            "pnl": self.pnl,
            "clv": self.clv,
        }


class PaperTrader:
    """Paper trading engine for strategy simulation."""

    def __init__(
        self,
        initial_bankroll: Optional[float] = None,
        storage: Optional[DatabaseStorage] = None,
    ) -> None:
        """Initialize paper trader.

        Args:
            initial_bankroll: Starting bankroll (uses config if None).
            storage: Database storage for persistence.
        """
        config = get_risk_config()
        self.bankroll = initial_bankroll or config.initial_bankroll
        self.initial_bankroll = self.bankroll
        self.storage = storage

        self._positions: List[PaperPosition] = []
        self._next_position_id = 1
        self._trade_history: List[Dict[str, Any]] = []

    @property
    def open_positions(self) -> List[PaperPosition]:
        """Get all open positions."""
        return [p for p in self._positions if p.status == "open"]

    @property
    def total_exposure(self) -> float:
        """Get total exposure from open positions."""
        return sum(p.stake for p in self.open_positions)

    @property
    def exposure_pct(self) -> float:
        """Get exposure as percentage of bankroll."""
        if self.bankroll == 0:
            return 1.0
        return self.total_exposure / self.bankroll

    @property
    def total_pnl(self) -> float:
        """Get total profit/loss."""
        return self.bankroll - self.initial_bankroll

    @property
    def roi(self) -> float:
        """Get return on investment."""
        if self.initial_bankroll == 0:
            return 0
        return self.total_pnl / self.initial_bankroll

    def open_position(
        self,
        opportunity: Opportunity,
        stake: Optional[float] = None,
    ) -> Optional[PaperPosition]:
        """Open a new paper position.

        Args:
            opportunity: Opportunity to bet on.
            stake: Override stake amount.

        Returns:
            PaperPosition if opened, None if rejected.
        """
        # Check exposure limits
        config = get_risk_config()
        if self.exposure_pct >= config.max_total_exposure:
            logger.warning("Cannot open position: exposure limit reached")
            return None

        # Calculate stake
        if stake is None:
            stake = opportunity.recommended_stake_pct * self.bankroll

        # Ensure minimum stake
        if stake < config.min_bet_pct * self.bankroll:
            logger.debug(f"Stake {stake:.2f} below minimum, skipping")
            return None

        # Ensure we have enough bankroll
        if stake > self.bankroll - self.total_exposure:
            stake = self.bankroll - self.total_exposure
            if stake <= 0:
                return None

        # Calculate decimal odds from market prob
        decimal_odds = 1 / opportunity.market_prob if opportunity.market_prob > 0 else 2.0

        position = PaperPosition(
            position_id=self._next_position_id,
            game_id=opportunity.game_id,
            market_type=opportunity.market_type,
            side=opportunity.side,
            entry_time=datetime.now(timezone.utc),
            stake=stake,
            odds_at_entry=decimal_odds,
            model_prob=opportunity.model_prob,
            market_prob=opportunity.market_prob,
            edge=opportunity.edge,
            kelly_fraction=opportunity.kelly_bet,
        )

        self._positions.append(position)
        self._next_position_id += 1

        logger.info(
            f"Opened position #{position.position_id}: "
            f"{position.side} ${stake:.2f} @ {decimal_odds:.2f}"
        )

        # Store in database if available
        if self.storage:
            self._store_position(position)

        return position

    def settle_position(
        self,
        position_id: int,
        won: bool,
        closing_odds: Optional[float] = None,
    ) -> Optional[PaperPosition]:
        """Settle a position.

        Args:
            position_id: Position ID to settle.
            won: Whether the bet won.
            closing_odds: Closing decimal odds.

        Returns:
            Settled position or None if not found.
        """
        position = next(
            (p for p in self._positions if p.position_id == position_id),
            None,
        )

        if position is None:
            logger.warning(f"Position {position_id} not found")
            return None

        if position.status != "open":
            logger.warning(f"Position {position_id} already settled")
            return None

        # Calculate PnL
        if won:
            position.status = "won"
            position.pnl = position.stake * (position.odds_at_entry - 1)
        else:
            position.status = "lost"
            position.pnl = -position.stake

        position.exit_time = datetime.now(timezone.utc)
        position.closing_odds = closing_odds

        # Calculate CLV if we have closing odds
        if closing_odds:
            closing_prob = 1 / closing_odds
            position.clv = closing_prob - position.market_prob

        # Update bankroll
        self.bankroll += position.pnl

        # Log the trade
        self._trade_history.append(
            {
                "position_id": position.position_id,
                "settled_at": position.exit_time,
                "pnl": position.pnl,
                "bankroll_after": self.bankroll,
            }
        )

        logger.info(
            f"Settled position #{position_id}: "
            f"{'WON' if won else 'LOST'} PnL=${position.pnl:.2f} "
            f"Bankroll=${self.bankroll:.2f}"
        )

        # Update database if available
        if self.storage:
            self._update_position(position)

        return position

    def settle_by_game(
        self,
        game_id: int,
        home_won: bool,
    ) -> List[PaperPosition]:
        """Settle all positions for a game.

        Args:
            game_id: Game ID.
            home_won: Whether home team won.

        Returns:
            List of settled positions.
        """
        settled = []

        for position in self.open_positions:
            if position.game_id != game_id:
                continue

            won = (position.side == "home" and home_won) or (
                position.side == "away" and not home_won
            )

            result = self.settle_position(position.position_id, won)
            if result:
                settled.append(result)

        return settled

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics.

        Returns:
            Dictionary of statistics.
        """
        settled = [p for p in self._positions if p.status in ("won", "lost")]

        if not settled:
            return {
                "total_positions": len(self._positions),
                "open_positions": len(self.open_positions),
                "settled_positions": 0,
                "bankroll": self.bankroll,
                "total_pnl": self.total_pnl,
                "roi": self.roi,
            }

        wins = [p for p in settled if p.status == "won"]
        win_rate = len(wins) / len(settled)

        pnls = [p.pnl for p in settled if p.pnl is not None]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0

        clvs = [p.clv for p in settled if p.clv is not None]
        avg_clv = sum(clvs) / len(clvs) if clvs else 0

        edges = [p.edge for p in settled]
        avg_edge = sum(edges) / len(edges) if edges else 0

        return {
            "total_positions": len(self._positions),
            "open_positions": len(self.open_positions),
            "settled_positions": len(settled),
            "wins": len(wins),
            "losses": len(settled) - len(wins),
            "win_rate": win_rate,
            "bankroll": self.bankroll,
            "total_pnl": self.total_pnl,
            "roi": self.roi,
            "avg_pnl": avg_pnl,
            "avg_edge": avg_edge,
            "avg_clv": avg_clv,
        }

    def _store_position(self, position: PaperPosition) -> None:
        """Store position in database."""
        if not self.storage:
            return

        bet_data = {
            "game_id": position.game_id,
            "prediction_id": None,
            "kalshi_market_id": None,
            "bet_type": "paper",
            "market_type": position.market_type,
            "side": position.side,
            "placed_at": position.entry_time,
            "stake": position.stake,
            "odds_at_placement": position.odds_at_entry,
            "expected_value": position.edge * position.stake,
            "edge": position.edge,
            "kelly_fraction": position.kelly_fraction,
            "status": "open",
            "notes": None,
        }

        self.storage.insert_bet(bet_data)

    def _update_position(self, position: PaperPosition) -> None:
        """Update position in database."""
        if not self.storage or position.pnl is None:
            return

        # Would update via storage.update_bet_settlement()
        pass

    def reset(self, initial_bankroll: Optional[float] = None) -> None:
        """Reset paper trader.

        Args:
            initial_bankroll: New starting bankroll.
        """
        config = get_risk_config()
        self.bankroll = initial_bankroll or config.initial_bankroll
        self.initial_bankroll = self.bankroll
        self._positions.clear()
        self._trade_history.clear()
        self._next_position_id = 1

        logger.info(f"Paper trader reset with ${self.bankroll:.2f}")
