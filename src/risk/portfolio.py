"""Portfolio tracking and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger

from src.utils.config import get_risk_config
from src.data.storage import DatabaseStorage


@dataclass
class PortfolioSnapshot:
    """A snapshot of portfolio state."""

    snapshot_time: datetime
    bankroll: float
    total_exposure: float
    open_positions: int
    daily_pnl: float
    weekly_pnl: float
    total_pnl: float
    win_rate: Optional[float] = None
    avg_edge: Optional[float] = None
    avg_clv: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "snapshot_time": self.snapshot_time,
            "bankroll": self.bankroll,
            "total_exposure": self.total_exposure,
            "open_positions": self.open_positions,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "avg_edge": self.avg_edge,
            "avg_clv": self.avg_clv,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
        }


class Portfolio:
    """Portfolio tracking and performance analysis."""

    def __init__(
        self,
        initial_bankroll: Optional[float] = None,
        storage: Optional[DatabaseStorage] = None,
    ) -> None:
        """Initialize portfolio.

        Args:
            initial_bankroll: Starting bankroll.
            storage: Database storage for persistence.
        """
        config = get_risk_config()
        self.initial_bankroll = initial_bankroll or config.initial_bankroll
        self.bankroll = self.initial_bankroll
        self.storage = storage

        self._pnl_history: List[Tuple[datetime, float]] = []
        self._snapshots: List[PortfolioSnapshot] = []
        self._peak_bankroll = self.initial_bankroll

    @property
    def total_pnl(self) -> float:
        """Total profit/loss."""
        return self.bankroll - self.initial_bankroll

    @property
    def roi(self) -> float:
        """Return on investment."""
        if self.initial_bankroll == 0:
            return 0
        return self.total_pnl / self.initial_bankroll

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self._peak_bankroll == 0:
            return 0
        return (self._peak_bankroll - self.bankroll) / self._peak_bankroll

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown in history."""
        if not self._pnl_history:
            return 0

        peak = self.initial_bankroll
        max_dd = 0

        for _, bankroll in self._pnl_history:
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def record_pnl(self, pnl: float, timestamp: Optional[datetime] = None) -> None:
        """Record a PnL event.

        Args:
            pnl: Profit/loss amount.
            timestamp: Event time.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.bankroll += pnl
        self._pnl_history.append((timestamp, self.bankroll))

        # Update peak
        if self.bankroll > self._peak_bankroll:
            self._peak_bankroll = self.bankroll

        logger.debug(f"PnL recorded: {pnl:+.2f}, Bankroll: {self.bankroll:.2f}")

    def take_snapshot(
        self,
        open_positions: int = 0,
        total_exposure: float = 0,
        win_rate: Optional[float] = None,
        avg_edge: Optional[float] = None,
        avg_clv: Optional[float] = None,
    ) -> PortfolioSnapshot:
        """Take a portfolio snapshot.

        Args:
            open_positions: Number of open positions.
            total_exposure: Total exposure amount.
            win_rate: Win rate if known.
            avg_edge: Average edge if known.
            avg_clv: Average CLV if known.

        Returns:
            PortfolioSnapshot object.
        """
        now = datetime.now(timezone.utc)

        # Calculate daily/weekly PnL
        daily_pnl = self._calculate_period_pnl(days=1)
        weekly_pnl = self._calculate_period_pnl(days=7)

        # Calculate Sharpe ratio
        sharpe = self._calculate_sharpe()

        snapshot = PortfolioSnapshot(
            snapshot_time=now,
            bankroll=self.bankroll,
            total_exposure=total_exposure,
            open_positions=open_positions,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            total_pnl=self.total_pnl,
            win_rate=win_rate,
            avg_edge=avg_edge,
            avg_clv=avg_clv,
            sharpe_ratio=sharpe,
            max_drawdown=self.max_drawdown,
        )

        self._snapshots.append(snapshot)

        # Store in database
        if self.storage:
            self.storage.insert_portfolio_snapshot(snapshot.to_dict())

        return snapshot

    def _calculate_period_pnl(self, days: int) -> float:
        """Calculate PnL over a period.

        Args:
            days: Number of days to look back.

        Returns:
            PnL over the period.
        """
        if not self._pnl_history:
            return 0

        now = datetime.now(timezone.utc)
        cutoff = now.replace(tzinfo=timezone.utc) - __import__("datetime").timedelta(days=days)

        # Find bankroll at start of period
        start_bankroll = self.initial_bankroll
        for timestamp, bankroll in self._pnl_history:
            if timestamp.replace(tzinfo=timezone.utc) <= cutoff:
                start_bankroll = bankroll
            else:
                break

        return self.bankroll - start_bankroll

    def _calculate_sharpe(self, annualize: bool = True) -> Optional[float]:
        """Calculate Sharpe ratio.

        Args:
            annualize: Whether to annualize the ratio.

        Returns:
            Sharpe ratio or None if insufficient data.
        """
        if len(self._pnl_history) < 10:
            return None

        # Calculate daily returns
        returns = []
        for i in range(1, len(self._pnl_history)):
            prev_bankroll = self._pnl_history[i - 1][1]
            curr_bankroll = self._pnl_history[i][1]
            if prev_bankroll > 0:
                ret = (curr_bankroll - prev_bankroll) / prev_bankroll
                returns.append(ret)

        if not returns:
            return None

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return None

        sharpe = mean_return / std_return

        if annualize:
            # Assume ~250 betting days per year
            sharpe *= np.sqrt(250)

        return float(sharpe)

    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity curve data.

        Returns:
            List of (timestamp, bankroll) tuples.
        """
        curve = [(datetime.now(timezone.utc), self.initial_bankroll)]
        curve.extend(self._pnl_history)
        return curve

    def get_recent_snapshots(self, count: int = 10) -> List[PortfolioSnapshot]:
        """Get recent snapshots.

        Args:
            count: Number of snapshots to return.

        Returns:
            List of recent snapshots.
        """
        return self._snapshots[-count:]

    def summarize(self) -> str:
        """Get a text summary of portfolio performance.

        Returns:
            Summary string.
        """
        sharpe = self._calculate_sharpe()
        sharpe_str = f"{sharpe:.2f}" if sharpe else "N/A"

        return (
            f"Portfolio Summary\n"
            f"{'=' * 40}\n"
            f"Bankroll: ${self.bankroll:,.2f}\n"
            f"Total PnL: ${self.total_pnl:+,.2f} ({self.roi:+.1%})\n"
            f"Max Drawdown: {self.max_drawdown:.1%}\n"
            f"Current Drawdown: {self.current_drawdown:.1%}\n"
            f"Sharpe Ratio: {sharpe_str}\n"
        )

    def reset(self, initial_bankroll: Optional[float] = None) -> None:
        """Reset portfolio.

        Args:
            initial_bankroll: New starting bankroll.
        """
        config = get_risk_config()
        self.initial_bankroll = initial_bankroll or config.initial_bankroll
        self.bankroll = self.initial_bankroll
        self._peak_bankroll = self.initial_bankroll
        self._pnl_history.clear()
        self._snapshots.clear()

        logger.info(f"Portfolio reset with ${self.bankroll:.2f}")
