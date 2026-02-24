"""Closing Line Value (CLV) tracking and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CLVRecord:
    """Record of CLV for a single bet."""

    bet_id: int
    game_id: int
    bet_time: datetime
    side: str  # 'home', 'away'
    bet_prob: float
    closing_prob: float
    clv: float
    won: Optional[bool] = None
    pnl: Optional[float] = None


@dataclass
class CLVStats:
    """Aggregate CLV statistics."""

    total_bets: int
    avg_clv: float
    median_clv: float
    std_clv: float
    positive_clv_pct: float
    total_clv_points: float

    # Breakdown by side
    home_avg_clv: Optional[float] = None
    away_avg_clv: Optional[float] = None

    # Correlation with results
    clv_win_correlation: Optional[float] = None

    # Time series
    rolling_clv: List[float] = field(default_factory=list)


class CLVTracker:
    """Track and analyze closing line value performance.

    CLV is considered the best measure of betting skill because
    consistently beating the closing line indicates a true edge.
    """

    def __init__(self) -> None:
        """Initialize CLV tracker."""
        self._records: List[CLVRecord] = []

    def add_record(self, record: CLVRecord) -> None:
        """Add a CLV record.

        Args:
            record: CLVRecord to add.
        """
        self._records.append(record)

    def add_bet_result(
        self,
        bet_id: int,
        game_id: int,
        bet_time: datetime,
        side: str,
        bet_prob: float,
        closing_prob: float,
        won: Optional[bool] = None,
        pnl: Optional[float] = None,
    ) -> CLVRecord:
        """Add a bet result and calculate CLV.

        Args:
            bet_id: Bet ID.
            game_id: Game ID.
            bet_time: Time bet was placed.
            side: Side bet on.
            bet_prob: Implied probability at bet time.
            closing_prob: Closing implied probability.
            won: Whether bet won.
            pnl: Profit/loss.

        Returns:
            Created CLVRecord.
        """
        clv = closing_prob - bet_prob

        record = CLVRecord(
            bet_id=bet_id,
            game_id=game_id,
            bet_time=bet_time,
            side=side,
            bet_prob=bet_prob,
            closing_prob=closing_prob,
            clv=clv,
            won=won,
            pnl=pnl,
        )

        self.add_record(record)
        return record

    def get_stats(
        self,
        min_bets: int = 10,
    ) -> Optional[CLVStats]:
        """Calculate aggregate CLV statistics.

        Args:
            min_bets: Minimum bets required.

        Returns:
            CLVStats or None if insufficient data.
        """
        if len(self._records) < min_bets:
            return None

        clv_values = [r.clv for r in self._records]

        # Basic stats
        avg_clv = np.mean(clv_values)
        median_clv = np.median(clv_values)
        std_clv = np.std(clv_values)
        positive_pct = sum(1 for c in clv_values if c > 0) / len(clv_values)
        total_clv = sum(clv_values)

        # By side
        home_clv = [r.clv for r in self._records if r.side == "home"]
        away_clv = [r.clv for r in self._records if r.side == "away"]
        home_avg = np.mean(home_clv) if home_clv else None
        away_avg = np.mean(away_clv) if away_clv else None

        # CLV-win correlation
        settled = [(r.clv, r.won) for r in self._records if r.won is not None]
        if len(settled) >= min_bets:
            clv_vals = [s[0] for s in settled]
            wins = [float(s[1]) for s in settled]
            correlation = np.corrcoef(clv_vals, wins)[0, 1]
        else:
            correlation = None

        # Rolling CLV
        rolling = self._calculate_rolling_clv(window=20)

        return CLVStats(
            total_bets=len(self._records),
            avg_clv=avg_clv,
            median_clv=median_clv,
            std_clv=std_clv,
            positive_clv_pct=positive_pct,
            total_clv_points=total_clv,
            home_avg_clv=home_avg,
            away_avg_clv=away_avg,
            clv_win_correlation=correlation,
            rolling_clv=rolling,
        )

    def _calculate_rolling_clv(self, window: int = 20) -> List[float]:
        """Calculate rolling average CLV.

        Args:
            window: Window size for rolling average.

        Returns:
            List of rolling CLV values.
        """
        if len(self._records) < window:
            return []

        clv_values = [r.clv for r in self._records]
        rolling = []

        for i in range(window - 1, len(clv_values)):
            window_values = clv_values[i - window + 1 : i + 1]
            rolling.append(np.mean(window_values))

        return rolling

    def analyze_by_time_to_close(
        self,
        buckets: List[int] | None = None,
    ) -> Dict[str, float]:
        """Analyze CLV by time to game close.

        Args:
            buckets: Time buckets in hours [24, 12, 6, 2].

        Returns:
            Dictionary of bucket -> avg CLV.
        """
        if buckets is None:
            buckets = [24, 12, 6, 2]

        results: Dict[str, List[float]] = {f">{b}h": [] for b in buckets}
        results["<2h"] = []

        # Would need game close time to implement fully
        # For now, return empty
        logger.warning("Time-to-close analysis requires game close times")

        return {k: np.mean(v) if v else 0 for k, v in results.items()}

    def get_sharpness_grade(self) -> str:
        """Get a letter grade for betting sharpness based on CLV.

        Returns:
            Grade from A+ to F.
        """
        stats = self.get_stats()
        if stats is None:
            return "N/A"

        avg_clv = stats.avg_clv

        # Grading scale based on average CLV
        # Positive CLV = beating the market
        if avg_clv >= 0.03:  # 3%+
            return "A+"
        elif avg_clv >= 0.02:
            return "A"
        elif avg_clv >= 0.01:
            return "B+"
        elif avg_clv >= 0.005:
            return "B"
        elif avg_clv >= 0:
            return "C"
        elif avg_clv >= -0.01:
            return "D"
        else:
            return "F"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to DataFrame.

        Returns:
            DataFrame of CLV records.
        """
        records = []
        for r in self._records:
            records.append(
                {
                    "bet_id": r.bet_id,
                    "game_id": r.game_id,
                    "bet_time": r.bet_time,
                    "side": r.side,
                    "bet_prob": r.bet_prob,
                    "closing_prob": r.closing_prob,
                    "clv": r.clv,
                    "won": r.won,
                    "pnl": r.pnl,
                }
            )

        return pd.DataFrame(records)

    def summarize(self) -> str:
        """Get a text summary of CLV performance.

        Returns:
            Summary string.
        """
        stats = self.get_stats()
        if stats is None:
            return "Insufficient data for CLV analysis"

        grade = self.get_sharpness_grade()

        return (
            f"CLV Summary ({stats.total_bets} bets)\n"
            f"{'=' * 40}\n"
            f"Average CLV: {stats.avg_clv:+.2%}\n"
            f"Median CLV: {stats.median_clv:+.2%}\n"
            f"Positive CLV Rate: {stats.positive_clv_pct:.1%}\n"
            f"Total CLV Points: {stats.total_clv_points:+.2%}\n"
            f"Sharpness Grade: {grade}\n"
        )

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
