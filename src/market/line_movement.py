"""Line movement tracking and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class LineSnapshot:
    """A snapshot of line/odds at a point in time."""

    game_id: int
    captured_at: datetime
    spread_home: Optional[float] = None
    spread_away: Optional[float] = None
    total: Optional[float] = None
    home_moneyline: Optional[float] = None
    away_moneyline: Optional[float] = None
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    source: str = "consensus"


@dataclass
class LineMovementSummary:
    """Summary of line movement for a game."""

    game_id: int
    opening_spread: Optional[float] = None
    current_spread: Optional[float] = None
    closing_spread: Optional[float] = None
    spread_movement: Optional[float] = None

    opening_total: Optional[float] = None
    current_total: Optional[float] = None
    closing_total: Optional[float] = None
    total_movement: Optional[float] = None

    opening_home_prob: Optional[float] = None
    current_home_prob: Optional[float] = None
    closing_home_prob: Optional[float] = None
    prob_movement: Optional[float] = None

    steam_move_detected: bool = False
    reverse_line_movement: bool = False
    sharp_action_side: Optional[str] = None

    snapshots: List[LineSnapshot] | None = None


class LineMovementTracker:
    """Track and analyze line movements."""

    def __init__(self) -> None:
        """Initialize line movement tracker."""
        self._snapshots: Dict[int, List[LineSnapshot]] = {}

    def add_snapshot(self, snapshot: LineSnapshot) -> None:
        """Add a line snapshot.

        Args:
            snapshot: LineSnapshot to add.
        """
        if snapshot.game_id not in self._snapshots:
            self._snapshots[snapshot.game_id] = []

        self._snapshots[snapshot.game_id].append(snapshot)

        # Keep sorted by time
        self._snapshots[snapshot.game_id].sort(key=lambda x: x.captured_at)

    def add_from_dict(self, data: Dict[str, Any]) -> None:
        """Add snapshot from dictionary.

        Args:
            data: Dictionary with snapshot data.
        """
        snapshot = LineSnapshot(
            game_id=data["game_id"],
            captured_at=data.get("captured_at", datetime.now()),
            spread_home=data.get("spread_home"),
            spread_away=data.get("spread_away"),
            total=data.get("total_line") or data.get("total"),
            home_moneyline=data.get("home_odds") or data.get("home_moneyline"),
            away_moneyline=data.get("away_odds") or data.get("away_moneyline"),
            home_prob=data.get("home_prob"),
            away_prob=data.get("away_prob"),
            source=data.get("source", "consensus"),
        )
        self.add_snapshot(snapshot)

    def get_summary(self, game_id: int) -> Optional[LineMovementSummary]:
        """Get line movement summary for a game.

        Args:
            game_id: Game ID.

        Returns:
            LineMovementSummary or None if no data.
        """
        if game_id not in self._snapshots or not self._snapshots[game_id]:
            return None

        snapshots = self._snapshots[game_id]

        if len(snapshots) < 1:
            return None

        opening = snapshots[0]
        current = snapshots[-1]

        summary = LineMovementSummary(
            game_id=game_id,
            opening_spread=opening.spread_home,
            current_spread=current.spread_home,
            opening_total=opening.total,
            current_total=current.total,
            opening_home_prob=opening.home_prob,
            current_home_prob=current.home_prob,
            snapshots=snapshots,
        )

        # Calculate movements
        if opening.spread_home is not None and current.spread_home is not None:
            summary.spread_movement = current.spread_home - opening.spread_home

        if opening.total is not None and current.total is not None:
            summary.total_movement = current.total - opening.total

        if opening.home_prob is not None and current.home_prob is not None:
            summary.prob_movement = current.home_prob - opening.home_prob

        # Detect steam moves
        summary.steam_move_detected = self._detect_steam_move(snapshots)

        # Detect sharp action
        summary.sharp_action_side = self._detect_sharp_action(snapshots)

        return summary

    def _detect_steam_move(self, snapshots: List[LineSnapshot]) -> bool:
        """Detect steam move (sharp sudden line movement).

        A steam move is a rapid line movement of 1+ points in a short time.

        Args:
            snapshots: List of line snapshots.

        Returns:
            True if steam move detected.
        """
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]

            # Check time difference
            time_diff = curr.captured_at - prev.captured_at
            if time_diff > timedelta(hours=2):
                continue

            # Check spread movement
            if prev.spread_home is not None and curr.spread_home is not None:
                spread_diff = abs(curr.spread_home - prev.spread_home)
                if spread_diff >= 1.0:
                    logger.debug(
                        f"Steam move detected: {spread_diff} point move in "
                        f"{time_diff.total_seconds() / 60:.0f} minutes"
                    )
                    return True

        return False

    def _detect_sharp_action(
        self,
        snapshots: List[LineSnapshot],
    ) -> Optional[str]:
        """Detect which side sharp money is on.

        Sharp indicators:
        - Line moving against public betting percentages
        - Significant line movement on limited handle
        - Steam moves

        Args:
            snapshots: List of line snapshots.

        Returns:
            'home', 'away', or None.
        """
        if len(snapshots) < 2:
            return None

        opening = snapshots[0]
        current = snapshots[-1]

        if opening.spread_home is None or current.spread_home is None:
            return None

        movement = current.spread_home - opening.spread_home

        # Significant movement toward one side suggests sharp action
        if movement <= -1.5:
            return "home"  # Line moving toward home = sharp on home
        elif movement >= 1.5:
            return "away"  # Line moving toward away = sharp on away

        return None

    def get_closing_line_value(
        self,
        game_id: int,
        bet_time: datetime,
        bet_side: str,
    ) -> Optional[float]:
        """Calculate closing line value for a bet.

        CLV = closing_prob - bet_prob for the side you bet

        Args:
            game_id: Game ID.
            bet_time: Time bet was placed.
            bet_side: Side bet on ('home' or 'away').

        Returns:
            CLV value or None.
        """
        if game_id not in self._snapshots:
            return None

        snapshots = self._snapshots[game_id]
        if not snapshots:
            return None

        # Find snapshot closest to bet time
        bet_snapshot = None
        for s in snapshots:
            if s.captured_at <= bet_time:
                bet_snapshot = s
            else:
                break

        if bet_snapshot is None:
            bet_snapshot = snapshots[0]

        closing = snapshots[-1]

        # Get probabilities
        if bet_side == "home":
            bet_prob = bet_snapshot.home_prob
            closing_prob = closing.home_prob
        else:
            bet_prob = bet_snapshot.away_prob
            closing_prob = closing.away_prob

        if bet_prob is None or closing_prob is None:
            return None

        return closing_prob - bet_prob

    def to_dataframe(self, game_id: Optional[int] = None) -> pd.DataFrame:
        """Convert snapshots to DataFrame.

        Args:
            game_id: Optional game ID to filter.

        Returns:
            DataFrame of snapshots.
        """
        records = []

        games = [game_id] if game_id else list(self._snapshots.keys())

        for gid in games:
            if gid not in self._snapshots:
                continue

            for snap in self._snapshots[gid]:
                records.append(
                    {
                        "game_id": snap.game_id,
                        "captured_at": snap.captured_at,
                        "spread_home": snap.spread_home,
                        "total": snap.total,
                        "home_moneyline": snap.home_moneyline,
                        "away_moneyline": snap.away_moneyline,
                        "home_prob": snap.home_prob,
                        "away_prob": snap.away_prob,
                        "source": snap.source,
                    }
                )

        return pd.DataFrame(records)

    def clear(self, game_id: Optional[int] = None) -> None:
        """Clear snapshots.

        Args:
            game_id: Optional game ID to clear. If None, clears all.
        """
        if game_id:
            if game_id in self._snapshots:
                del self._snapshots[game_id]
        else:
            self._snapshots.clear()
