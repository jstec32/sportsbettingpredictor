"""Player-level metrics and injury impact calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class PlayerImpact:
    """Player impact rating."""

    player_id: str
    player_name: str
    team_id: int
    position: str
    impact_rating: float  # 0 to 1 scale
    injury_status: Optional[str] = None


@dataclass
class InjuryImpact:
    """Total injury impact for a team."""

    team_id: int
    total_impact: float
    key_players_out: List[str]
    position_impact: Dict[str, float]


# Position importance weights by sport
POSITION_WEIGHTS = {
    "NFL": {
        "QB": 1.0,
        "RB": 0.4,
        "WR": 0.35,
        "TE": 0.25,
        "OL": 0.2,
        "OT": 0.25,
        "OG": 0.15,
        "C": 0.2,
        "DL": 0.2,
        "DE": 0.25,
        "DT": 0.2,
        "LB": 0.25,
        "CB": 0.3,
        "S": 0.2,
        "K": 0.15,
        "P": 0.1,
    },
    "NBA": {
        "PG": 0.6,
        "SG": 0.5,
        "SF": 0.5,
        "PF": 0.5,
        "C": 0.55,
        "G": 0.55,
        "F": 0.5,
    },
}

# Injury status severity
INJURY_STATUS_SEVERITY = {
    "out": 1.0,
    "doubtful": 0.85,
    "questionable": 0.5,
    "probable": 0.1,
    "day-to-day": 0.3,
    "ir": 1.0,
    "pup": 1.0,
    "suspended": 1.0,
}


class PlayerMetricsCalculator:
    """Calculate player-level metrics and injury impacts."""

    def __init__(self) -> None:
        """Initialize the player metrics calculator."""
        self._player_impacts: Dict[str, PlayerImpact] = {}

    def get_position_weight(self, position: str | None, league: str) -> float:
        """Get the importance weight for a position.

        Args:
            position: Position code (can be None).
            league: League code.

        Returns:
            Position weight (0 to 1).
        """
        if not position:
            return 0.2  # Default weight for unknown position

        weights = POSITION_WEIGHTS.get(league.upper(), {})
        # Try exact match first, then partial match
        if position in weights:
            return weights[position]

        for pos, weight in weights.items():
            if pos in position or position in pos:
                return weight

        return 0.2  # Default weight

    def get_injury_severity(self, status: str) -> float:
        """Get the severity factor for an injury status.

        Args:
            status: Injury status string.

        Returns:
            Severity factor (0 to 1).
        """
        status_lower = status.lower()
        for key, severity in INJURY_STATUS_SEVERITY.items():
            if key in status_lower:
                return severity
        return 0.5  # Default to questionable

    def calculate_player_impact(
        self,
        player_id: str,
        player_name: str,
        team_id: int,
        position: str,
        league: str,
        stats: Dict[str, float] | None = None,
    ) -> PlayerImpact:
        """Calculate a player's impact rating.

        Args:
            player_id: Player ID.
            player_name: Player name.
            team_id: Team ID.
            position: Player position.
            league: League code.
            stats: Optional player statistics.

        Returns:
            PlayerImpact object.
        """
        base_weight = self.get_position_weight(position, league)

        # Adjust based on stats if provided
        if stats:
            # Example adjustments based on stats
            if league == "NFL" and position == "QB":
                # QB rating adjustment
                qb_rating = stats.get("passer_rating", 90)
                base_weight *= min(1.2, qb_rating / 100)
            elif league == "NBA":
                # Usage rate adjustment
                usage = stats.get("usage_rate", 20)
                base_weight *= min(1.3, usage / 20)

        impact = PlayerImpact(
            player_id=player_id,
            player_name=player_name,
            team_id=team_id,
            position=position,
            impact_rating=min(1.0, base_weight),
        )

        self._player_impacts[player_id] = impact
        return impact

    def calculate_injury_impact(
        self,
        injuries: List[Dict[str, Any]],
        team_id: int,
        league: str,
    ) -> InjuryImpact:
        """Calculate total injury impact for a team.

        Args:
            injuries: List of injury dictionaries.
            team_id: Team ID.
            league: League code.

        Returns:
            InjuryImpact object.
        """
        total_impact = 0.0
        key_players_out: List[str] = []
        position_impact: Dict[str, float] = {}

        for injury in injuries:
            position = injury.get("position", "")
            status = injury.get("injury_status", "")
            player_name = injury.get("player_name", "Unknown")

            # Get position weight
            pos_weight = self.get_position_weight(position, league)

            # Get injury severity
            severity = self.get_injury_severity(status)

            # Calculate this player's impact
            impact = pos_weight * severity

            # Track position-level impact
            if position:
                position_impact[position] = position_impact.get(position, 0) + impact

            # Add to total
            total_impact += impact

            # Track key players (high impact or starter)
            if impact > 0.3:
                key_players_out.append(f"{player_name} ({position})")

        # Normalize total impact (cap at 0.5 - no team loses more than 50%)
        total_impact = min(0.5, total_impact)

        return InjuryImpact(
            team_id=team_id,
            total_impact=total_impact,
            key_players_out=key_players_out,
            position_impact=position_impact,
        )

    def build_injury_features(
        self,
        home_injuries: List[Dict[str, Any]],
        away_injuries: List[Dict[str, Any]],
        home_team_id: int,
        away_team_id: int,
        league: str,
    ) -> Dict[str, float]:
        """Build injury-related features for a game.

        Args:
            home_injuries: Home team injuries.
            away_injuries: Away team injuries.
            home_team_id: Home team ID.
            away_team_id: Away team ID.
            league: League code.

        Returns:
            Dictionary of injury features.
        """
        home_impact = self.calculate_injury_impact(home_injuries, home_team_id, league)
        away_impact = self.calculate_injury_impact(away_injuries, away_team_id, league)

        return {
            "home_injury_impact": home_impact.total_impact,
            "away_injury_impact": away_impact.total_impact,
            "injury_impact_diff": away_impact.total_impact - home_impact.total_impact,
            "home_key_players_out": len(home_impact.key_players_out),
            "away_key_players_out": len(away_impact.key_players_out),
            "home_qb_out": 1.0 if any("QB" in p for p in home_impact.key_players_out) else 0.0,
            "away_qb_out": 1.0 if any("QB" in p for p in away_impact.key_players_out) else 0.0,
        }

    def get_player_impact(self, player_id: str) -> Optional[PlayerImpact]:
        """Get stored player impact.

        Args:
            player_id: Player ID.

        Returns:
            PlayerImpact or None.
        """
        return self._player_impacts.get(player_id)

    def load_player_impacts(self, impacts_data: List[Dict[str, Any]]) -> None:
        """Load player impacts from stored data.

        Args:
            impacts_data: List of impact dictionaries.
        """
        for data in impacts_data:
            self._player_impacts[data["player_id"]] = PlayerImpact(
                player_id=data["player_id"],
                player_name=data["player_name"],
                team_id=data["team_id"],
                position=data["position"],
                impact_rating=data["impact_rating"],
            )
