"""Team-level metrics and statistics calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import get_model_config


@dataclass
class EloRating:
    """Elo rating for a team."""

    team_id: int
    rating: float
    games_played: int
    last_updated: datetime


@dataclass
class TeamMetrics:
    """Comprehensive team metrics."""

    team_id: int
    elo_rating: float
    offensive_rating: Optional[float] = None
    defensive_rating: Optional[float] = None
    net_rating: Optional[float] = None
    pace: Optional[float] = None
    win_pct: Optional[float] = None
    ats_record: Optional[float] = None
    home_record: Optional[float] = None
    away_record: Optional[float] = None
    last_10_record: Optional[float] = None
    strength_of_schedule: Optional[float] = None


class TeamMetricsCalculator:
    """Calculate team-level metrics for predictions."""

    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        self.config = get_model_config()
        self.elo_config = self.config.elo_config
        self._elo_ratings: Dict[int, EloRating] = {}

    @property
    def initial_elo(self) -> float:
        """Initial Elo rating for new teams."""
        return self.elo_config.get("initial_rating", 1500)

    @property
    def k_factor(self) -> float:
        """K-factor for Elo updates."""
        return self.elo_config.get("k_factor", 20)

    @property
    def home_advantage(self) -> Dict[str, float]:
        """Home advantage by league."""
        return self.elo_config.get("home_advantage", {"nfl": 65, "nba": 100})

    def get_elo(self, team_id: int) -> float:
        """Get current Elo rating for a team.

        Args:
            team_id: Team ID.

        Returns:
            Current Elo rating.
        """
        if team_id in self._elo_ratings:
            return self._elo_ratings[team_id].rating
        return self.initial_elo

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using Elo formula.

        Args:
            rating_a: Team A's Elo rating.
            rating_b: Team B's Elo rating.

        Returns:
            Expected score (win probability) for team A.
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_elo(
        self,
        team_id: int,
        opponent_rating: float,
        actual_score: float,
        margin: Optional[float] = None,
    ) -> float:
        """Update a team's Elo rating after a game.

        Args:
            team_id: Team ID.
            opponent_rating: Opponent's Elo rating.
            actual_score: Actual result (1 for win, 0 for loss, 0.5 for tie).
            margin: Point margin (optional, for margin-adjusted Elo).

        Returns:
            New Elo rating.
        """
        current_rating = self.get_elo(team_id)
        expected = self.expected_score(current_rating, opponent_rating)

        # Apply margin multiplier if provided
        k = self.k_factor
        if margin is not None:
            margin_mult = self.elo_config.get("margin_multiplier", 0.04)
            k *= 1.0 + margin_mult * abs(margin)

        new_rating = current_rating + k * (actual_score - expected)

        # Update stored rating
        if team_id in self._elo_ratings:
            self._elo_ratings[team_id].rating = new_rating
            self._elo_ratings[team_id].games_played += 1
            self._elo_ratings[team_id].last_updated = datetime.now()
        else:
            self._elo_ratings[team_id] = EloRating(
                team_id=team_id,
                rating=new_rating,
                games_played=1,
                last_updated=datetime.now(),
            )

        return new_rating

    def regress_to_mean(self, team_id: int, regression_factor: Optional[float] = None) -> float:
        """Regress a team's Elo rating toward the mean (used for new seasons).

        Args:
            team_id: Team ID.
            regression_factor: How much to regress (0=none, 1=fully to mean).

        Returns:
            New Elo rating.
        """
        if regression_factor is None:
            regression_factor = self.elo_config.get("season_regression", 0.33)

        current = self.get_elo(team_id)
        new_rating = current + regression_factor * (self.initial_elo - current)

        if team_id in self._elo_ratings:
            self._elo_ratings[team_id].rating = new_rating

        return new_rating

    def calculate_win_probability(
        self,
        home_team_id: int,
        away_team_id: int,
        league: str,
        is_neutral: bool = False,
    ) -> float:
        """Calculate win probability for home team.

        Args:
            home_team_id: Home team ID.
            away_team_id: Away team ID.
            league: League code.
            is_neutral: Whether game is at neutral site.

        Returns:
            Win probability for home team.
        """
        home_elo = self.get_elo(home_team_id)
        away_elo = self.get_elo(away_team_id)

        # Add home advantage
        if not is_neutral:
            advantage = self.home_advantage.get(league.lower(), 50)
            home_elo += advantage

        return self.expected_score(home_elo, away_elo)

    def process_game_result(
        self,
        home_team_id: int,
        away_team_id: int,
        home_score: int,
        away_score: int,
        league: str,
        is_neutral: bool = False,
    ) -> Tuple[float, float]:
        """Process a game result and update Elo ratings.

        Args:
            home_team_id: Home team ID.
            away_team_id: Away team ID.
            home_score: Home team score.
            away_score: Away team score.
            league: League code.
            is_neutral: Whether game was at neutral site.

        Returns:
            Tuple of (new_home_elo, new_away_elo).
        """
        home_elo = self.get_elo(home_team_id)
        away_elo = self.get_elo(away_team_id)

        # Adjust for home advantage
        if not is_neutral:
            advantage = self.home_advantage.get(league.lower(), 50)
            home_elo_adj = home_elo + advantage
        else:
            home_elo_adj = home_elo

        # Determine result
        if home_score > away_score:
            home_result, away_result = 1.0, 0.0
        elif away_score > home_score:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        margin = home_score - away_score

        # Update ratings
        new_home = self.update_elo(
            home_team_id,
            away_elo,
            home_result,
            margin=margin,
        )
        new_away = self.update_elo(
            away_team_id,
            home_elo_adj,
            away_result,
            margin=-margin,
        )

        return new_home, new_away

    def calculate_team_metrics(
        self,
        team_id: int,
        games: pd.DataFrame,
    ) -> TeamMetrics:
        """Calculate comprehensive team metrics from game history.

        Args:
            team_id: Team ID.
            games: DataFrame of team's games.

        Returns:
            TeamMetrics object.
        """
        if games.empty:
            return TeamMetrics(team_id=team_id, elo_rating=self.get_elo(team_id))

        # Filter to team's games
        home_games = games[games["home_team_id"] == team_id]
        away_games = games[games["away_team_id"] == team_id]

        # Win percentage
        home_wins = len(home_games[home_games["home_score"] > home_games["away_score"]])
        away_wins = len(away_games[away_games["away_score"] > away_games["home_score"]])
        total_games = len(home_games) + len(away_games)
        win_pct = (home_wins + away_wins) / total_games if total_games > 0 else 0.5

        # Home/away records
        home_record = home_wins / len(home_games) if len(home_games) > 0 else 0.5
        away_record = away_wins / len(away_games) if len(away_games) > 0 else 0.5

        # Points metrics
        home_pts_for = home_games["home_score"].mean() if len(home_games) > 0 else 0
        home_pts_against = home_games["away_score"].mean() if len(home_games) > 0 else 0
        away_pts_for = away_games["away_score"].mean() if len(away_games) > 0 else 0
        away_pts_against = away_games["home_score"].mean() if len(away_games) > 0 else 0

        offensive_rating = (home_pts_for + away_pts_for) / 2 if total_games > 0 else None
        defensive_rating = (home_pts_against + away_pts_against) / 2 if total_games > 0 else None
        net_rating = (
            offensive_rating - defensive_rating if offensive_rating and defensive_rating else None
        )

        return TeamMetrics(
            team_id=team_id,
            elo_rating=self.get_elo(team_id),
            offensive_rating=offensive_rating,
            defensive_rating=defensive_rating,
            net_rating=net_rating,
            win_pct=win_pct,
            home_record=home_record,
            away_record=away_record,
        )

    def build_features(
        self,
        home_team_id: int,
        away_team_id: int,
        league: str,
        is_neutral: bool = False,
    ) -> Dict[str, float]:
        """Build feature dictionary for a game.

        Args:
            home_team_id: Home team ID.
            away_team_id: Away team ID.
            league: League code.
            is_neutral: Whether game is at neutral site.

        Returns:
            Dictionary of features.
        """
        home_elo = self.get_elo(home_team_id)
        away_elo = self.get_elo(away_team_id)
        elo_diff = home_elo - away_elo

        home_adv = 0 if is_neutral else self.home_advantage.get(league.lower(), 50)

        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": elo_diff,
            "elo_diff_with_hfa": elo_diff + home_adv,
            "home_advantage": home_adv,
            "is_neutral": float(is_neutral),
        }

    def load_ratings(self, ratings_data: List[Dict[str, Any]]) -> None:
        """Load Elo ratings from stored data.

        Args:
            ratings_data: List of rating dictionaries.
        """
        for data in ratings_data:
            self._elo_ratings[data["team_id"]] = EloRating(
                team_id=data["team_id"],
                rating=data["rating"],
                games_played=data.get("games_played", 0),
                last_updated=data.get("last_updated", datetime.now()),
            )

    def get_all_ratings(self) -> Dict[int, float]:
        """Get all current Elo ratings.

        Returns:
            Dictionary mapping team_id to rating.
        """
        return {tid: r.rating for tid, r in self._elo_ratings.items()}
