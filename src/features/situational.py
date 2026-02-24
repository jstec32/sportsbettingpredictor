"""Situational features like rest days, travel, back-to-backs, and recent form."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SituationalFactors:
    """Situational factors for a team entering a game."""

    rest_days: int
    travel_miles: float
    time_zone_change: int
    is_back_to_back: bool
    is_second_of_back_to_back: bool
    games_in_last_7_days: int
    games_in_last_14_days: int


@dataclass
class FormFactors:
    """Recent form and momentum for a team."""

    win_pct_last5: float
    win_pct_last10: float
    win_streak: int           # Positive = winning streak, negative = losing streak
    home_win_pct: float       # Season home win%
    away_win_pct: float       # Season away win%
    home_win_pct_last10: float
    away_win_pct_last10: float
    point_diff_last5: float   # Avg scoring margin last 5 games
    point_diff_last10: float  # Avg scoring margin last 10 games


# Stadium / arena coordinates (lat, lon)
VENUE_COORDS: Dict[str, Tuple[float, float]] = {
    # NFL
    "State Farm Stadium": (33.5276, -112.2626),
    "Mercedes-Benz Stadium": (33.7553, -84.4006),
    "M&T Bank Stadium": (39.2780, -76.6227),
    "Highmark Stadium": (42.7738, -78.7870),
    "Bank of America Stadium": (35.2258, -80.8528),
    "Soldier Field": (41.8623, -87.6167),
    "Paycor Stadium": (39.0954, -84.5160),
    "FirstEnergy Stadium": (41.5061, -81.6995),
    "AT&T Stadium": (32.7473, -97.0945),
    "Empower Field at Mile High": (39.7439, -105.0201),
    "Ford Field": (42.3400, -83.0456),
    "Lambeau Field": (44.5013, -88.0622),
    "NRG Stadium": (29.6847, -95.4107),
    "Lucas Oil Stadium": (39.7601, -86.1639),
    "TIAA Bank Field": (30.3239, -81.6373),
    "Arrowhead Stadium": (39.0489, -94.4839),
    "Allegiant Stadium": (36.0909, -115.1833),
    "SoFi Stadium": (33.9535, -118.3392),
    "Hard Rock Stadium": (25.9580, -80.2389),
    "U.S. Bank Stadium": (44.9737, -93.2575),
    "Gillette Stadium": (42.0909, -71.2643),
    "Caesars Superdome": (29.9511, -90.0812),
    "MetLife Stadium": (40.8135, -74.0745),
    "Lumen Field": (47.5952, -122.3316),
    "Lincoln Financial Field": (39.9008, -75.1675),
    "Acrisure Stadium": (40.4468, -80.0158),
    "Levi's Stadium": (37.4032, -121.9698),
    "Raymond James Stadium": (27.9759, -82.5033),
    "Nissan Stadium": (36.1665, -86.7713),
    "FedExField": (38.9076, -76.8645),
    # NBA arenas
    "State Farm Arena": (33.7573, -84.3963),
    "TD Garden": (42.3662, -71.0621),
    "Spectrum Center": (35.2251, -80.8392),
    "United Center": (41.8807, -87.6742),
    "Rocket Mortgage FieldHouse": (41.4965, -81.6882),
    "American Airlines Center": (32.7905, -96.8103),
    "Ball Arena": (39.7486, -105.0076),
    "Little Caesars Arena": (42.3410, -83.0554),
    "Chase Center": (37.7679, -122.3879),
    "Toyota Center": (29.7508, -95.3621),
    "Gainbridge Fieldhouse": (39.7640, -86.1555),
    "Crypto.com Arena": (34.0430, -118.2673),
    "FedExForum": (35.1383, -90.0505),
    "Kaseya Center": (25.7814, -80.1870),
    "Fiserv Forum": (43.0451, -87.9170),
    "Target Center": (44.9795, -93.2762),
    "Smoothie King Center": (29.9490, -90.0822),
    "Madison Square Garden": (40.7505, -73.9934),
    "Paycom Center": (35.4634, -97.5151),
    "Amway Center": (28.5392, -81.3839),
    "Wells Fargo Center": (39.9012, -75.1720),
    "Footprint Center": (33.4457, -112.0712),
    "Moda Center": (45.5316, -122.6668),
    "Golden 1 Center": (38.5802, -121.4997),
    "AT&T Center": (29.4270, -98.4375),
    "Climate Pledge Arena": (47.6221, -122.3540),
    "Scotiabank Arena": (43.6435, -79.3791),
    "Vivint Arena": (40.7683, -111.9011),
    "Capital One Arena": (38.8981, -77.0209),
    "Kia Center": (28.5392, -81.3839),
}

# Approximate timezone by longitude bucket (for teams without explicit tz data)
_TZ_OFFSETS: Dict[str, int] = {
    "America/New_York": -5,
    "America/Chicago": -6,
    "America/Denver": -7,
    "America/Los_Angeles": -8,
    "America/Phoenix": -7,
    "EST": -5,
    "CST": -6,
    "MST": -7,
    "PST": -8,
}


class SituationalFeatures:
    """Calculate situational features for individual games."""

    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two coordinates using Haversine formula.

        Returns distance in miles.
        """
        R = 3959.0  # Earth's radius in miles
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    def calculate_rest_days(
        self,
        game_date: datetime,
        last_game_date: Optional[datetime],
    ) -> int:
        """Days of rest since last game. Returns 7 if no prior game."""
        if last_game_date is None:
            return 7
        delta = game_date - last_game_date
        return max(0, delta.days - 1)  # -1 because game day itself doesn't count

    def calculate_travel(
        self,
        home_venue: str,
        away_venue: str,
        home_lat: Optional[float] = None,
        home_lon: Optional[float] = None,
        away_lat: Optional[float] = None,
        away_lon: Optional[float] = None,
    ) -> float:
        """Travel distance for away team in miles. Returns 0 if coordinates unavailable."""
        if home_lat is None or home_lon is None:
            home_coords = VENUE_COORDS.get(home_venue)
            if not home_coords:
                return 0.0
            home_lat, home_lon = home_coords

        if away_lat is None or away_lon is None:
            away_coords = VENUE_COORDS.get(away_venue)
            if not away_coords:
                return 0.0
            away_lat, away_lon = away_coords

        return self.haversine_distance(home_lat, home_lon, away_lat, away_lon)

    def calculate_timezone_change(
        self,
        home_tz: Optional[str],
        away_tz: Optional[str],
    ) -> int:
        """Timezone hours the away team crosses (positive = east, negative = west)."""
        if not home_tz or not away_tz:
            return 0
        home_offset = _TZ_OFFSETS.get(home_tz, -6)
        away_offset = _TZ_OFFSETS.get(away_tz, -6)
        return home_offset - away_offset

    def calculate_situational_factors(
        self,
        game_date: datetime,
        team_games: pd.DataFrame,
        is_home: bool = True,
    ) -> SituationalFactors:
        """Calculate fatigue and schedule situational factors for one team.

        Args:
            game_date: Date of the upcoming game.
            team_games: DataFrame of all the team's historical games (all columns).
            is_home: Whether this team is the home team (unused here, for extension).

        Returns:
            SituationalFactors dataclass.
        """
        past = team_games[team_games["game_datetime"] < game_date].sort_values(
            "game_datetime", ascending=False
        )

        if len(past) > 0:
            last_game_dt = past.iloc[0]["game_datetime"]
            if isinstance(last_game_dt, str):
                last_game_dt = datetime.fromisoformat(last_game_dt)
            elif hasattr(last_game_dt, "to_pydatetime"):
                last_game_dt = last_game_dt.to_pydatetime()
            rest_days = self.calculate_rest_days(game_date, last_game_dt)
        else:
            rest_days = 7

        is_b2b = rest_days == 0

        week_ago = game_date - timedelta(days=7)
        two_weeks_ago = game_date - timedelta(days=14)
        games_7d = int((past["game_datetime"] >= week_ago).sum())
        games_14d = int((past["game_datetime"] >= two_weeks_ago).sum())

        return SituationalFactors(
            rest_days=rest_days,
            travel_miles=0.0,
            time_zone_change=0,
            is_back_to_back=is_b2b,
            is_second_of_back_to_back=is_b2b,
            games_in_last_7_days=games_7d,
            games_in_last_14_days=games_14d,
        )

    def build_situational_features(
        self,
        home_factors: SituationalFactors,
        away_factors: SituationalFactors,
        travel_miles: float = 0.0,
        tz_change: int = 0,
    ) -> Dict[str, float]:
        """Build schedule/fatigue feature dictionary."""
        return {
            "home_rest_days": float(home_factors.rest_days),
            "away_rest_days": float(away_factors.rest_days),
            "rest_days_diff": float(home_factors.rest_days - away_factors.rest_days),
            "home_b2b": float(home_factors.is_back_to_back),
            "away_b2b": float(away_factors.is_back_to_back),
            "travel_miles": travel_miles,
            "travel_factor": min(1.0, travel_miles / 2500.0),
            "tz_change": float(tz_change),
            "tz_change_abs": float(abs(tz_change)),
            "home_games_7d": float(home_factors.games_in_last_7_days),
            "away_games_7d": float(away_factors.games_in_last_7_days),
            "home_games_14d": float(home_factors.games_in_last_14_days),
            "away_games_14d": float(away_factors.games_in_last_14_days),
            "fatigue_diff": float(
                away_factors.games_in_last_7_days - home_factors.games_in_last_7_days
            ),
        }

    def get_rest_advantage(
        self,
        home_rest: int,
        away_rest: int,
        league: str,
    ) -> float:
        """Probability boost from rest differential (-1 to 1)."""
        if league.upper() == "NBA":
            if away_rest == 0 and home_rest > 0:
                return 0.05
            elif home_rest == 0 and away_rest > 0:
                return -0.05
            else:
                return min(0.03, (home_rest - away_rest) * 0.01)
        else:
            return min(0.02, (home_rest - away_rest) * 0.005)


class FormCalculator:
    """Calculate recent form and momentum features for a team."""

    def _team_results(
        self,
        team_id: int,
        past_games: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the team's game results sorted newest-first.

        Each row has columns: game_datetime, won (bool), point_diff (from team's perspective).
        """
        if past_games.empty:
            return pd.DataFrame(columns=["game_datetime", "won", "point_diff"])

        home_mask = past_games["home_team_id"] == team_id
        away_mask = past_games["away_team_id"] == team_id

        records = []
        for _, row in past_games[home_mask | away_mask].iterrows():
            is_home = row["home_team_id"] == team_id
            home_score = row.get("home_score", 0) or 0
            away_score = row.get("away_score", 0) or 0
            if is_home:
                won = home_score > away_score
                diff = home_score - away_score
            else:
                won = away_score > home_score
                diff = away_score - home_score
            records.append({
                "game_datetime": row["game_datetime"],
                "is_home": is_home,
                "won": won,
                "point_diff": float(diff),
            })

        if not records:
            return pd.DataFrame(columns=["game_datetime", "is_home", "won", "point_diff"])

        df = pd.DataFrame(records)
        return df.sort_values("game_datetime", ascending=False)

    def calculate_form(
        self,
        team_id: int,
        past_games: pd.DataFrame,
    ) -> FormFactors:
        """Calculate recent form for a team using only past games.

        Args:
            team_id: Team ID to compute form for.
            past_games: DataFrame of all games before the target game date.

        Returns:
            FormFactors dataclass.
        """
        results = self._team_results(team_id, past_games)

        # Defaults for teams with no history
        if results.empty:
            return FormFactors(
                win_pct_last5=0.5,
                win_pct_last10=0.5,
                win_streak=0,
                home_win_pct=0.5,
                away_win_pct=0.5,
                home_win_pct_last10=0.5,
                away_win_pct_last10=0.5,
                point_diff_last5=0.0,
                point_diff_last10=0.0,
            )

        last5 = results.head(5)
        last10 = results.head(10)

        win_pct_last5 = float(last5["won"].mean()) if len(last5) > 0 else 0.5
        win_pct_last10 = float(last10["won"].mean()) if len(last10) > 0 else 0.5
        point_diff_last5 = float(last5["point_diff"].mean()) if len(last5) > 0 else 0.0
        point_diff_last10 = float(last10["point_diff"].mean()) if len(last10) > 0 else 0.0

        # Win/loss streak (positive = wins, negative = losses)
        streak = 0
        if len(results) > 0:
            first_result = bool(results.iloc[0]["won"])
            for _, row in results.iterrows():
                if bool(row["won"]) == first_result:
                    streak += 1 if first_result else -1
                else:
                    break

        # Home/away splits (all history)
        home_games = results[results["is_home"]]
        away_games = results[~results["is_home"]]
        home_win_pct = float(home_games["won"].mean()) if len(home_games) > 0 else 0.5
        away_win_pct = float(away_games["won"].mean()) if len(away_games) > 0 else 0.5

        # Home/away splits (last 10 only)
        home10 = last10[last10["is_home"]]
        away10 = last10[~last10["is_home"]]
        home_win_pct_last10 = float(home10["won"].mean()) if len(home10) > 0 else 0.5
        away_win_pct_last10 = float(away10["won"].mean()) if len(away10) > 0 else 0.5

        return FormFactors(
            win_pct_last5=win_pct_last5,
            win_pct_last10=win_pct_last10,
            win_streak=streak,
            home_win_pct=home_win_pct,
            away_win_pct=away_win_pct,
            home_win_pct_last10=home_win_pct_last10,
            away_win_pct_last10=away_win_pct_last10,
            point_diff_last5=point_diff_last5,
            point_diff_last10=point_diff_last10,
        )

    def build_form_features(
        self,
        home_form: FormFactors,
        away_form: FormFactors,
    ) -> Dict[str, float]:
        """Build form-based feature dictionary for a game."""
        return {
            # Win% last 5
            "home_win_pct_l5": home_form.win_pct_last5,
            "away_win_pct_l5": away_form.win_pct_last5,
            "win_pct_l5_diff": home_form.win_pct_last5 - away_form.win_pct_last5,
            # Win% last 10
            "home_win_pct_l10": home_form.win_pct_last10,
            "away_win_pct_l10": away_form.win_pct_last10,
            "win_pct_l10_diff": home_form.win_pct_last10 - away_form.win_pct_last10,
            # Streaks
            "home_streak": float(home_form.win_streak),
            "away_streak": float(away_form.win_streak),
            "streak_diff": float(home_form.win_streak - away_form.win_streak),
            # Home/away situational splits
            "home_home_win_pct": home_form.home_win_pct,
            "away_away_win_pct": away_form.away_win_pct,
            "situational_split_diff": home_form.home_win_pct - away_form.away_win_pct,
            # Recent scoring margin
            "home_margin_l5": home_form.point_diff_last5,
            "away_margin_l5": away_form.point_diff_last5,
            "margin_l5_diff": home_form.point_diff_last5 - away_form.point_diff_last5,
            "home_margin_l10": home_form.point_diff_last10,
            "away_margin_l10": away_form.point_diff_last10,
            "margin_l10_diff": home_form.point_diff_last10 - away_form.point_diff_last10,
        }


class SituationalFeatureCalculator:
    """High-level calculator that computes all situational features for a games DataFrame.

    This is the primary interface used by training and prediction pipelines.
    """

    def __init__(self) -> None:
        self._sit = SituationalFeatures()
        self._form = FormCalculator()

    def compute_features_for_games(
        self,
        games: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute situational and form features for every game in the DataFrame.

        Uses only information available BEFORE each game (strict no look-ahead).
        Games are processed in chronological order.

        Args:
            games: DataFrame with columns:
                   game_datetime, home_team_id, away_team_id,
                   home_score, away_score (for completed games).

        Returns:
            DataFrame with all situational feature columns, aligned to input index.
        """
        if games.empty:
            return pd.DataFrame(index=games.index)

        games = games.copy()

        # Normalise datetime
        games["game_datetime"] = pd.to_datetime(games["game_datetime"], utc=True).dt.tz_localize(None)
        games = games.sort_values("game_datetime").reset_index(drop=False)

        feature_rows: List[Dict[str, Any]] = []

        for pos in range(len(games)):
            game = games.iloc[pos]
            game_date = game["game_datetime"]
            home_id = int(game["home_team_id"])
            away_id = int(game["away_team_id"])

            # All completed games strictly before this game
            past = games.iloc[:pos]
            past = past[past["home_score"].notna() & past["away_score"].notna()]

            # Filter to each team's past games
            home_past = past[
                (past["home_team_id"] == home_id) | (past["away_team_id"] == home_id)
            ]
            away_past = past[
                (past["home_team_id"] == away_id) | (past["away_team_id"] == away_id)
            ]

            # Schedule / fatigue features
            home_sit = self._sit.calculate_situational_factors(game_date, home_past, is_home=True)
            away_sit = self._sit.calculate_situational_factors(game_date, away_past, is_home=False)
            sit_feats = self._sit.build_situational_features(home_sit, away_sit)

            # Form features
            home_form = self._form.calculate_form(home_id, home_past)
            away_form = self._form.calculate_form(away_id, away_past)
            form_feats = self._form.build_form_features(home_form, away_form)

            feature_rows.append({**sit_feats, **form_feats})

        result = pd.DataFrame(feature_rows, index=games["index"])
        return result

    def compute_single_game_features(
        self,
        game_date: datetime,
        home_team_id: int,
        away_team_id: int,
        historical_games: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute situational features for a single upcoming game.

        Args:
            game_date: Date/time of the upcoming game.
            home_team_id: Home team ID.
            away_team_id: Away team ID.
            historical_games: All historical completed games.

        Returns:
            Feature dictionary ready for model input.
        """
        historical_games = historical_games.copy()
        historical_games["game_datetime"] = pd.to_datetime(
            historical_games["game_datetime"], utc=True
        ).dt.tz_localize(None)
        if hasattr(game_date, "tzinfo") and game_date.tzinfo is not None:
            game_date = game_date.replace(tzinfo=None)

        past = historical_games[
            (historical_games["game_datetime"] < game_date)
            & historical_games["home_score"].notna()
            & historical_games["away_score"].notna()
        ]

        home_past = past[
            (past["home_team_id"] == home_team_id) | (past["away_team_id"] == home_team_id)
        ]
        away_past = past[
            (past["home_team_id"] == away_team_id) | (past["away_team_id"] == away_team_id)
        ]

        home_sit = self._sit.calculate_situational_factors(game_date, home_past, is_home=True)
        away_sit = self._sit.calculate_situational_factors(game_date, away_past, is_home=False)
        sit_feats = self._sit.build_situational_features(home_sit, away_sit)

        home_form = self._form.calculate_form(home_team_id, home_past)
        away_form = self._form.calculate_form(away_team_id, away_past)
        form_feats = self._form.build_form_features(home_form, away_form)

        return {**sit_feats, **form_feats}
