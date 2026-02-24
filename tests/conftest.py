"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import pytest
from datetime import datetime, timezone

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "postgresql://localhost:5432/sports_betting_test"


@pytest.fixture
def sample_team_data():
    """Sample team data for testing."""
    return {
        "external_id": "1",
        "league": "NFL",
        "name": "Kansas City Chiefs",
        "abbreviation": "KC",
        "city": "Kansas City",
        "conference": "AFC",
        "division": "West",
        "stadium_name": "Arrowhead Stadium",
        "stadium_lat": 39.0489,
        "stadium_lon": -94.4839,
        "is_dome": False,
        "timezone": "America/Chicago",
    }


@pytest.fixture
def sample_game_data():
    """Sample game data for testing."""
    return {
        "external_id": "401547417",
        "league": "NFL",
        "season": 2024,
        "season_type": "regular",
        "week": 1,
        "game_datetime": datetime(2024, 9, 5, 20, 20, tzinfo=timezone.utc),
        "home_team_id": 1,
        "away_team_id": 2,
        "home_score": 27,
        "away_score": 20,
        "status": "final",
        "venue_name": "Arrowhead Stadium",
        "venue_lat": 39.0489,
        "venue_lon": -94.4839,
        "is_neutral_site": False,
    }


@pytest.fixture
def sample_odds_data():
    """Sample odds data for testing."""
    return {
        "game_id": 1,
        "source": "consensus",
        "market_type": "h2h",
        "captured_at": datetime.now(timezone.utc),
        "home_odds": -150,
        "away_odds": 130,
        "home_prob": 0.60,
        "away_prob": 0.43,
        "spread_home": -3.5,
        "spread_away": 3.5,
        "total_line": 47.5,
        "over_odds": -110,
        "under_odds": -110,
    }


@pytest.fixture
def sample_prediction():
    """Sample prediction for testing."""
    from src.models.base import Prediction

    return Prediction(
        game_id=1,
        model_id="test_model",
        model_version="1.0.0",
        prediction_type="moneyline",
        predicted_at=datetime.now(timezone.utc),
        home_win_prob=0.65,
        away_win_prob=0.35,
        predicted_spread=-4.0,
        confidence=0.7,
        features={"elo_diff": 100, "home_advantage": 50},
    )


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "game_id": 1,
        "home_team": "Kansas City Chiefs",
        "away_team": "Baltimore Ravens",
        "game_datetime": datetime(2024, 9, 5, 20, 20, tzinfo=timezone.utc),
        "home_prob": 0.60,
        "away_prob": 0.40,
        "spread_home": -3.5,
        "total": 47.5,
    }


@pytest.fixture
def mock_elo_ratings():
    """Mock Elo ratings for testing."""
    return {
        1: 1550,  # Team 1 - above average
        2: 1500,  # Team 2 - average
        3: 1450,  # Team 3 - below average
        4: 1600,  # Team 4 - strong
        5: 1400,  # Team 5 - weak
    }


@pytest.fixture
def sample_games_dataframe():
    """Sample games DataFrame for testing."""
    import pandas as pd

    data = [
        {
            "home_team_id": 1,
            "away_team_id": 2,
            "home_score": 24,
            "away_score": 17,
            "is_neutral_site": False,
            "league": "NFL",
            "game_datetime": datetime(2024, 9, 8, 13, 0, tzinfo=timezone.utc),
        },
        {
            "home_team_id": 3,
            "away_team_id": 4,
            "home_score": 14,
            "away_score": 28,
            "is_neutral_site": False,
            "league": "NFL",
            "game_datetime": datetime(2024, 9, 8, 16, 25, tzinfo=timezone.utc),
        },
        {
            "home_team_id": 2,
            "away_team_id": 1,
            "home_score": 21,
            "away_score": 21,
            "is_neutral_site": False,
            "league": "NFL",
            "game_datetime": datetime(2024, 9, 15, 13, 0, tzinfo=timezone.utc),
        },
    ]

    df = pd.DataFrame(data)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    return df
