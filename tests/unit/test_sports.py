"""Tests for ESPN API client."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from src.data.sports import ESPNClient, Team, Game, Injury


class TestESPNClient:
    """Tests for ESPNClient."""

    def test_init(self):
        """Test client initialization."""
        client = ESPNClient()
        assert client.BASE_URL == "https://site.api.espn.com/apis/site/v2/sports"

    def test_get_sport_league_nfl(self):
        """Test sport/league config for NFL."""
        client = ESPNClient()
        sport, league = client._get_sport_league("NFL")
        assert sport == "football"
        assert league == "nfl"

    def test_get_sport_league_nba(self):
        """Test sport/league config for NBA."""
        client = ESPNClient()
        sport, league = client._get_sport_league("NBA")
        assert sport == "basketball"
        assert league == "nba"

    def test_get_sport_league_invalid(self):
        """Test sport/league config for invalid league."""
        client = ESPNClient()
        with pytest.raises(ValueError):
            client._get_sport_league("INVALID")

    def test_parse_game(self):
        """Test game data parsing."""
        client = ESPNClient()

        event_data = {
            "id": "401547417",
            "date": "2024-09-05T20:20:00Z",
            "season": {"year": 2024, "type": 2},
            "week": {"number": 1},
            "status": {"type": {"name": "STATUS_FINAL"}},
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"id": "12", "displayName": "Kansas City Chiefs"},
                            "score": "27",
                        },
                        {
                            "homeAway": "away",
                            "team": {"id": "33", "displayName": "Baltimore Ravens"},
                            "score": "20",
                        },
                    ],
                    "venue": {"fullName": "Arrowhead Stadium"},
                    "neutralSite": False,
                }
            ],
        }

        game = client._parse_game(event_data, "NFL")

        assert isinstance(game, Game)
        assert game.external_id == "401547417"
        assert game.league == "NFL"
        assert game.season == 2024
        assert game.home_team_id == "12"
        assert game.away_team_id == "33"
        assert game.home_score == 27
        assert game.away_score == 20
        assert game.status == "final"

    def test_parse_game_scheduled(self):
        """Test parsing scheduled game."""
        client = ESPNClient()

        event_data = {
            "id": "401547500",
            "date": "2024-09-15T20:20:00Z",
            "season": {"year": 2024, "type": 2},
            "week": {"number": 2},
            "status": {"type": {"name": "STATUS_SCHEDULED"}},
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"id": "33", "displayName": "Baltimore Ravens"},
                        },
                        {
                            "homeAway": "away",
                            "team": {"id": "12", "displayName": "Kansas City Chiefs"},
                        },
                    ],
                    "neutralSite": False,
                }
            ],
        }

        game = client._parse_game(event_data, "NFL")

        assert game.status == "scheduled"
        assert game.home_score is None
        assert game.away_score is None

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = ESPNClient()

        with patch.object(client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"events": []}
            result = await client.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        client = ESPNClient()

        with patch.object(client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Connection error")
            result = await client.health_check()
            assert result is False


class TestTeam:
    """Tests for Team model."""

    def test_team_creation(self):
        """Test team model creation."""
        team = Team(
            external_id="12",
            league="NFL",
            name="Kansas City Chiefs",
            abbreviation="KC",
        )

        assert team.external_id == "12"
        assert team.league == "NFL"
        assert team.name == "Kansas City Chiefs"

    def test_team_with_optional_fields(self):
        """Test team model with optional fields."""
        team = Team(
            external_id="12",
            league="NFL",
            name="Kansas City Chiefs",
            abbreviation="KC",
            city="Kansas City",
            conference="AFC",
            division="West",
            logo_url="https://example.com/logo.png",
        )

        assert team.city == "Kansas City"
        assert team.conference == "AFC"
        assert team.division == "West"


class TestGame:
    """Tests for Game model."""

    def test_game_creation(self):
        """Test game model creation."""
        game = Game(
            external_id="401547417",
            league="NFL",
            season=2024,
            season_type="regular",
            game_datetime=datetime(2024, 9, 5, 20, 20, tzinfo=timezone.utc),
            home_team_id="12",
            away_team_id="33",
            home_team_name="Kansas City Chiefs",
            away_team_name="Baltimore Ravens",
            status="final",
        )

        assert game.external_id == "401547417"
        assert game.season == 2024
        assert game.home_team_id == "12"

    def test_game_with_scores(self):
        """Test game model with scores."""
        game = Game(
            external_id="401547417",
            league="NFL",
            season=2024,
            season_type="regular",
            game_datetime=datetime(2024, 9, 5, 20, 20, tzinfo=timezone.utc),
            home_team_id="12",
            away_team_id="33",
            home_team_name="Kansas City Chiefs",
            away_team_name="Baltimore Ravens",
            home_score=27,
            away_score=20,
            status="final",
        )

        assert game.home_score == 27
        assert game.away_score == 20


class TestInjury:
    """Tests for Injury model."""

    def test_injury_creation(self):
        """Test injury model creation."""
        injury = Injury(
            player_external_id="3139477",
            player_name="Patrick Mahomes",
            team_id="12",
            injury_status="probable",
        )

        assert injury.player_name == "Patrick Mahomes"
        assert injury.injury_status == "probable"

    def test_injury_with_details(self):
        """Test injury model with details."""
        injury = Injury(
            player_external_id="3139477",
            player_name="Patrick Mahomes",
            team_id="12",
            position="QB",
            injury_type="Ankle",
            injury_status="questionable",
        )

        assert injury.position == "QB"
        assert injury.injury_type == "Ankle"
