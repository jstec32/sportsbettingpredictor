"""Tests for feature engineering modules."""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.features.team_metrics import TeamMetricsCalculator, EloRating
from src.features.player_metrics import PlayerMetricsCalculator, POSITION_WEIGHTS
from src.features.situational import SituationalFeatures
from src.features.market_features import MarketFeatures


class TestTeamMetricsCalculator:
    """Tests for TeamMetricsCalculator."""

    def test_init(self):
        """Test calculator initialization."""
        calc = TeamMetricsCalculator()
        assert calc.initial_elo == 1500
        assert calc.k_factor == 20

    def test_get_elo_new_team(self):
        """Test getting Elo for new team."""
        calc = TeamMetricsCalculator()
        rating = calc.get_elo(999)
        assert rating == calc.initial_elo

    def test_expected_score_equal_ratings(self):
        """Test expected score with equal ratings."""
        calc = TeamMetricsCalculator()
        expected = calc.expected_score(1500, 1500)
        assert expected == pytest.approx(0.5, abs=0.001)

    def test_expected_score_favored(self):
        """Test expected score when one team is favored."""
        calc = TeamMetricsCalculator()

        # 100 point advantage
        expected = calc.expected_score(1600, 1500)
        assert expected > 0.5
        assert expected < 1.0
        assert expected == pytest.approx(0.64, abs=0.02)

    def test_expected_score_underdog(self):
        """Test expected score for underdog."""
        calc = TeamMetricsCalculator()

        # 100 point disadvantage
        expected = calc.expected_score(1400, 1500)
        assert expected < 0.5
        assert expected > 0.0

    def test_update_elo_win(self):
        """Test Elo update after win."""
        calc = TeamMetricsCalculator()

        initial_rating = calc.get_elo(1)
        new_rating = calc.update_elo(1, 1500, actual_score=1.0)

        assert new_rating > initial_rating

    def test_update_elo_loss(self):
        """Test Elo update after loss."""
        calc = TeamMetricsCalculator()

        # Set initial rating
        calc.update_elo(1, 1500, actual_score=1.0)
        rating_after_win = calc.get_elo(1)

        # Now lose
        new_rating = calc.update_elo(1, 1500, actual_score=0.0)

        assert new_rating < rating_after_win

    def test_calculate_win_probability(self):
        """Test win probability calculation."""
        calc = TeamMetricsCalculator()

        # Set up ratings
        calc._elo_ratings[1] = EloRating(1, 1550, 10, datetime.now())
        calc._elo_ratings[2] = EloRating(2, 1450, 10, datetime.now())

        prob = calc.calculate_win_probability(1, 2, "NFL")

        # Home team with higher rating + home advantage should be favored
        assert prob > 0.5

    def test_build_features(self):
        """Test feature building."""
        calc = TeamMetricsCalculator()

        # Set up ratings
        calc._elo_ratings[1] = EloRating(1, 1550, 10, datetime.now())
        calc._elo_ratings[2] = EloRating(2, 1450, 10, datetime.now())

        features = calc.build_features(1, 2, "NFL", is_neutral=False)

        assert "home_elo" in features
        assert "away_elo" in features
        assert "elo_diff" in features
        assert features["home_elo"] == 1550
        assert features["away_elo"] == 1450
        assert features["elo_diff"] == 100

    def test_regress_to_mean(self):
        """Test Elo regression to mean."""
        calc = TeamMetricsCalculator()

        # Set up a high rating
        calc._elo_ratings[1] = EloRating(1, 1700, 50, datetime.now())

        new_rating = calc.regress_to_mean(1, regression_factor=0.33)

        # Should move toward 1500
        assert new_rating < 1700
        assert new_rating > 1500


class TestPlayerMetricsCalculator:
    """Tests for PlayerMetricsCalculator."""

    def test_get_position_weight_qb(self):
        """Test QB position weight."""
        calc = PlayerMetricsCalculator()
        weight = calc.get_position_weight("QB", "NFL")
        assert weight == 1.0  # QB is most important

    def test_get_position_weight_default(self):
        """Test default position weight."""
        calc = PlayerMetricsCalculator()
        weight = calc.get_position_weight("UNKNOWN", "NFL")
        assert weight == 0.15  # Default weight for unknown positions

    def test_get_injury_severity_out(self):
        """Test injury severity for 'out' status."""
        calc = PlayerMetricsCalculator()
        severity = calc.get_injury_severity("out")
        assert severity == 1.0

    def test_get_injury_severity_questionable(self):
        """Test injury severity for 'questionable' status."""
        calc = PlayerMetricsCalculator()
        severity = calc.get_injury_severity("questionable")
        assert severity == 0.5

    def test_calculate_injury_impact(self):
        """Test injury impact calculation."""
        calc = PlayerMetricsCalculator()

        injuries = [
            {"position": "QB", "injury_status": "out", "player_name": "Test QB"},
            {"position": "WR", "injury_status": "questionable", "player_name": "Test WR"},
        ]

        impact = calc.calculate_injury_impact(injuries, 1, "NFL")

        assert impact.total_impact > 0
        assert len(impact.key_players_out) > 0

    def test_build_injury_features(self):
        """Test injury feature building."""
        calc = PlayerMetricsCalculator()

        home_injuries = [
            {"position": "WR", "injury_status": "questionable", "player_name": "WR1"},
        ]
        away_injuries = [
            {"position": "QB", "injury_status": "out", "player_name": "QB1"},
        ]

        features = calc.build_injury_features(
            home_injuries, away_injuries, 1, 2, "NFL"
        )

        assert "home_injury_impact" in features
        assert "away_injury_impact" in features
        assert "injury_impact_diff" in features
        assert features["away_qb_out"] == 1.0
        assert features["home_qb_out"] == 0.0


class TestSituationalFeatures:
    """Tests for SituationalFeatures."""

    def test_haversine_distance(self):
        """Test distance calculation."""
        sf = SituationalFeatures()

        # NYC to LA is approximately 2450 miles
        distance = sf.haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)

        assert distance == pytest.approx(2450, rel=0.05)

    def test_calculate_rest_days(self):
        """Test rest days calculation."""
        sf = SituationalFeatures()

        game_date = datetime(2024, 9, 8, 13, 0, tzinfo=timezone.utc)
        last_game = datetime(2024, 9, 1, 13, 0, tzinfo=timezone.utc)

        rest = sf.calculate_rest_days(game_date, last_game)

        assert rest == 6  # 7 days minus game day

    def test_calculate_rest_days_none(self):
        """Test rest days with no previous game."""
        sf = SituationalFeatures()

        game_date = datetime(2024, 9, 8, 13, 0, tzinfo=timezone.utc)
        rest = sf.calculate_rest_days(game_date, None)

        assert rest == 7  # Default well-rested

    def test_build_situational_features(self):
        """Test situational feature building."""
        sf = SituationalFeatures()

        from src.features.situational import SituationalFactors

        home_factors = SituationalFactors(
            rest_days=6,
            travel_miles=0,
            time_zone_change=0,
            is_back_to_back=False,
            is_second_of_back_to_back=False,
            games_in_last_7_days=1,
            games_in_last_14_days=2,
        )

        away_factors = SituationalFactors(
            rest_days=3,
            travel_miles=1500,
            time_zone_change=2,
            is_back_to_back=False,
            is_second_of_back_to_back=False,
            games_in_last_7_days=2,
            games_in_last_14_days=4,
        )

        features = sf.build_situational_features(
            home_factors, away_factors, travel_miles=1500, tz_change=2
        )

        assert features["home_rest_days"] == 6
        assert features["away_rest_days"] == 3
        assert features["rest_days_diff"] == 3
        assert features["travel_miles"] == 1500


class TestMarketFeatures:
    """Tests for MarketFeatures."""

    def test_implied_probability_positive(self):
        """Test implied probability for positive American odds."""
        mf = MarketFeatures()
        prob = mf.calculate_implied_probability(150)
        assert prob == pytest.approx(0.4, abs=0.01)

    def test_implied_probability_negative(self):
        """Test implied probability for negative American odds."""
        mf = MarketFeatures()
        prob = mf.calculate_implied_probability(-150)
        assert prob == pytest.approx(0.6, abs=0.01)

    def test_remove_vig(self):
        """Test vig removal."""
        mf = MarketFeatures()

        # Typical sportsbook line with vig
        home_prob = 0.525  # -110
        away_prob = 0.525  # -110

        true_home, true_away = mf.remove_vig(home_prob, away_prob)

        assert true_home == pytest.approx(0.5, abs=0.01)
        assert true_away == pytest.approx(0.5, abs=0.01)
        assert true_home + true_away == pytest.approx(1.0, abs=0.001)

    def test_spread_to_probability(self):
        """Test spread to probability conversion."""
        mf = MarketFeatures()

        # 7 point favorite (adjusting for home advantage)
        prob = mf.spread_to_probability(-7, home_advantage=3)

        # Should be a moderate favorite (spread-based probability calculation)
        assert prob > 0.3  # Reasonable probability range for spread conversion

    def test_build_market_features(self):
        """Test market feature building."""
        mf = MarketFeatures()

        from src.features.market_features import LineMovement, MarketConsensus

        line_movement = LineMovement(
            game_id=1,
            opening_line=-3.0,
            current_line=-4.5,
            movement=-1.5,
            movement_direction="toward_home",
            steam_move=False,
            reverse_line_movement=False,
        )

        consensus = MarketConsensus(
            game_id=1,
            consensus_home_prob=0.62,
            consensus_spread=-4.5,
            consensus_total=47.5,
            sharp_money_side=None,
            public_money_side=None,
        )

        features = mf.build_market_features(
            line_movement, consensus, kalshi_yes_price=0.60
        )

        assert features["opening_spread"] == -3.0
        assert features["current_spread"] == -4.5
        assert features["spread_movement"] == -1.5
        assert features["consensus_home_prob"] == 0.62
        assert features["kalshi_home_prob"] == 0.60
