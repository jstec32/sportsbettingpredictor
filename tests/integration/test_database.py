"""Integration tests for database operations."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


class TestDatabaseStorageIntegration:
    """Integration tests for DatabaseStorage.

    Note: These tests use mocking to avoid requiring a real database.
    For full integration tests, run with a test database.
    """

    def test_parse_database_url(self):
        """Test database URL parsing."""
        from scripts.setup_db import parse_database_url

        url = "postgresql://user:password@localhost:5432/sports_betting"
        config = parse_database_url(url)

        assert config["user"] == "user"
        assert config["password"] == "password"
        assert config["host"] == "localhost"
        assert config["port"] == "5432"
        assert config["database"] == "sports_betting"

    def test_parse_database_url_no_password(self):
        """Test database URL parsing without password."""
        from scripts.setup_db import parse_database_url

        url = "postgresql://user@localhost:5432/sports_betting"
        config = parse_database_url(url)

        assert config["user"] == "user"
        assert config["password"] == ""

    def test_storage_initialization(self):
        """Test storage initialization."""
        with patch.dict('os.environ', {'DATABASE_URL': 'postgresql://test:test@localhost:5432/test'}):
            from src.data.storage import DatabaseStorage
            storage = DatabaseStorage()
            assert storage.database_url == "postgresql://test:test@localhost:5432/test"

    def test_team_data_serialization(self, sample_team_data):
        """Test team data can be serialized for database."""
        import json

        # Ensure all values are serializable
        serialized = json.dumps(sample_team_data)
        deserialized = json.loads(serialized)

        assert deserialized["external_id"] == sample_team_data["external_id"]
        assert deserialized["name"] == sample_team_data["name"]

    def test_game_data_serialization(self, sample_game_data):
        """Test game data can be serialized for database."""
        import json

        # Convert datetime for serialization
        data = sample_game_data.copy()
        data["game_datetime"] = data["game_datetime"].isoformat()

        serialized = json.dumps(data)
        deserialized = json.loads(serialized)

        assert deserialized["external_id"] == sample_game_data["external_id"]

    def test_prediction_to_dict(self, sample_prediction):
        """Test prediction to_dict method."""
        data = sample_prediction.to_dict()

        assert data["game_id"] == sample_prediction.game_id
        assert data["model_id"] == sample_prediction.model_id
        assert data["home_win_prob"] == sample_prediction.home_win_prob
        assert "features_json" in data


class TestSchemaValidation:
    """Tests for database schema validation."""

    def test_schema_file_exists(self):
        """Test that schema file exists."""
        from pathlib import Path

        schema_path = Path("scripts/schema.sql")
        assert schema_path.exists()

    def test_schema_contains_required_tables(self):
        """Test that schema contains all required tables."""
        from pathlib import Path

        schema_path = Path("scripts/schema.sql")
        schema_sql = schema_path.read_text()

        required_tables = [
            "teams",
            "games",
            "kalshi_markets",
            "odds_history",
            "injuries",
            "weather_conditions",
            "predictions",
            "bets",
            "portfolio_snapshots",
            "api_requests",
        ]

        for table in required_tables:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in schema_sql, f"Missing table: {table}"

    def test_schema_contains_indexes(self):
        """Test that schema contains indexes."""
        from pathlib import Path

        schema_path = Path("scripts/schema.sql")
        schema_sql = schema_path.read_text()

        assert "CREATE INDEX" in schema_sql
        assert "idx_games_datetime" in schema_sql
        assert "idx_bets_status" in schema_sql

    def test_schema_contains_triggers(self):
        """Test that schema contains update triggers."""
        from pathlib import Path

        schema_path = Path("scripts/schema.sql")
        schema_sql = schema_path.read_text()

        assert "CREATE OR REPLACE FUNCTION update_updated_at_column" in schema_sql
        assert "CREATE TRIGGER" in schema_sql
