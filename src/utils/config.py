"""Configuration management using Pydantic settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql://localhost:5432/sports_betting",
        description="PostgreSQL connection string",
    )

    # APIs
    odds_api_key: str = Field(default="", description="The Odds API key")
    kalshi_email: str = Field(default="", description="Kalshi account email")
    kalshi_password: str = Field(default="", description="Kalshi account password")

    # Application
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Environment name")

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"


class DatabaseConfig:
    """Database configuration loaded from YAML."""

    def __init__(self, environment: str = "development") -> None:
        self.environment = environment
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load database configuration from YAML file."""
        config_path = PROJECT_ROOT / "config" / "database.yaml"
        if config_path.exists():
            with open(config_path) as f:
                all_config = yaml.safe_load(f)
                self._config = all_config.get(self.environment, {})

    @property
    def driver(self) -> str:
        return self._config.get("driver", "postgresql")

    @property
    def host(self) -> str:
        host = self._config.get("host", "localhost")
        return os.path.expandvars(host) if "$" in str(host) else host

    @property
    def port(self) -> int:
        port = self._config.get("port", 5432)
        return int(os.path.expandvars(str(port))) if "$" in str(port) else int(port)

    @property
    def database(self) -> str:
        return self._config.get("database", "sports_betting")

    @property
    def pool_size(self) -> int:
        return self._config.get("pool_size", 5)


class ModelConfig:
    """Model configuration loaded from YAML."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load model configuration from YAML file."""
        config_path = PROJECT_ROOT / "config" / "models.yaml"
        if config_path.exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f)

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get configuration for a specific model."""
        return self._config.get(model_name, {})

    @property
    def elo_config(self) -> dict[str, Any]:
        return self._config.get("elo", {})

    @property
    def regression_config(self) -> dict[str, Any]:
        return self._config.get("regression", {})

    @property
    def ensemble_config(self) -> dict[str, Any]:
        return self._config.get("ensemble", {})


class RiskConfig:
    """Risk management configuration loaded from YAML."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load risk configuration from YAML file."""
        config_path = PROJECT_ROOT / "config" / "risk_limits.yaml"
        if config_path.exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f)

    @property
    def initial_bankroll(self) -> float:
        return self._config.get("bankroll", {}).get("initial", 10000)

    @property
    def min_operational_bankroll(self) -> float:
        return self._config.get("bankroll", {}).get("min_operational", 5000)

    @property
    def kelly_fraction(self) -> float:
        return self._config.get("position_sizing", {}).get("kelly_fraction", 0.25)

    @property
    def max_bet_pct(self) -> float:
        return self._config.get("position_sizing", {}).get("max_bet_pct", 0.05)

    @property
    def min_bet_pct(self) -> float:
        return self._config.get("position_sizing", {}).get("min_bet_pct", 0.01)

    @property
    def min_edge_threshold(self) -> float:
        return self._config.get("position_sizing", {}).get("min_edge_threshold", 0.03)

    @property
    def max_total_exposure(self) -> float:
        return self._config.get("exposure_limits", {}).get("max_total_exposure", 0.20)

    @property
    def max_single_game(self) -> float:
        return self._config.get("exposure_limits", {}).get("max_single_game", 0.05)

    @property
    def max_daily_bets(self) -> int:
        return self._config.get("exposure_limits", {}).get("max_daily_bets", 10)

    @property
    def halt_threshold(self) -> float:
        return self._config.get("drawdown_limits", {}).get("halt_threshold", 0.20)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache()
def get_database_config(environment: str = "development") -> DatabaseConfig:
    """Get cached database config instance."""
    return DatabaseConfig(environment)


@lru_cache()
def get_model_config() -> ModelConfig:
    """Get cached model config instance."""
    return ModelConfig()


@lru_cache()
def get_risk_config() -> RiskConfig:
    """Get cached risk config instance."""
    return RiskConfig()
