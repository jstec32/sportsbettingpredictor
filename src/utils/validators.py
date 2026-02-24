"""Data validation utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_probability(value: float | Decimal, name: str = "probability") -> float:
    """Validate that a value is a valid probability (0 to 1).

    Args:
        value: The value to validate.
        name: Name of the field for error messages.

    Returns:
        The validated probability as a float.

    Raises:
        ValidationError: If value is not a valid probability.
    """
    try:
        prob = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number: {e}")

    if prob < 0 or prob > 1:
        raise ValidationError(f"{name} must be between 0 and 1, got {prob}")

    return prob


def validate_odds(
    value: float | Decimal,
    format: str = "decimal",
    name: str = "odds",
) -> float:
    """Validate odds value.

    Args:
        value: The odds value to validate.
        format: Odds format ('decimal', 'american', 'probability').
        name: Name of the field for error messages.

    Returns:
        The validated odds as a float.

    Raises:
        ValidationError: If value is not valid odds.
    """
    try:
        odds = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number: {e}")

    if format == "decimal":
        if odds < 1.0:
            raise ValidationError(f"Decimal {name} must be >= 1.0, got {odds}")
    elif format == "american":
        if odds == 0 or (-100 < odds < 100 and odds != 0):
            raise ValidationError(f"American {name} must be >= 100 or <= -100, got {odds}")
    elif format == "probability":
        return validate_probability(odds, name)

    return odds


def validate_edge(value: float | Decimal, name: str = "edge") -> float:
    """Validate edge value.

    Args:
        value: The edge value to validate.
        name: Name of the field for error messages.

    Returns:
        The validated edge as a float.

    Raises:
        ValidationError: If value is not a valid edge.
    """
    try:
        edge = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number: {e}")

    if edge < -1 or edge > 1:
        raise ValidationError(f"{name} must be between -1 and 1, got {edge}")

    return edge


def validate_stake(
    value: float | Decimal,
    min_stake: float = 0.01,
    max_stake: float | None = None,
    name: str = "stake",
) -> float:
    """Validate stake/bet amount.

    Args:
        value: The stake value to validate.
        min_stake: Minimum allowed stake.
        max_stake: Maximum allowed stake (optional).
        name: Name of the field for error messages.

    Returns:
        The validated stake as a float.

    Raises:
        ValidationError: If value is not a valid stake.
    """
    try:
        stake = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number: {e}")

    if stake < min_stake:
        raise ValidationError(f"{name} must be >= {min_stake}, got {stake}")

    if max_stake is not None and stake > max_stake:
        raise ValidationError(f"{name} must be <= {max_stake}, got {stake}")

    return stake


def validate_league(value: str, name: str = "league") -> str:
    """Validate league code.

    Args:
        value: The league code to validate.
        name: Name of the field for error messages.

    Returns:
        The validated league code (uppercase).

    Raises:
        ValidationError: If value is not a valid league.
    """
    valid_leagues = {"NFL", "NBA"}
    league = str(value).upper()

    if league not in valid_leagues:
        raise ValidationError(f"{name} must be one of {valid_leagues}, got {value}")

    return league


def validate_game_datetime(value: datetime | str, name: str = "game_datetime") -> datetime:
    """Validate game datetime.

    Args:
        value: The datetime to validate.
        name: Name of the field for error messages.

    Returns:
        The validated datetime (timezone-aware).

    Raises:
        ValidationError: If value is not a valid datetime.
    """
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValidationError(f"{name} must be a valid ISO datetime: {e}")
    elif isinstance(value, datetime):
        dt = value
    else:
        raise ValidationError(f"{name} must be a datetime or ISO string, got {type(value)}")

    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def validate_dict_keys(
    data: dict[str, Any],
    required_keys: set[str],
    optional_keys: set[str] | None = None,
    name: str = "data",
) -> None:
    """Validate that a dictionary has required keys.

    Args:
        data: The dictionary to validate.
        required_keys: Set of required key names.
        optional_keys: Set of optional key names (for strict mode).
        name: Name of the data for error messages.

    Raises:
        ValidationError: If required keys are missing.
    """
    if not isinstance(data, dict):
        raise ValidationError(f"{name} must be a dictionary, got {type(data)}")

    missing = required_keys - set(data.keys())
    if missing:
        raise ValidationError(f"{name} missing required keys: {missing}")

    if optional_keys is not None:
        allowed = required_keys | optional_keys
        extra = set(data.keys()) - allowed
        if extra:
            raise ValidationError(f"{name} has unexpected keys: {extra}")


def odds_to_probability(odds: float, format: str = "decimal") -> float:
    """Convert odds to implied probability.

    Args:
        odds: The odds value.
        format: Odds format ('decimal', 'american').

    Returns:
        Implied probability (0 to 1).
    """
    if format == "decimal":
        return 1.0 / odds
    elif format == "american":
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)
    else:
        raise ValueError(f"Unknown odds format: {format}")


def probability_to_odds(prob: float, format: str = "decimal") -> float:
    """Convert probability to odds.

    Args:
        prob: The probability (0 to 1).
        format: Desired odds format ('decimal', 'american').

    Returns:
        Odds in the specified format.
    """
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {prob}")

    if format == "decimal":
        return 1.0 / prob
    elif format == "american":
        if prob >= 0.5:
            return -100.0 * prob / (1.0 - prob)
        else:
            return 100.0 * (1.0 - prob) / prob
    else:
        raise ValueError(f"Unknown odds format: {format}")
