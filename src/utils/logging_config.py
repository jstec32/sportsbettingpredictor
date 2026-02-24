"""Logging configuration using loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.config import get_settings


def setup_logging(
    log_level: str | None = None,
    log_file: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
                   Defaults to settings LOG_LEVEL.
        log_file: Path to log file. If None, only console logging.
        rotation: When to rotate log file (e.g., "10 MB", "1 day").
        retention: How long to keep old log files.
    """
    settings = get_settings()
    level = log_level or settings.log_level

    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="gz",
        )


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logger.bind(name=name)


class LogContext:
    """Context manager for adding context to log messages."""

    def __init__(self, **kwargs: Any) -> None:
        self.context = kwargs
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        self._token = logger.contextualize(**self.context)
        self._token.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token:
            self._token.__exit__(*args)
