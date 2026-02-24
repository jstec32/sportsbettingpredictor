#!/usr/bin/env python3
"""Capture current odds from The Odds API and store in database.

This script should be run regularly (e.g., every 30 minutes during game days)
to build up historical odds data for backtesting with real market probabilities.

Usage:
    python scripts/capture_odds.py --sport nba
    python scripts/capture_odds.py --sport nfl --sport nba  # Both sports

Scheduling:
    # Add to crontab to run every 30 minutes during typical game hours
    # NBA games usually 7pm-11pm ET
    # NFL games: Sunday 1pm-11pm ET, Monday/Thursday nights

    # Example crontab entries:
    # 30 18-23 * * * cd /path/to/SportsBettingPredictor && python scripts/capture_odds.py --sport nba
    # 0,30 13-23 * * 0 cd /path/to/SportsBettingPredictor && python scripts/capture_odds.py --sport nfl
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime

import click
from dotenv import load_dotenv
from loguru import logger

from src.data import OddsAPIClient, DatabaseStorage
from src.data.ingestion import DataIngestionPipeline
from src.utils.logging_config import setup_logging

load_dotenv()


@click.command()
@click.option(
    "--sport",
    type=click.Choice(["nfl", "nba"]),
    multiple=True,
    default=["nba"],
    help="Sport(s) to capture odds for (can specify multiple)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(sport: tuple[str, ...], verbose: bool) -> None:
    """Capture current odds and store in database."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")

    leagues = [s.upper() for s in sport]
    logger.info(f"Capturing odds for: {', '.join(leagues)}")

    asyncio.run(capture_odds(leagues))


async def capture_odds(leagues: list[str]) -> None:
    """Capture and store odds for specified leagues."""
    storage = DatabaseStorage()

    async with DataIngestionPipeline(storage=storage) as pipeline:
        if not pipeline.odds:
            logger.error("No Odds API client available. Check ODDS_API_KEY in .env")
            return

        for league in leagues:
            logger.info(f"Processing {league}...")

            try:
                # Ingest odds using the pipeline
                count = await pipeline.ingest_odds(league)
                logger.info(f"  Captured {count} odds records for {league}")

                # Log API usage
                remaining = await get_api_usage(pipeline.odds)
                if remaining is not None:
                    logger.info(f"  API requests remaining this month: {remaining}")

                    # Warn if running low
                    if remaining < 50:
                        logger.warning(f"  LOW API QUOTA: Only {remaining} requests left!")

            except Exception as e:
                logger.error(f"  Error capturing {league} odds: {e}")

    logger.info("Odds capture complete!")


async def get_api_usage(client: OddsAPIClient) -> int | None:
    """Get remaining API requests for the month."""
    try:
        # The Odds API returns usage in response headers
        # We track this in the api_requests table
        return client._requests_remaining
    except Exception:
        return None


if __name__ == "__main__":
    main()
