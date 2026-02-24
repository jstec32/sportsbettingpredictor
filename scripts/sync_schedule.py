#!/usr/bin/env python3
"""Sync upcoming game schedule from ESPN into the database.

Run this daily (or before capturing odds) so that capture_odds.py can match
upcoming games by team name and store odds with a valid game_id.

Without scheduled game rows in the DB, odds records are stored with game_id=NULL
and cannot be joined to games for backtesting or live edge detection.

Usage:
    python scripts/sync_schedule.py                  # next 7 days, NBA
    python scripts/sync_schedule.py --days 3         # next 3 days
    python scripts/sync_schedule.py --league nfl     # NFL schedule
    python scripts/sync_schedule.py --league all     # both leagues
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime, timedelta, timezone

import click
from dotenv import load_dotenv
from loguru import logger

from src.data.ingestion import DataIngestionPipeline
from src.data.storage import DatabaseStorage
from src.utils.logging_config import setup_logging

load_dotenv()


@click.command()
@click.option(
    "--league",
    type=click.Choice(["nfl", "nba", "all"]),
    default="nba",
    help="League to sync",
)
@click.option(
    "--days",
    type=int,
    default=7,
    help="Number of days ahead to sync (default: 7)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(league: str, days: int, verbose: bool) -> None:
    """Sync upcoming game schedule from ESPN into the database."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    leagues = ["NFL", "NBA"] if league == "all" else [league.upper()]
    logger.info(f"Syncing {', '.join(leagues)} schedule for next {days} days...")
    asyncio.run(sync_schedule(leagues, days))


async def sync_schedule(leagues: list[str], days: int) -> None:
    """Sync upcoming games for each league."""
    storage = DatabaseStorage()

    async with DataIngestionPipeline(storage=storage) as pipeline:
        for league in leagues:
            total = 0
            today = datetime.now(timezone.utc).date()

            for i in range(days):
                target_date = today + timedelta(days=i)
                date_str = target_date.strftime("%Y%m%d")

                try:
                    count = await pipeline.ingest_games(league, date=date_str)
                    if count:
                        logger.info(f"  {league} {target_date}: {count} games synced")
                        total += count
                    # Brief delay to be respectful to ESPN's unofficial API
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"  {league} {target_date}: {e}")

            logger.info(f"{league}: {total} total game records synced over {days} days")

    # Show what's now upcoming in the DB
    upcoming = storage.execute(
        """
        SELECT league, game_datetime::date AS date, COUNT(*) AS games
        FROM games
        WHERE status IN ('scheduled', 'in_progress')
          AND game_datetime >= NOW()
          AND game_datetime < NOW() + INTERVAL '8 days'
        GROUP BY league, game_datetime::date
        ORDER BY league, date
        """
    )
    if upcoming:
        logger.info("Upcoming games now in DB:")
        for row in upcoming:
            logger.info(f"  {row[0]} {row[1]}: {row[2]} games")
    else:
        logger.warning("No upcoming games found in DB after sync — check ESPN API response")


if __name__ == "__main__":
    main()
