#!/usr/bin/env python3
"""Settle completed games by pulling final scores from ESPN.

Calls ingest_games() for yesterday and today so that any games that have
finished get updated from status='scheduled' → status='final' with scores.
Without this, the training dataset doesn't grow and real odds can't be
joined to completed games for backtesting.

Usage:
    python scripts/settle_scores.py               # yesterday + today, NBA
    python scripts/settle_scores.py --days 2      # last 2 days
    python scripts/settle_scores.py --league all  # NBA + NFL

Cron (2 AM daily — catches last night's final scores):
    0 2 * * * cd /path/to/SportsBettingPredictor && .venv/bin/python scripts/settle_scores.py >> logs/settle.log 2>&1
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
    help="League to settle",
)
@click.option(
    "--days",
    type=int,
    default=2,
    help="Number of past days to settle (default: 2 = yesterday + today)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(league: str, days: int, verbose: bool) -> None:
    """Settle completed games by pulling final scores from ESPN."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    leagues = ["NFL", "NBA"] if league == "all" else [league.upper()]
    logger.info(f"Settling scores for {', '.join(leagues)} (last {days} days)...")
    asyncio.run(settle_scores(leagues, days))


async def settle_scores(leagues: list[str], days: int) -> None:
    """Pull final scores for recent games and update DB."""
    storage = DatabaseStorage()

    async with DataIngestionPipeline(storage=storage) as pipeline:
        for league in leagues:
            total_updated = 0
            today = datetime.now(timezone.utc).date()

            # Work backwards: today, yesterday, day before, ...
            for i in range(days):
                target_date = today - timedelta(days=i)
                date_str = target_date.strftime("%Y%m%d")

                try:
                    count = await pipeline.ingest_games(league, date=date_str)
                    if count:
                        logger.info(f"  {league} {target_date}: {count} games settled")
                        total_updated += count
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"  {league} {target_date}: {e}")

            logger.info(f"{league}: processed {total_updated} game records")

    # Report on what settled
    rows = storage.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'final')           AS final,
            COUNT(*) FILTER (WHERE status = 'scheduled')       AS scheduled,
            COUNT(*) FILTER (WHERE status = 'in_progress')     AS in_progress
        FROM games
        WHERE league = 'NBA' AND season = 2026
        """
    )
    if rows:
        r = rows[0]
        logger.info(
            f"NBA 2026 game states — final: {r[0]}, scheduled: {r[1]}, in_progress: {r[2]}"
        )

    # Report on games with real odds that are now final (usable for backtesting)
    usable = storage.execute(
        """
        SELECT COUNT(DISTINCT oh.game_id)
        FROM odds_history oh
        JOIN games g ON oh.game_id = g.id
        WHERE g.status = 'final'
          AND g.league = 'NBA'
          AND oh.game_id IS NOT NULL
        """
    )
    if usable:
        logger.info(f"NBA games with real odds + final score (backtest-ready): {usable[0][0]}")


if __name__ == "__main__":
    main()
