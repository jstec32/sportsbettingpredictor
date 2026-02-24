#!/usr/bin/env python3
"""Backfill historical data from APIs."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime, timedelta

import click
from dotenv import load_dotenv
from loguru import logger

from src.data import ESPNClient, OddsAPIClient, DatabaseStorage
from src.data.ingestion import DataIngestionPipeline
from src.utils.logging_config import setup_logging

load_dotenv()


# Season date ranges for NBA and NFL
SEASON_DATES = {
    "NBA": {
        2024: (datetime(2023, 10, 24), datetime(2024, 6, 17)),  # 2023-24 season
        2025: (datetime(2024, 10, 22), datetime(2025, 6, 22)),  # 2024-25 season
        2026: (datetime(2025, 10, 28), datetime(2026, 6, 21)),  # 2025-26 season
    },
    "NFL": {
        2023: (datetime(2023, 9, 7), datetime(2024, 2, 11)),   # 2023 season
        2024: (datetime(2024, 9, 5), datetime(2025, 2, 9)),    # 2024 season
        2025: (datetime(2025, 9, 4), datetime(2026, 2, 8)),    # 2025 season
    },
}


@click.command()
@click.option(
    "--sport",
    type=click.Choice(["nfl", "nba", "all"]),
    default="all",
    help="Sport to backfill",
)
@click.option(
    "--seasons",
    default="2024",
    help="Comma-separated seasons to backfill (e.g., 2023,2024)",
)
@click.option(
    "--skip-teams",
    is_flag=True,
    help="Skip team data backfill",
)
@click.option(
    "--skip-games",
    is_flag=True,
    help="Skip game data backfill",
)
@click.option(
    "--skip-odds",
    is_flag=True,
    help="Skip odds data backfill",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    sport: str,
    seasons: str,
    skip_teams: bool,
    skip_games: bool,
    skip_odds: bool,
    verbose: bool,
) -> None:
    """Backfill historical sports data."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    logger.info("Starting data backfill...")

    leagues = []
    if sport in ("nfl", "all"):
        leagues.append("NFL")
    if sport in ("nba", "all"):
        leagues.append("NBA")

    season_list = [int(s.strip()) for s in seasons.split(",")]

    asyncio.run(
        run_backfill(
            leagues=leagues,
            seasons=season_list,
            skip_teams=skip_teams,
            skip_games=skip_games,
            skip_odds=skip_odds,
        )
    )


async def run_backfill(
    leagues: list[str],
    seasons: list[int],
    skip_teams: bool,
    skip_games: bool,
    skip_odds: bool,
) -> None:
    """Run the backfill process."""
    storage = DatabaseStorage()

    async with DataIngestionPipeline(storage=storage) as pipeline:
        for league in leagues:
            logger.info(f"Processing {league}...")

            # Teams
            if not skip_teams:
                count = await pipeline.ingest_teams(league)
                logger.info(f"  Teams: {count}")

            # Games for each season
            if not skip_games:
                for season in seasons:
                    logger.info(f"  Season {season}...")
                    try:
                        # Get date range for the season
                        if league in SEASON_DATES and season in SEASON_DATES[league]:
                            start_date, end_date = SEASON_DATES[league][season]
                            # Don't fetch future games
                            now = datetime.now()
                            if end_date > now:
                                end_date = now - timedelta(days=1)
                            if start_date > now:
                                logger.info(f"    Season hasn't started yet")
                                continue

                            logger.info(f"    Fetching games from {start_date.date()} to {end_date.date()}...")
                            games = await pipeline.espn.get_games_by_date_range(
                                league, start_date, end_date
                            )
                        else:
                            # Try the old method as fallback
                            games = await pipeline.espn.get_schedule(league, season)

                        game_count = 0

                        for game in games:
                            # Look up team IDs (pass league to avoid cross-sport conflicts)
                            home_team = storage.get_team_by_external_id(game.home_team_id, league)
                            away_team = storage.get_team_by_external_id(game.away_team_id, league)

                            if not home_team or not away_team:
                                continue

                            game_data = {
                                "external_id": game.external_id,
                                "league": game.league,
                                "season": game.season,
                                "season_type": game.season_type,
                                "week": game.week,
                                "game_datetime": game.game_datetime,
                                "home_team_id": home_team["id"],
                                "away_team_id": away_team["id"],
                                "home_score": game.home_score,
                                "away_score": game.away_score,
                                "status": game.status,
                                "venue_name": game.venue_name,
                                "venue_lat": None,
                                "venue_lon": None,
                                "is_neutral_site": game.is_neutral_site,
                            }
                            storage.upsert_game(game_data)
                            game_count += 1

                        logger.info(f"    Games: {game_count}")

                    except Exception as e:
                        logger.error(f"    Error: {e}")

            # Odds (be careful with rate limits!)
            if not skip_odds and pipeline.odds:
                logger.info(f"  Fetching current odds...")
                try:
                    count = await pipeline.ingest_odds(league)
                    logger.info(f"    Odds records: {count}")
                except Exception as e:
                    logger.error(f"    Odds error: {e}")

    logger.info("Backfill complete!")


if __name__ == "__main__":
    main()
