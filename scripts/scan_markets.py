#!/usr/bin/env python3
"""Scan markets for betting opportunities."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger

from src.data import KalshiClient, OddsAPIClient, ESPNClient, DatabaseStorage
from src.data.ingestion import DataIngestionPipeline
from src.market.edge_calculator import EdgeCalculator, Opportunity
from src.models import BaseModel
from src.utils.config import get_settings
from src.utils.logging_config import setup_logging

load_dotenv()


@click.command()
@click.option(
    "--sport",
    type=click.Choice(["nfl", "nba", "all"]),
    default="all",
    help="Sport to scan",
)
@click.option(
    "--min-edge",
    type=float,
    default=0.03,
    help="Minimum edge threshold",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.08,
    help="Minimum confidence threshold (default 0.08 = 54%/46% prediction)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file for opportunities",
)
@click.option(
    "--refresh-data",
    is_flag=True,
    help="Refresh market data before scanning",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    sport: str,
    min_edge: float,
    min_confidence: float,
    output: str | None,
    refresh_data: bool,
    verbose: bool,
) -> None:
    """Scan markets for betting opportunities."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    logger.info("Starting market scan...")

    settings = get_settings()

    asyncio.run(
        scan_markets(
            sport=sport,
            min_edge=min_edge,
            min_confidence=min_confidence,
            output=output,
            refresh_data=refresh_data,
            settings=settings,
        )
    )


async def scan_markets(
    sport: str,
    min_edge: float,
    min_confidence: float,
    output: str | None,
    refresh_data: bool,
    settings: any,
) -> None:
    """Scan markets for opportunities."""
    storage = DatabaseStorage()
    edge_calculator = EdgeCalculator(
        min_edge=min_edge,
        min_confidence=min_confidence,
    )

    # Determine leagues
    leagues = []
    if sport in ("nfl", "all"):
        leagues.append("NFL")
    if sport in ("nba", "all"):
        leagues.append("NBA")

    # Refresh data if requested
    if refresh_data:
        logger.info("Refreshing market data...")
        async with DataIngestionPipeline(storage=storage) as pipeline:
            for league in leagues:
                try:
                    await pipeline.ingest_games(league)
                    await pipeline.ingest_odds(league)
                except Exception as e:
                    logger.error(f"Error refreshing {league} data: {e}")

        # Also refresh Kalshi markets
        try:
            await pipeline.ingest_kalshi_markets(sport=sport)
        except Exception as e:
            logger.error(f"Error refreshing Kalshi data: {e}")

    # Load models
    models = {}
    for league in leagues:
        model_path = Path(f"models/{league.lower()}_ensemble.pkl")
        if model_path.exists():
            try:
                models[league] = BaseModel.load(str(model_path))
                logger.info(f"Loaded model for {league}")
            except Exception as e:
                logger.warning(f"Could not load model for {league}: {e}")

    # Get upcoming games
    all_opportunities = []

    for league in leagues:
        logger.info(f"Scanning {league}...")

        games = storage.get_upcoming_games(league=league, days_ahead=7)
        logger.info(f"  Found {len(games)} upcoming games")

        if not games:
            continue

        # Get market data for games
        markets = get_market_data(storage, [g["id"] for g in games])

        # Generate predictions if we have a model
        if league in models:
            predictions = generate_predictions(models[league], games)

            # Find opportunities
            opportunities = edge_calculator.find_opportunities(predictions, markets)
            opportunities = edge_calculator.filter_opportunities(
                opportunities,
                max_opportunities=10,
                max_per_game=1,
            )

            all_opportunities.extend(opportunities)
            logger.info(f"  Found {len(opportunities)} opportunities")
        else:
            logger.info(f"  No model available for {league}")

    # Sort all opportunities by edge
    all_opportunities.sort(key=lambda x: x.edge, reverse=True)

    # Display results
    if all_opportunities:
        logger.info("")
        logger.info("=" * 60)
        logger.info("TOP OPPORTUNITIES")
        logger.info("=" * 60)

        for i, opp in enumerate(all_opportunities[:10], 1):
            logger.info(
                f"{i}. {opp.home_team} vs {opp.away_team}"
            )
            logger.info(
                f"   Side: {opp.side.upper()} | "
                f"Edge: {opp.edge:.1%} | "
                f"Confidence: {opp.confidence:.1%}"
            )
            logger.info(
                f"   Model: {opp.model_prob:.1%} | "
                f"Market: {opp.market_prob:.1%} | "
                f"Kelly: {opp.recommended_stake_pct:.1%}"
            )
            logger.info("")

    else:
        logger.info("No opportunities found meeting criteria")

    # Save results
    if output or all_opportunities:
        output_path = Path(output) if output else Path(
            f"reports/opportunities_{datetime.now():%Y%m%d_%H%M}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                [opp.to_dict() for opp in all_opportunities],
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Results saved to {output_path}")


def get_market_data(
    storage: DatabaseStorage,
    game_ids: list[int],
) -> list[dict]:
    """Get market data for games."""
    if not game_ids:
        return []

    # Query latest odds for each game with team names
    query = """
        SELECT DISTINCT ON (o.game_id)
            o.game_id,
            o.home_prob,
            o.away_prob,
            o.spread_home,
            o.total_line,
            o.captured_at,
            g.game_datetime,
            ht.name as home_team,
            at.name as away_team
        FROM odds_history o
        JOIN games g ON o.game_id = g.id
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE o.game_id = ANY(:game_ids)
        ORDER BY o.game_id, o.captured_at DESC
    """

    try:
        result = storage.execute(query, {"game_ids": game_ids})
        return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return []


def generate_predictions(model: BaseModel, games: list[dict]) -> list:
    """Generate predictions for games."""
    from src.models.base import Prediction
    import pandas as pd

    predictions = []

    for game in games:
        features = {
            "home_team_id": game["home_team_id"],
            "away_team_id": game["away_team_id"],
            "is_neutral_site": game.get("is_neutral_site", False),
            "season": game.get("season", 2026),
        }

        try:
            pred = model.make_prediction(
                game_id=game["id"],
                features=features,
                prediction_type="moneyline",
            )
            predictions.append(pred)
        except Exception as e:
            logger.debug(f"Error generating prediction for game {game['id']}: {e}")

    return predictions


if __name__ == "__main__":
    main()
