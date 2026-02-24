from __future__ import annotations

#!/usr/bin/env python3
"""Paper trading simulation."""

import asyncio
from datetime import datetime, timezone

import click
from dotenv import load_dotenv
from loguru import logger

from src.data.storage import DatabaseStorage
from src.execution.paper_trader import PaperTrader
from src.execution.position_sizer import PositionSizer
from src.market.edge_calculator import EdgeCalculator
from src.risk.limits import RiskLimits
from src.utils.logging_config import setup_logging

load_dotenv()


@click.command()
@click.option(
    "--bankroll",
    type=float,
    default=10000,
    help="Initial paper trading bankroll",
)
@click.option(
    "--min-edge",
    type=float,
    default=0.03,
    help="Minimum edge to place bets",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show opportunities without placing bets",
)
@click.option(
    "--settle",
    is_flag=True,
    help="Settle open positions based on game results",
)
@click.option(
    "--status",
    is_flag=True,
    help="Show current paper trading status",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    bankroll: float,
    min_edge: float,
    dry_run: bool,
    settle: bool,
    status: bool,
    verbose: bool,
) -> None:
    """Run paper trading simulation."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")

    storage = DatabaseStorage()
    paper_trader = PaperTrader(initial_bankroll=bankroll, storage=storage)
    edge_calculator = EdgeCalculator(min_edge=min_edge)
    position_sizer = PositionSizer()
    risk_limits = RiskLimits()

    if status:
        show_status(paper_trader, storage)
        return

    if settle:
        settle_positions(paper_trader, storage)
        return

    # Find and evaluate opportunities
    logger.info("Scanning for opportunities...")

    opportunities = asyncio.run(find_opportunities(storage, edge_calculator))

    if not opportunities:
        logger.info("No opportunities found meeting criteria")
        return

    logger.info(f"Found {len(opportunities)} opportunities")

    # Check risk limits
    can_bet, reason = risk_limits.can_place_bet(
        bankroll=paper_trader.bankroll,
        total_exposure=paper_trader.total_exposure,
        bet_size=0,
        daily_bets=len(paper_trader.open_positions),
        current_drawdown=0,
    )

    if not can_bet:
        logger.warning(f"Cannot place bets: {reason}")
        return

    # Display opportunities
    for i, opp in enumerate(opportunities[:10], 1):
        logger.info(
            f"{i}. {opp.home_team} vs {opp.away_team} - "
            f"{opp.side.upper()} | Edge: {opp.edge:.1%} | "
            f"Kelly: {opp.recommended_stake_pct:.1%}"
        )

    if dry_run:
        logger.info("Dry run - no bets placed")
        return

    # Place paper bets
    for opp in opportunities[:5]:  # Limit to top 5
        # Size position
        size = position_sizer.size_position(
            edge=opp.edge,
            confidence=opp.confidence,
            bankroll=paper_trader.bankroll,
            decimal_odds=1 / opp.market_prob if opp.market_prob > 0 else 2.0,
            current_exposure=paper_trader.total_exposure,
        )

        if size.position_size > 0:
            position = paper_trader.open_position(opp, stake=size.position_size)
            if position:
                logger.info(
                    f"Opened position: {opp.side} on game {opp.game_id} "
                    f"for ${size.position_size:.2f}"
                )

    # Show updated status
    show_status(paper_trader, storage)


def show_status(paper_trader: PaperTrader, storage: DatabaseStorage) -> None:
    """Show paper trading status."""
    stats = paper_trader.get_stats()

    logger.info("=" * 50)
    logger.info("Paper Trading Status")
    logger.info("=" * 50)
    logger.info(f"Bankroll: ${stats['bankroll']:,.2f}")
    logger.info(f"Total PnL: ${stats['total_pnl']:+,.2f} ({stats['roi']:+.1%})")
    logger.info(f"Open Positions: {stats['open_positions']}")
    logger.info(f"Settled: {stats['settled_positions']}")

    if stats['settled_positions'] > 0:
        logger.info(f"Win Rate: {stats['win_rate']:.1%}")
        logger.info(f"Avg PnL: ${stats['avg_pnl']:+.2f}")
        logger.info(f"Avg Edge: {stats['avg_edge']:.1%}")

    # Show open positions
    if paper_trader.open_positions:
        logger.info("")
        logger.info("Open Positions:")
        for pos in paper_trader.open_positions:
            logger.info(
                f"  #{pos.position_id}: {pos.side} ${pos.stake:.2f} @ {pos.odds_at_entry:.2f}"
            )


def settle_positions(paper_trader: PaperTrader, storage: DatabaseStorage) -> None:
    """Settle open positions based on game results."""
    logger.info("Settling open positions...")

    open_positions = paper_trader.open_positions

    if not open_positions:
        logger.info("No open positions to settle")
        return

    # Get game results for open positions
    game_ids = list(set(p.game_id for p in open_positions))

    for game_id in game_ids:
        # Query game result
        query = """
            SELECT home_score, away_score, status
            FROM games
            WHERE id = :game_id
        """
        result = storage.execute(query, {"game_id": game_id})

        if not result:
            continue

        row = result[0]
        if row._mapping["status"] != "final":
            logger.debug(f"Game {game_id} not yet final")
            continue

        home_score = row._mapping["home_score"]
        away_score = row._mapping["away_score"]

        if home_score is None or away_score is None:
            continue

        home_won = home_score > away_score
        settled = paper_trader.settle_by_game(game_id, home_won)

        for pos in settled:
            logger.info(
                f"Settled position #{pos.position_id}: "
                f"{'WON' if pos.status == 'won' else 'LOST'} "
                f"PnL=${pos.pnl:.2f}"
            )

    show_status(paper_trader, storage)


async def find_opportunities(
    storage: DatabaseStorage,
    edge_calculator: EdgeCalculator,
) -> list:
    """Find betting opportunities from current data."""
    # This would integrate with models and market data
    # For now, return empty list
    logger.debug("Finding opportunities from model predictions and market data...")

    # In a full implementation, this would:
    # 1. Load trained models
    # 2. Get upcoming games
    # 3. Generate predictions
    # 4. Compare to market odds
    # 5. Find edges

    return []


if __name__ == "__main__":
    main()
