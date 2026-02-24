from __future__ import annotations

#!/usr/bin/env python3
"""Generate performance reports."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger

from src.data.storage import DatabaseStorage
from src.risk.portfolio import Portfolio
from src.market.clv_tracker import CLVTracker
from src.utils.logging_config import setup_logging

load_dotenv()


@click.command()
@click.option(
    "--period",
    type=click.Choice(["day", "week", "month", "all"]),
    default="week",
    help="Reporting period",
)
@click.option(
    "--metrics",
    default="roi,sharpe,clv,drawdown",
    help="Comma-separated metrics to include",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    period: str,
    metrics: str,
    output: str | None,
    output_format: str,
    verbose: bool,
) -> None:
    """Generate performance reports."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    logger.info(f"Generating {period} report...")

    storage = DatabaseStorage()
    metrics_list = [m.strip() for m in metrics.split(",")]

    # Determine date range
    now = datetime.now(timezone.utc)
    if period == "day":
        start_date = now - timedelta(days=1)
    elif period == "week":
        start_date = now - timedelta(days=7)
    elif period == "month":
        start_date = now - timedelta(days=30)
    else:
        start_date = None

    # Generate report data
    report = generate_report(storage, start_date, now, metrics_list)

    # Output report
    if output_format == "json":
        output_text = json.dumps(report, indent=2, default=str)
    else:
        output_text = format_text_report(report, period)

    # Display or save
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
        logger.info(f"Report saved to {output_path}")
    else:
        print(output_text)


def generate_report(
    storage: DatabaseStorage,
    start_date: datetime | None,
    end_date: datetime,
    metrics: list[str],
) -> dict:
    """Generate report data."""
    report = {
        "generated_at": end_date.isoformat(),
        "period_start": start_date.isoformat() if start_date else "all time",
        "period_end": end_date.isoformat(),
    }

    # Get betting performance
    report["betting"] = get_betting_stats(storage, start_date)

    # Get portfolio metrics
    report["portfolio"] = get_portfolio_stats(storage, start_date)

    # Get model performance
    report["models"] = get_model_stats(storage, start_date)

    # Get data quality summary
    report["data_quality"] = get_data_quality_summary(storage)

    return report


def get_betting_stats(storage: DatabaseStorage, start_date: datetime | None) -> dict:
    """Get betting statistics."""
    date_filter = ""
    params = {}

    if start_date:
        date_filter = "AND placed_at >= :start_date"
        params["start_date"] = start_date

    # Overall stats
    query = f"""
        SELECT
            COUNT(*) as total_bets,
            SUM(CASE WHEN status = 'settled' AND pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN status = 'settled' AND pnl < 0 THEN 1 ELSE 0 END) as losses,
            SUM(stake) as total_staked,
            SUM(CASE WHEN status = 'settled' THEN pnl ELSE 0 END) as total_pnl,
            AVG(edge) as avg_edge,
            AVG(CASE WHEN status = 'settled' THEN clv ELSE NULL END) as avg_clv
        FROM bets
        WHERE bet_type = 'paper' {date_filter}
    """

    result = storage.execute(query, params)

    if not result:
        return {}

    row = result[0]._mapping

    total_bets = row["total_bets"] or 0
    wins = row["wins"] or 0
    losses = row["losses"] or 0
    total_staked = float(row["total_staked"] or 0)
    total_pnl = float(row["total_pnl"] or 0)

    return {
        "total_bets": total_bets,
        "settled_bets": wins + losses,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "total_staked": total_staked,
        "total_pnl": total_pnl,
        "roi": total_pnl / total_staked if total_staked > 0 else 0,
        "avg_edge": float(row["avg_edge"] or 0),
        "avg_clv": float(row["avg_clv"] or 0),
    }


def get_portfolio_stats(storage: DatabaseStorage, start_date: datetime | None) -> dict:
    """Get portfolio statistics."""
    # Get latest snapshot
    query = """
        SELECT *
        FROM portfolio_snapshots
        ORDER BY snapshot_time DESC
        LIMIT 1
    """

    result = storage.execute(query)

    if not result:
        return {}

    row = result[0]._mapping

    return {
        "bankroll": float(row["bankroll"] or 0),
        "total_exposure": float(row["total_exposure"] or 0),
        "open_positions": row["open_positions"] or 0,
        "total_pnl": float(row["total_pnl"] or 0),
        "max_drawdown": float(row["max_drawdown"] or 0),
        "sharpe_ratio": float(row["sharpe_ratio"]) if row["sharpe_ratio"] else None,
    }


def get_model_stats(storage: DatabaseStorage, start_date: datetime | None) -> dict:
    """Get model performance statistics."""
    date_filter = ""
    params = {}

    if start_date:
        date_filter = "AND predicted_at >= :start_date"
        params["start_date"] = start_date

    query = f"""
        SELECT
            model_id,
            COUNT(*) as predictions,
            AVG(confidence) as avg_confidence,
            AVG(edge) as avg_edge
        FROM predictions
        WHERE 1=1 {date_filter}
        GROUP BY model_id
    """

    result = storage.execute(query, params)

    models = {}
    for row in result:
        model_id = row._mapping["model_id"]
        models[model_id] = {
            "predictions": row._mapping["predictions"],
            "avg_confidence": float(row._mapping["avg_confidence"] or 0),
            "avg_edge": float(row._mapping["avg_edge"] or 0),
        }

    return models


def get_data_quality_summary(storage: DatabaseStorage) -> dict:
    """Get data quality summary."""
    # Teams
    teams_query = "SELECT COUNT(*) FROM teams"
    teams_result = storage.execute(teams_query)
    team_count = teams_result[0][0] if teams_result else 0

    # Games
    games_query = """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'final' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'scheduled' THEN 1 ELSE 0 END) as scheduled
        FROM games
    """
    games_result = storage.execute(games_query)
    games_row = games_result[0]._mapping if games_result else {}

    # Odds
    odds_query = """
        SELECT COUNT(*) as count,
               MAX(captured_at) as last_update
        FROM odds_history
    """
    odds_result = storage.execute(odds_query)
    odds_row = odds_result[0]._mapping if odds_result else {}

    return {
        "teams": team_count,
        "games_total": games_row.get("total", 0),
        "games_completed": games_row.get("completed", 0),
        "games_scheduled": games_row.get("scheduled", 0),
        "odds_records": odds_row.get("count", 0),
        "odds_last_update": odds_row.get("last_update"),
    }


def format_text_report(report: dict, period: str) -> str:
    """Format report as text."""
    lines = []

    lines.append("=" * 60)
    lines.append(f"SPORTS BETTING PERFORMANCE REPORT - {period.upper()}")
    lines.append("=" * 60)
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Period: {report['period_start']} to {report['period_end']}")
    lines.append("")

    # Betting Performance
    betting = report.get("betting", {})
    lines.append("BETTING PERFORMANCE")
    lines.append("-" * 40)
    lines.append(f"Total Bets: {betting.get('total_bets', 0)}")
    lines.append(f"Win Rate: {betting.get('win_rate', 0):.1%}")
    lines.append(f"Total P&L: ${betting.get('total_pnl', 0):,.2f}")
    lines.append(f"ROI: {betting.get('roi', 0):.1%}")
    lines.append(f"Avg Edge: {betting.get('avg_edge', 0):.2%}")
    lines.append(f"Avg CLV: {betting.get('avg_clv', 0):.2%}")
    lines.append("")

    # Portfolio
    portfolio = report.get("portfolio", {})
    lines.append("PORTFOLIO STATUS")
    lines.append("-" * 40)
    lines.append(f"Bankroll: ${portfolio.get('bankroll', 0):,.2f}")
    lines.append(f"Total P&L: ${portfolio.get('total_pnl', 0):,.2f}")
    lines.append(f"Max Drawdown: {portfolio.get('max_drawdown', 0):.1%}")
    if portfolio.get("sharpe_ratio"):
        lines.append(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}")
    lines.append("")

    # Data Quality
    data = report.get("data_quality", {})
    lines.append("DATA STATUS")
    lines.append("-" * 40)
    lines.append(f"Teams: {data.get('teams', 0)}")
    lines.append(f"Games: {data.get('games_total', 0)} total, {data.get('games_scheduled', 0)} upcoming")
    lines.append(f"Odds Records: {data.get('odds_records', 0)}")
    lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    main()
