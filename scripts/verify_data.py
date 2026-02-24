from __future__ import annotations

#!/usr/bin/env python3
"""Verify data quality and completeness."""

from datetime import datetime, timedelta, timezone

import click
from dotenv import load_dotenv
from loguru import logger

from src.data.storage import DatabaseStorage
from src.risk.alerts import AlertManager, AlertType, AlertSeverity
from src.utils.logging_config import setup_logging

load_dotenv()


@click.command()
@click.option(
    "--check-all",
    is_flag=True,
    help="Run all data quality checks",
)
@click.option(
    "--check-teams",
    is_flag=True,
    help="Check team data",
)
@click.option(
    "--check-games",
    is_flag=True,
    help="Check game data",
)
@click.option(
    "--check-odds",
    is_flag=True,
    help="Check odds data",
)
@click.option(
    "--check-api",
    is_flag=True,
    help="Check API request logs",
)
@click.option(
    "--alert-on-issues",
    is_flag=True,
    help="Create alerts for issues found",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    check_all: bool,
    check_teams: bool,
    check_games: bool,
    check_odds: bool,
    check_api: bool,
    alert_on_issues: bool,
    verbose: bool,
) -> None:
    """Verify data quality and completeness."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    logger.info("Starting data verification...")

    storage = DatabaseStorage()
    alert_manager = AlertManager() if alert_on_issues else None

    # If no specific checks requested, run all
    if not any([check_teams, check_games, check_odds, check_api]):
        check_all = True

    issues_found = 0

    if check_all or check_teams:
        issues_found += verify_teams(storage, alert_manager)

    if check_all or check_games:
        issues_found += verify_games(storage, alert_manager)

    if check_all or check_odds:
        issues_found += verify_odds(storage, alert_manager)

    if check_all or check_api:
        issues_found += verify_api_requests(storage, alert_manager)

    # Summary
    logger.info("")
    logger.info("=" * 50)
    if issues_found == 0:
        logger.info("✓ All data quality checks passed!")
    else:
        logger.warning(f"⚠ Found {issues_found} data quality issues")

    if alert_manager:
        active = alert_manager.get_active_alerts()
        if active:
            logger.info(f"  {len(active)} alerts created")


def verify_teams(storage: DatabaseStorage, alert_manager: AlertManager | None) -> int:
    """Verify team data."""
    logger.info("Checking team data...")
    issues = 0

    # Check team counts
    query = """
        SELECT league, COUNT(*) as count
        FROM teams
        GROUP BY league
    """
    result = storage.execute(query)

    expected_counts = {"NFL": 32, "NBA": 30}

    for row in result:
        league = row._mapping["league"]
        count = row._mapping["count"]
        expected = expected_counts.get(league, 0)

        if count != expected:
            issues += 1
            msg = f"{league} has {count} teams, expected {expected}"
            logger.warning(f"  ⚠ {msg}")
            if alert_manager:
                alert_manager.alert_data_quality(msg, count)
        else:
            logger.info(f"  ✓ {league}: {count} teams")

    # Check for missing essential data
    query = """
        SELECT COUNT(*) as count
        FROM teams
        WHERE name IS NULL OR abbreviation IS NULL
    """
    result = storage.execute(query)
    missing = result[0]._mapping["count"]

    if missing > 0:
        issues += 1
        msg = f"{missing} teams with missing name/abbreviation"
        logger.warning(f"  ⚠ {msg}")
        if alert_manager:
            alert_manager.alert_data_quality(msg, missing)

    return issues


def verify_games(storage: DatabaseStorage, alert_manager: AlertManager | None) -> int:
    """Verify game data."""
    logger.info("Checking game data...")
    issues = 0

    # Check recent games
    query = """
        SELECT league, status, COUNT(*) as count
        FROM games
        WHERE game_datetime > NOW() - INTERVAL '7 days'
        GROUP BY league, status
        ORDER BY league, status
    """
    result = storage.execute(query)

    logger.info("  Recent games (last 7 days):")
    for row in result:
        league = row._mapping["league"]
        status = row._mapping["status"]
        count = row._mapping["count"]
        logger.info(f"    {league} - {status}: {count}")

    # Check for orphaned games (no team reference)
    query = """
        SELECT COUNT(*) as count
        FROM games
        WHERE home_team_id IS NULL OR away_team_id IS NULL
    """
    result = storage.execute(query)
    orphaned = result[0]._mapping["count"]

    if orphaned > 0:
        issues += 1
        msg = f"{orphaned} games with missing team references"
        logger.warning(f"  ⚠ {msg}")
        if alert_manager:
            alert_manager.alert_data_quality(msg, orphaned)

    # Check for games with missing scores that should be final
    query = """
        SELECT COUNT(*) as count
        FROM games
        WHERE status = 'final'
          AND (home_score IS NULL OR away_score IS NULL)
    """
    result = storage.execute(query)
    missing_scores = result[0]._mapping["count"]

    if missing_scores > 0:
        issues += 1
        msg = f"{missing_scores} final games with missing scores"
        logger.warning(f"  ⚠ {msg}")
        if alert_manager:
            alert_manager.alert_data_quality(msg, missing_scores)

    return issues


def verify_odds(storage: DatabaseStorage, alert_manager: AlertManager | None) -> int:
    """Verify odds data."""
    logger.info("Checking odds data...")
    issues = 0

    # Check recent odds captures
    query = """
        SELECT DATE(captured_at) as date, COUNT(*) as count
        FROM odds_history
        WHERE captured_at > NOW() - INTERVAL '7 days'
        GROUP BY DATE(captured_at)
        ORDER BY date DESC
    """
    result = storage.execute(query)

    if not result:
        issues += 1
        msg = "No odds captured in the last 7 days"
        logger.warning(f"  ⚠ {msg}")
        if alert_manager:
            alert_manager.alert_data_quality(msg)
    else:
        logger.info("  Recent odds captures:")
        for row in result:
            date = row._mapping["date"]
            count = row._mapping["count"]
            logger.info(f"    {date}: {count} records")

    # Check for stale odds (games with old odds data)
    query = """
        SELECT g.id, g.game_datetime, MAX(oh.captured_at) as last_odds
        FROM games g
        LEFT JOIN odds_history oh ON g.id = oh.game_id
        WHERE g.status = 'scheduled'
          AND g.game_datetime > NOW()
          AND g.game_datetime < NOW() + INTERVAL '2 days'
        GROUP BY g.id, g.game_datetime
        HAVING MAX(oh.captured_at) IS NULL
           OR MAX(oh.captured_at) < NOW() - INTERVAL '12 hours'
    """
    result = storage.execute(query)
    stale_count = len(result)

    if stale_count > 0:
        issues += 1
        msg = f"{stale_count} upcoming games with stale/missing odds"
        logger.warning(f"  ⚠ {msg}")
        if alert_manager:
            alert_manager.alert_data_quality(msg, stale_count)

    return issues


def verify_api_requests(storage: DatabaseStorage, alert_manager: AlertManager | None) -> int:
    """Verify API request logs."""
    logger.info("Checking API request logs...")
    issues = 0

    # Check for recent API errors
    query = """
        SELECT api_name, COUNT(*) as error_count
        FROM api_requests
        WHERE request_time > NOW() - INTERVAL '24 hours'
          AND (response_status >= 400 OR error_message IS NOT NULL)
        GROUP BY api_name
    """
    result = storage.execute(query)

    if result:
        for row in result:
            api_name = row._mapping["api_name"]
            error_count = row._mapping["error_count"]
            issues += 1
            msg = f"{api_name}: {error_count} errors in last 24h"
            logger.warning(f"  ⚠ {msg}")
            if alert_manager:
                alert_manager.alert_api_error(api_name, f"{error_count} errors in last 24h")

    # Check API response times
    query = """
        SELECT api_name,
               AVG(response_time_ms) as avg_time,
               MAX(response_time_ms) as max_time
        FROM api_requests
        WHERE request_time > NOW() - INTERVAL '24 hours'
          AND response_time_ms IS NOT NULL
        GROUP BY api_name
    """
    result = storage.execute(query)

    logger.info("  API response times (last 24h):")
    for row in result:
        api_name = row._mapping["api_name"]
        avg_time = row._mapping["avg_time"]
        max_time = row._mapping["max_time"]
        logger.info(f"    {api_name}: avg={avg_time:.0f}ms, max={max_time:.0f}ms")

        if avg_time and avg_time > 5000:
            issues += 1
            msg = f"{api_name} slow response times (avg: {avg_time:.0f}ms)"
            logger.warning(f"    ⚠ {msg}")

    # Check rate limit usage (The Odds API)
    query = """
        SELECT MIN(requests_remaining) as min_remaining
        FROM api_requests
        WHERE api_name = 'OddsAPIClient'
          AND request_time > NOW() - INTERVAL '24 hours'
          AND requests_remaining IS NOT NULL
    """
    result = storage.execute(query)

    if result and result[0]._mapping["min_remaining"] is not None:
        remaining = result[0]._mapping["min_remaining"]
        logger.info(f"  The Odds API requests remaining: {remaining}")
        if remaining < 50:
            issues += 1
            msg = f"Low Odds API quota: {remaining} requests remaining"
            logger.warning(f"  ⚠ {msg}")
            if alert_manager:
                alert_manager.create_alert(
                    AlertType.API_ERROR,
                    AlertSeverity.WARNING,
                    msg,
                )

    return issues


if __name__ == "__main__":
    main()
