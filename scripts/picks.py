#!/usr/bin/env python3
"""Daily picks report — model predictions vs market odds for today's NBA games.

Trains the Elo model on all completed historical games (building current team
ratings), then generates predictions for today's scheduled games. Shows every
game in a formatted table with model probability, market odds, edge, and
quarter-Kelly stake recommendation.

Saves a picks log to reports/picks_YYYYMMDD.json for simulation tracking.
Compare against settle_scores output to measure model performance over time.

Usage:
    python scripts/picks.py                        # today's games
    python scripts/picks.py --date 2026-02-25      # specific date
    python scripts/picks.py --min-edge 0.0         # show all games (no filter)
    python scripts/picks.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import click
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.data.storage import DatabaseStorage
from src.features.situational import SituationalFeatureCalculator
from src.market.edge_calculator import EdgeCalculator
from src.models.elo import EloModel
from src.utils.logging_config import setup_logging

load_dotenv()

# Quarter-Kelly fraction (matches risk_limits.yaml)
KELLY_FRACTION = 0.25
# Cap per bet (matches risk_limits.yaml)
MAX_BET_PCT = 0.05


@click.command()
@click.option(
    "--date",
    "target_date",
    default=None,
    help="Date to generate picks for (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--min-edge",
    type=float,
    default=0.03,
    help="Minimum edge to flag a pick (default: 3%). Use 0.0 to show all games.",
)
@click.option(
    "--seasons",
    default="2024,2025,2026",
    help="Comma-separated training seasons",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    target_date: str | None,
    min_edge: float,
    seasons: str,
    verbose: bool,
) -> None:
    """Generate daily picks report."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")

    # Resolve target date
    if target_date:
        pick_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    else:
        pick_date = datetime.now(timezone.utc).date()

    season_list = [int(s.strip()) for s in seasons.split(",")]

    logger.info(f"Generating NBA picks for {pick_date}...")

    storage = DatabaseStorage()

    # --- 1. Load upcoming games for the target date ---
    upcoming = _load_upcoming_games(storage, pick_date)
    if not upcoming:
        logger.warning(f"No scheduled NBA games found for {pick_date}. Run make sync-schedule first.")
        return

    logger.info(f"Found {len(upcoming)} games scheduled for {pick_date}")

    # --- 2. Load latest market odds for those games ---
    game_ids = [g["id"] for g in upcoming]
    market_odds = _load_market_odds(storage, game_ids)
    odds_by_game = {row["game_id"]: row for row in market_odds}
    real_odds_count = len(odds_by_game)
    logger.info(
        f"Market odds: {real_odds_count} real, {len(upcoming) - real_odds_count} missing "
        f"({'run make capture-odds' if real_odds_count < len(upcoming) else 'complete'})"
    )

    # --- 3. Load completed historical games for training ---
    history = _load_history(storage, season_list)
    if history.empty:
        logger.error("No historical data found. Cannot train model.")
        return
    logger.info(f"Loaded {len(history)} completed games for training")

    # --- 4. Compute situational features ---
    # Pass history + upcoming together so upcoming games get accurate rest/form features
    upcoming_df = pd.DataFrame(upcoming)
    upcoming_df["home_score"] = None
    upcoming_df["away_score"] = None
    upcoming_df["home_win"] = None

    combined = pd.concat([history, upcoming_df], ignore_index=True).sort_values("game_datetime")

    logger.info("Computing situational features...")
    sit_calc = SituationalFeatureCalculator()
    sit_features = sit_calc.compute_features_for_games(combined)
    combined = pd.concat([combined.reset_index(drop=True), sit_features], axis=1)

    # Split back out
    history_idx = combined["home_win"].notna()
    hist_df = combined[history_idx].copy()
    upcoming_features = combined[~history_idx].copy()

    # --- 5. Train Elo on full history ---
    logger.info("Training Elo model on full history...")
    elo = EloModel()
    X_hist = hist_df[["home_team_id", "away_team_id", "is_neutral_site",
                       "season", "home_score", "away_score"]].copy()
    y_hist = hist_df["home_win"].astype(int)
    elo.train(X_hist, y_hist)

    # --- 6. Generate predictions for each upcoming game ---
    picks = []
    edge_calc = EdgeCalculator(min_edge=min_edge)

    for _, game in upcoming_df.iterrows():
        game_id = game["id"]

        # Build feature row for prediction (scores absent — future game)
        feat_row = upcoming_features[upcoming_features["id"] == game_id]
        if feat_row.empty:
            continue

        X_pred = feat_row[["home_team_id", "away_team_id", "is_neutral_site",
                            "season"]].copy()
        proba = elo.predict_proba(X_pred)[0]
        away_prob, home_prob = float(proba[0]), float(proba[1])
        confidence = abs(home_prob - 0.5) * 2

        # Market odds
        market = odds_by_game.get(game_id)
        market_home_prob = float(market["home_prob"]) if market else None
        market_away_prob = float(market["away_prob"]) if market else None
        home_odds = float(market["home_odds"]) if market else None
        away_odds = float(market["away_odds"]) if market else None

        # Edge and Kelly
        home_edge = edge_calc.calculate_edge(home_prob, market_home_prob) if market_home_prob else None
        away_edge = edge_calc.calculate_edge(away_prob, market_away_prob) if market_away_prob else None

        if market_home_prob:
            home_kelly = min(
                edge_calc.calculate_kelly(home_prob, 1 / market_home_prob) * KELLY_FRACTION,
                MAX_BET_PCT,
            )
            away_kelly = min(
                edge_calc.calculate_kelly(away_prob, 1 / market_away_prob) * KELLY_FRACTION,
                MAX_BET_PCT,
            )
        else:
            home_kelly = away_kelly = None

        # Determine model's pick and its edge
        model_pick = "home" if home_prob > away_prob else "away"
        pick_edge = home_edge if model_pick == "home" else away_edge
        pick_kelly = home_kelly if model_pick == "home" else away_kelly

        picks.append({
            "game_id": game_id,
            "game_datetime": game["game_datetime"].isoformat() if hasattr(game["game_datetime"], "isoformat") else str(game["game_datetime"]),
            "home_team": game["home_team_name"],
            "away_team": game["away_team_name"],
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "confidence": round(confidence, 4),
            "model_pick": model_pick,
            "market_home_prob": round(market_home_prob, 4) if market_home_prob else None,
            "market_away_prob": round(market_away_prob, 4) if market_away_prob else None,
            "home_moneyline": home_odds,
            "away_moneyline": away_odds,
            "home_edge": round(home_edge, 4) if home_edge is not None else None,
            "away_edge": round(away_edge, 4) if away_edge is not None else None,
            "pick_edge": round(pick_edge, 4) if pick_edge is not None else None,
            "pick_kelly": round(pick_kelly, 4) if pick_kelly is not None else None,
            "has_market_odds": market is not None,
            "flagged": bool(pick_edge is not None and pick_edge >= min_edge),
        })

    # --- 7. Display table ---
    _print_picks_table(picks, pick_date, min_edge)

    # --- 8. Save picks log ---
    _save_picks(picks, pick_date)


def _load_upcoming_games(storage: DatabaseStorage, pick_date: Any) -> list[dict]:
    """Load today's scheduled games with team names."""
    query = """
        SELECT g.id, g.game_datetime, g.season, g.is_neutral_site,
               g.home_team_id, g.away_team_id,
               ht.name as home_team_name, at.name as away_team_name,
               ht.abbreviation as home_abbr, at.abbreviation as away_abbr
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.league = 'NBA'
          AND g.status = 'scheduled'
          AND g.game_datetime::date = :pick_date
        ORDER BY g.game_datetime
    """
    result = storage.execute(query, {"pick_date": str(pick_date)})
    return [dict(row._mapping) for row in result]


def _load_market_odds(storage: DatabaseStorage, game_ids: list[int]) -> list[dict]:
    """Get most recent odds for each game (latest captured, not opening line)."""
    if not game_ids:
        return []
    query = """
        SELECT DISTINCT ON (game_id)
            game_id, home_prob, away_prob, home_odds, away_odds, captured_at
        FROM odds_history
        WHERE game_id = ANY(:game_ids)
          AND game_id IS NOT NULL
        ORDER BY game_id, captured_at DESC
    """
    result = storage.execute(query, {"game_ids": game_ids})
    return [dict(row._mapping) for row in result]


def _load_history(storage: DatabaseStorage, seasons: list[int]) -> pd.DataFrame:
    """Load all completed games for training."""
    query = """
        SELECT g.id, g.game_datetime, g.season, g.is_neutral_site,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               ht.name as home_team_name, at.name as away_team_name
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.league = 'NBA'
          AND g.status = 'final'
          AND g.season = ANY(:seasons)
        ORDER BY g.game_datetime
    """
    result = storage.execute(query, {"seasons": seasons})
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame([dict(row._mapping) for row in result])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    return df


def _print_picks_table(picks: list[dict], pick_date: Any, min_edge: float) -> None:
    """Print formatted picks table to stdout."""
    flagged = [p for p in picks if p["flagged"]]

    print()
    print("=" * 72)
    print(f"  NBA PICKS — {_fmt_date(pick_date)}  |  Elo model  |  {len(picks)} games")
    print("=" * 72)
    print(f"  {'#':<3} {'Time ET':<8} {'Matchup':<42} {'Pick':<5} {'Model':>6} {'Market':>7} {'Edge':>6} {'Kelly':>6}")
    print(f"  {'-'*3} {'-'*8} {'-'*42} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")

    for i, p in enumerate(picks, 1):
        time_str = _fmt_time(p["game_datetime"])
        matchup = f"{p['away_team'][:18]} @ {p['home_team'][:18]}"[:42]

        pick_team = p["home_team"] if p["model_pick"] == "home" else p["away_team"]
        pick_abbr = pick_team.split()[-1][:5]

        model_pct = f"{(p['home_prob'] if p['model_pick'] == 'home' else p['away_prob']):.0%}"

        if p["has_market_odds"]:
            mkt_pct = f"{(p['market_home_prob'] if p['model_pick'] == 'home' else p['market_away_prob']):.0%}"
            edge_str = f"{p['pick_edge']:+.1%}" if p["pick_edge"] is not None else "  —  "
            kelly_str = f"{p['pick_kelly']:.1%}" if p["pick_kelly"] else "  —  "
        else:
            mkt_pct = "  —  "
            edge_str = "  —  "
            kelly_str = "  —  "

        flag = " ★" if p["flagged"] else ""

        print(f"  {i:<3} {time_str:<8} {matchup:<42} {pick_abbr:<5} {model_pct:>6} {mkt_pct:>7} {edge_str:>6} {kelly_str:>6}{flag}")

    print()
    if flagged:
        print(f"  ★ {len(flagged)} game(s) with edge ≥ {min_edge:.0%} — recommended picks:")
        for p in flagged:
            pick_team = p["home_team"] if p["model_pick"] == "home" else p["away_team"]
            ml = p["home_moneyline"] if p["model_pick"] == "home" else p["away_moneyline"]
            ml_str = f"{ml:+.0f}" if ml else "N/A"
            print(f"    → {pick_team} ({ml_str})  edge={p['pick_edge']:+.1%}  kelly={p['pick_kelly']:.1%}")
    else:
        print(f"  No picks with edge ≥ {min_edge:.0%} today.")
    print()


def _save_picks(picks: list[dict], pick_date: Any) -> None:
    """Save picks log to reports/picks_YYYYMMDD.json."""
    Path("reports").mkdir(exist_ok=True)
    out_path = Path(f"reports/picks_{str(pick_date).replace('-', '')}.json")

    payload = {
        "date": str(pick_date),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "elo",
        "total_games": len(picks),
        "flagged_picks": sum(1 for p in picks if p["flagged"]),
        "picks": picks,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(f"Picks saved to {out_path}")


def _fmt_date(d: Any) -> str:
    from datetime import date
    if isinstance(d, date):
        return d.strftime("%a %b %-d, %Y")
    return str(d)


def _fmt_time(dt_str: str) -> str:
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        # Convert UTC to ET (UTC-5 in Feb)
        et = dt - timedelta(hours=5)
        return et.strftime("%-I:%M%p").lower()
    except Exception:
        return "  —   "


if __name__ == "__main__":
    main()
