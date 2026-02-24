#!/usr/bin/env python3
"""Generate NBA predictions with Elo model and situational adjustments.

Usage:
    python scripts/predict.py
    python scripts/predict.py --min-edge 0.05
    python scripts/predict.py --output predictions.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.data.odds import OddsAPIClient
from src.data.storage import DatabaseStorage
from src.features.player_metrics import PlayerMetricsCalculator, InjuryImpact
from src.models.elo import EloModel
from src.utils.logging_config import setup_logging

load_dotenv()


class NBAPredictor:
    """NBA game predictor with Elo and situational factors."""

    # Back-to-back penalty (research shows ~3-5% impact)
    B2B_PENALTY = 0.04

    # Rest advantage per day (caps at 2 days)
    REST_ADVANTAGE_PER_DAY = 0.01
    MAX_REST_ADVANTAGE = 0.02

    # Injury impact multiplier (max ~7.5% adjustment at 0.5 impact diff)
    INJURY_IMPACT_MULTIPLIER = 0.15

    def __init__(self) -> None:
        """Initialize predictor."""
        self.storage = DatabaseStorage()
        self.elo = EloModel(model_id="nba_elo_v1")
        self.metrics = PlayerMetricsCalculator()
        self._team_names: dict[int, str] = {}
        self._team_by_name: dict[str, int] = {}
        self._trained = False

    def _load_teams(self) -> None:
        """Load team mappings from database."""
        query = """SELECT id, name, abbreviation FROM teams WHERE league = 'NBA'"""
        teams = self.storage.execute(query)
        for row in teams:
            self._team_names[row[0]] = row[2]
            self._team_by_name[row[1]] = row[0]

    # Common name variations from APIs to database names
    NAME_ALIASES = {
        "Los Angeles Clippers": "LA Clippers",
        "Los Angeles Lakers": "Los Angeles Lakers",
    }

    def get_team_id(self, name: str) -> int | None:
        """Get team ID from name with fuzzy matching."""
        # Check direct match
        if name in self._team_by_name:
            return self._team_by_name[name]

        # Check aliases
        if name in self.NAME_ALIASES:
            alias = self.NAME_ALIASES[name]
            if alias in self._team_by_name:
                return self._team_by_name[alias]

        # Fuzzy match - check if either contains the other
        for db_name, tid in self._team_by_name.items():
            if name in db_name or db_name in name:
                return tid

        # Try matching on city or team name parts
        name_parts = name.lower().split()
        for db_name, tid in self._team_by_name.items():
            db_parts = db_name.lower().split()
            # Check if team nickname matches (e.g., "Clippers", "Lakers")
            if name_parts[-1] in db_parts:
                return tid

        return None

    def train(self) -> None:
        """Train Elo model on historical games."""
        self._load_teams()

        query = """
            SELECT g.home_team_id, g.away_team_id, g.home_score, g.away_score,
                   g.game_datetime
            FROM games g
            WHERE g.league = 'NBA' AND g.home_score IS NOT NULL
            ORDER BY g.game_datetime
        """
        result = self.storage.execute(query)
        df = pd.DataFrame(
            result,
            columns=[
                "home_team_id",
                "away_team_id",
                "home_score",
                "away_score",
                "game_datetime",
            ],
        )
        df["league"] = "NBA"
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

        self.elo.train(df, df["home_win"])
        self._trained = True
        logger.info(f"Trained on {len(df)} games")

    def calculate_rest_days(self, team_id: int, game_date: datetime) -> int:
        """Calculate days since team's last game."""
        query = """
            SELECT g.game_datetime
            FROM games g
            WHERE g.league = 'NBA'
              AND (g.home_team_id = :team_id OR g.away_team_id = :team_id)
              AND g.game_datetime < :game_date
            ORDER BY g.game_datetime DESC
            LIMIT 1
        """
        result = self.storage.execute(
            query, {"team_id": team_id, "game_date": game_date}
        )
        if not result:
            return 7  # Well rested

        last_game = result[0][0]
        if last_game.tzinfo:
            last_game = last_game.replace(tzinfo=None)
        game_date_naive = (
            game_date.replace(tzinfo=None) if game_date.tzinfo else game_date
        )

        delta = (game_date_naive - last_game).days
        return max(0, delta - 1)

    def get_rest_adjustment(self, home_rest: int, away_rest: int) -> float:
        """Calculate probability adjustment based on rest differential."""
        home_b2b = home_rest == 0
        away_b2b = away_rest == 0

        adjustment = 0.0

        # Back-to-back penalties
        if away_b2b and not home_b2b:
            adjustment += self.B2B_PENALTY
        elif home_b2b and not away_b2b:
            adjustment -= self.B2B_PENALTY

        # General rest advantage (smaller effect, only if no B2B)
        if not home_b2b and not away_b2b:
            rest_diff = home_rest - away_rest
            adjustment += np.clip(
                rest_diff * self.REST_ADVANTAGE_PER_DAY,
                -self.MAX_REST_ADVANTAGE,
                self.MAX_REST_ADVANTAGE,
            )

        return adjustment

    def get_injury_adjustment(
        self,
        home_id: int,
        away_id: int,
    ) -> tuple[float, InjuryImpact, InjuryImpact]:
        """Calculate probability adjustment based on team injuries.

        Args:
            home_id: Home team ID.
            away_id: Away team ID.

        Returns:
            Tuple of (adjustment, home_impact, away_impact).
            Positive adjustment means home team benefits (away has more injuries).
        """
        home_injuries = self.storage.get_team_injuries(home_id)
        away_injuries = self.storage.get_team_injuries(away_id)

        home_impact = self.metrics.calculate_injury_impact(
            home_injuries, home_id, "NBA"
        )
        away_impact = self.metrics.calculate_injury_impact(
            away_injuries, away_id, "NBA"
        )

        # If away team has more injuries, home benefits (positive adjustment)
        impact_diff = away_impact.total_impact - home_impact.total_impact
        adjustment = impact_diff * self.INJURY_IMPACT_MULTIPLIER

        return adjustment, home_impact, away_impact

    def predict_game(
        self,
        home_id: int,
        away_id: int,
        game_time: datetime,
    ) -> dict[str, Any]:
        """Generate prediction for a single game."""
        if not self._trained:
            raise ValueError("Model not trained. Call train() first.")

        home_abbr = self._team_names.get(home_id, str(home_id))
        away_abbr = self._team_names.get(away_id, str(away_id))

        # Get rest days
        home_rest = self.calculate_rest_days(home_id, game_time)
        away_rest = self.calculate_rest_days(away_id, game_time)

        # Get injury adjustment
        injury_adj, home_inj, away_inj = self.get_injury_adjustment(home_id, away_id)

        # Base Elo prediction
        pred_df = pd.DataFrame(
            [
                {
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "league": "NBA",
                    "is_neutral": False,
                }
            ]
        )
        proba = self.elo.predict_proba(pred_df)[0]
        base_home_prob = proba[1]

        # Apply rest and injury adjustments
        rest_adj = self.get_rest_adjustment(home_rest, away_rest)
        total_adj = rest_adj + injury_adj
        adjusted_home_prob = float(np.clip(base_home_prob + total_adj, 0.05, 0.95))

        # Format injury summaries
        home_injuries_list = self.storage.get_team_injuries(home_id)
        away_injuries_list = self.storage.get_team_injuries(away_id)

        def format_injury_summary(injuries: list[dict[str, Any]]) -> str:
            if not injuries:
                return "None"
            summaries = []
            for inj in injuries[:3]:  # Show top 3 by impact
                status = inj.get("injury_status", "").lower()
                name = inj.get("player_name", "Unknown")
                summaries.append(f"{name} ({status})")
            more = len(injuries) - 3
            result = ", ".join(summaries)
            if more > 0:
                result += f", +{more} more"
            return result

        return {
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "game_time": game_time.isoformat(),
            "home_elo": self.elo.get_elo(home_id),
            "away_elo": self.elo.get_elo(away_id),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "home_b2b": home_rest == 0,
            "away_b2b": away_rest == 0,
            "base_home_prob": float(base_home_prob),
            "rest_adjustment": float(rest_adj),
            "injury_adjustment": float(injury_adj),
            "home_injury_impact": float(home_inj.total_impact),
            "away_injury_impact": float(away_inj.total_impact),
            "home_injuries_summary": format_injury_summary(home_injuries_list),
            "away_injuries_summary": format_injury_summary(away_injuries_list),
            "adjusted_home_prob": adjusted_home_prob,
            "adjusted_away_prob": 1 - adjusted_home_prob,
        }


async def generate_predictions(
    min_edge: float = 0.03,
    output_file: str | None = None,
) -> list[dict[str, Any]]:
    """Generate predictions for upcoming NBA games."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        logger.error("ODDS_API_KEY not found in environment")
        return []

    predictor = NBAPredictor()
    predictor.train()

    async with OddsAPIClient(api_key=api_key) as client:
        odds = await client.get_consensus_odds("NBA")

        predictions = []
        bets = []

        for game in odds:
            home_id = predictor.get_team_id(game.home_team)
            away_id = predictor.get_team_id(game.away_team)

            if not home_id or not away_id:
                logger.warning(f"Could not match teams: {game.away_team} @ {game.home_team}")
                continue

            pred = predictor.predict_game(home_id, away_id, game.commence_time)

            # Add market data
            pred["market_home_prob"] = float(game.home_prob) if game.home_prob else None
            pred["market_away_prob"] = float(game.away_prob) if game.away_prob else None

            if pred["market_home_prob"]:
                pred["home_edge"] = pred["adjusted_home_prob"] - pred["market_home_prob"]
                pred["away_edge"] = pred["adjusted_away_prob"] - pred["market_away_prob"]

                # Determine best bet
                if pred["home_edge"] >= min_edge:
                    pred["recommended_bet"] = pred["home_team"]
                    pred["bet_side"] = "HOME"
                    pred["bet_edge"] = pred["home_edge"]
                elif pred["away_edge"] >= min_edge:
                    pred["recommended_bet"] = pred["away_team"]
                    pred["bet_side"] = "AWAY"
                    pred["bet_edge"] = pred["away_edge"]
                else:
                    pred["recommended_bet"] = None
                    pred["bet_side"] = None
                    pred["bet_edge"] = None

                if pred["recommended_bet"]:
                    bets.append(pred)

            predictions.append(pred)

        # Print results
        print(f"\n{'='*90}")
        print(f"  NBA PREDICTIONS - {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
        print(f"  Model: Elo + Back-to-Back/Rest Adjustments")
        print(f"  Minimum Edge: {min_edge*100:.0f}%")
        print(f"{'='*90}\n")

        for pred in predictions:
            home = pred["home_team"]
            away = pred["away_team"]
            home_rest = "B2B" if pred["home_b2b"] else f"{pred['home_rest_days']}d"
            away_rest = "B2B" if pred["away_b2b"] else f"{pred['away_rest_days']}d"

            game_time = datetime.fromisoformat(pred["game_time"])
            time_str = game_time.strftime("%m/%d %I:%M %p")

            print(
                f"  {time_str}: {away} ({pred['away_elo']:.0f}) [{away_rest}] "
                f"@ {home} ({pred['home_elo']:.0f}) [{home_rest}]"
            )

            # Show injury summaries
            home_inj_summary = pred.get("home_injuries_summary", "None")
            away_inj_summary = pred.get("away_injuries_summary", "None")
            if home_inj_summary != "None" or away_inj_summary != "None":
                print(f"    {home} Injuries: {home_inj_summary}")
                print(f"    {away} Injuries: {away_inj_summary}")

            print(f"  {'-'*60}")

            if pred.get("market_home_prob"):
                print(f"  {'':14} {'Model':>10} {'Market':>10} {'Edge':>10}")
                print(
                    f"  {home+' Win:':14} {pred['adjusted_home_prob']*100:>9.1f}% "
                    f"{pred['market_home_prob']*100:>9.1f}% {pred['home_edge']*100:>+9.1f}%"
                )
                print(
                    f"  {away+' Win:':14} {pred['adjusted_away_prob']*100:>9.1f}% "
                    f"{pred['market_away_prob']*100:>9.1f}% {pred['away_edge']*100:>+9.1f}%"
                )

                if pred["recommended_bet"]:
                    # Build adjustment notes
                    notes = []
                    if pred["rest_adjustment"] > 0.02:
                        notes.append("AWAY B2B")
                    elif pred["rest_adjustment"] < -0.02:
                        notes.append("HOME B2B")
                    if pred.get("injury_adjustment", 0) > 0.02:
                        notes.append("AWAY INJ")
                    elif pred.get("injury_adjustment", 0) < -0.02:
                        notes.append("HOME INJ")
                    note_str = f" [{', '.join(notes)}]" if notes else ""
                    print(
                        f"\n  >>> BET: {pred['recommended_bet']} "
                        f"+{pred['bet_edge']*100:.1f}%{note_str}"
                    )
                else:
                    print(f"\n  >>> PASS")
            else:
                print("  Market odds not available")

            print()

        # Summary
        print(f"{'='*90}")
        print(f"  RECOMMENDED BETS: {len(bets)} games with >= {min_edge*100:.0f}% edge")
        print(f"{'='*90}")
        if bets:
            bets.sort(key=lambda x: -x["bet_edge"])
            for pred in bets:
                # Build adjustment notes
                notes = []
                if pred["rest_adjustment"] > 0.02:
                    notes.append("AWAY B2B")
                elif pred["rest_adjustment"] < -0.02:
                    notes.append("HOME B2B")
                if pred.get("injury_adjustment", 0) > 0.02:
                    notes.append("AWAY INJ")
                elif pred.get("injury_adjustment", 0) < -0.02:
                    notes.append("HOME INJ")
                note_str = f" [{', '.join(notes)}]" if notes else ""
                print(
                    f"  {pred['recommended_bet']:4} ({pred['bet_side']}): "
                    f"+{pred['bet_edge']*100:.1f}% edge{note_str}"
                )
        else:
            print(f"  No bets meet {min_edge*100:.0f}% threshold")

        print(f"\n  API requests remaining: {client._requests_remaining}")

        # Save to file if requested
        if output_file:
            output = {
                "generated_at": datetime.now().isoformat(),
                "min_edge": min_edge,
                "predictions": predictions,
                "recommended_bets": [
                    {
                        "team": p["recommended_bet"],
                        "side": p["bet_side"],
                        "edge": p["bet_edge"],
                        "game": f"{p['away_team']} @ {p['home_team']}",
                        "game_time": p["game_time"],
                    }
                    for p in bets
                ],
            }
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Predictions saved to {output_file}")

        return predictions


@click.command()
@click.option(
    "--min-edge",
    type=float,
    default=0.03,
    help="Minimum edge threshold for bet recommendations (default: 0.03 = 3%)",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output file for predictions JSON",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(min_edge: float, output: str | None, verbose: bool) -> None:
    """Generate NBA predictions with market edge analysis."""
    setup_logging(log_level="DEBUG" if verbose else "WARNING")
    asyncio.run(generate_predictions(min_edge=min_edge, output_file=output))


if __name__ == "__main__":
    main()
