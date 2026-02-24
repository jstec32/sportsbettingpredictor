"""Sample data for testing."""

from datetime import datetime, timezone


SAMPLE_NFL_TEAMS = [
    {
        "external_id": "12",
        "league": "NFL",
        "name": "Kansas City Chiefs",
        "abbreviation": "KC",
        "city": "Kansas City",
        "conference": "AFC",
        "division": "West",
    },
    {
        "external_id": "33",
        "league": "NFL",
        "name": "Baltimore Ravens",
        "abbreviation": "BAL",
        "city": "Baltimore",
        "conference": "AFC",
        "division": "North",
    },
    {
        "external_id": "26",
        "league": "NFL",
        "name": "San Francisco 49ers",
        "abbreviation": "SF",
        "city": "San Francisco",
        "conference": "NFC",
        "division": "West",
    },
    {
        "external_id": "6",
        "league": "NFL",
        "name": "Dallas Cowboys",
        "abbreviation": "DAL",
        "city": "Dallas",
        "conference": "NFC",
        "division": "East",
    },
]


SAMPLE_NBA_TEAMS = [
    {
        "external_id": "2",
        "league": "NBA",
        "name": "Boston Celtics",
        "abbreviation": "BOS",
        "city": "Boston",
        "conference": "Eastern",
        "division": "Atlantic",
    },
    {
        "external_id": "10",
        "league": "NBA",
        "name": "Denver Nuggets",
        "abbreviation": "DEN",
        "city": "Denver",
        "conference": "Western",
        "division": "Northwest",
    },
    {
        "external_id": "14",
        "league": "NBA",
        "name": "Los Angeles Lakers",
        "abbreviation": "LAL",
        "city": "Los Angeles",
        "conference": "Western",
        "division": "Pacific",
    },
]


SAMPLE_GAMES = [
    {
        "external_id": "401547417",
        "league": "NFL",
        "season": 2024,
        "season_type": "regular",
        "week": 1,
        "game_datetime": datetime(2024, 9, 5, 20, 20, tzinfo=timezone.utc),
        "home_team_external_id": "12",
        "away_team_external_id": "33",
        "home_score": 27,
        "away_score": 20,
        "status": "final",
    },
    {
        "external_id": "401547418",
        "league": "NFL",
        "season": 2024,
        "season_type": "regular",
        "week": 1,
        "game_datetime": datetime(2024, 9, 8, 20, 20, tzinfo=timezone.utc),
        "home_team_external_id": "6",
        "away_team_external_id": "26",
        "home_score": 17,
        "away_score": 30,
        "status": "final",
    },
    {
        "external_id": "401547500",
        "league": "NFL",
        "season": 2024,
        "season_type": "regular",
        "week": 2,
        "game_datetime": datetime(2024, 9, 15, 20, 20, tzinfo=timezone.utc),
        "home_team_external_id": "33",
        "away_team_external_id": "12",
        "home_score": None,
        "away_score": None,
        "status": "scheduled",
    },
]


SAMPLE_ODDS = [
    {
        "game_external_id": "401547417",
        "source": "draftkings",
        "market_type": "h2h",
        "captured_at": datetime(2024, 9, 5, 18, 0, tzinfo=timezone.utc),
        "home_odds": -150,
        "away_odds": 130,
        "spread_home": -3.5,
        "total_line": 46.5,
    },
    {
        "game_external_id": "401547417",
        "source": "fanduel",
        "market_type": "h2h",
        "captured_at": datetime(2024, 9, 5, 18, 0, tzinfo=timezone.utc),
        "home_odds": -145,
        "away_odds": 125,
        "spread_home": -3.0,
        "total_line": 47.0,
    },
    {
        "game_external_id": "401547418",
        "source": "draftkings",
        "market_type": "h2h",
        "captured_at": datetime(2024, 9, 8, 18, 0, tzinfo=timezone.utc),
        "home_odds": 110,
        "away_odds": -130,
        "spread_home": 2.5,
        "total_line": 48.5,
    },
]


SAMPLE_KALSHI_MARKETS = [
    {
        "ticker": "KXNFL-24SEP05-KC-BAL-KC",
        "event_ticker": "KXNFL-24SEP05-KC-BAL",
        "series_ticker": "KXNFL",
        "market_type": "binary",
        "title": "Will Kansas City Chiefs beat Baltimore Ravens?",
        "yes_bid": 0.58,
        "yes_ask": 0.62,
        "last_price": 0.60,
        "volume": 15000,
        "status": "open",
    },
    {
        "ticker": "KXNFL-24SEP08-DAL-SF-SF",
        "event_ticker": "KXNFL-24SEP08-DAL-SF",
        "series_ticker": "KXNFL",
        "market_type": "binary",
        "title": "Will San Francisco 49ers beat Dallas Cowboys?",
        "yes_bid": 0.55,
        "yes_ask": 0.59,
        "last_price": 0.57,
        "volume": 8500,
        "status": "open",
    },
]


SAMPLE_INJURIES = [
    {
        "team_external_id": "12",
        "player_external_id": "3139477",
        "player_name": "Patrick Mahomes",
        "position": "QB",
        "injury_type": "Ankle",
        "injury_status": "probable",
    },
    {
        "team_external_id": "33",
        "player_external_id": "3916387",
        "player_name": "Lamar Jackson",
        "position": "QB",
        "injury_type": None,
        "injury_status": "active",
    },
    {
        "team_external_id": "26",
        "player_external_id": "4361307",
        "player_name": "Brock Purdy",
        "position": "QB",
        "injury_type": "Shoulder",
        "injury_status": "questionable",
    },
]
