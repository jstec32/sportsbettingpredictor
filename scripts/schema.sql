-- Sports Betting Predictor Database Schema
-- PostgreSQL 14+

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(50) UNIQUE NOT NULL,
    league VARCHAR(10) NOT NULL CHECK (league IN ('NFL', 'NBA')),
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(10) NOT NULL,
    city VARCHAR(100),
    conference VARCHAR(50),
    division VARCHAR(50),
    stadium_name VARCHAR(100),
    stadium_lat DECIMAL(10, 6),
    stadium_lon DECIMAL(10, 6),
    is_dome BOOLEAN DEFAULT FALSE,
    timezone VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Games table
CREATE TABLE IF NOT EXISTS games (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(100) UNIQUE NOT NULL,
    league VARCHAR(10) NOT NULL,
    season INTEGER NOT NULL,
    season_type VARCHAR(20) NOT NULL,
    week INTEGER,
    game_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR(20) DEFAULT 'scheduled',
    venue_name VARCHAR(100),
    venue_lat DECIMAL(10, 6),
    venue_lon DECIMAL(10, 6),
    is_neutral_site BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Kalshi markets table
CREATE TABLE IF NOT EXISTS kalshi_markets (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(100) UNIQUE NOT NULL,
    event_ticker VARCHAR(100) NOT NULL,
    series_ticker VARCHAR(100),
    game_id INTEGER REFERENCES games(id),
    market_type VARCHAR(50) NOT NULL,
    title TEXT,
    subtitle TEXT,
    yes_bid DECIMAL(6, 4),
    yes_ask DECIMAL(6, 4),
    no_bid DECIMAL(6, 4),
    no_ask DECIMAL(6, 4),
    last_price DECIMAL(6, 4),
    volume INTEGER,
    open_interest INTEGER,
    status VARCHAR(20),
    close_time TIMESTAMP WITH TIME ZONE,
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Odds history table (from sportsbooks via The Odds API)
CREATE TABLE IF NOT EXISTS odds_history (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id),
    source VARCHAR(50) NOT NULL,
    market_type VARCHAR(30) NOT NULL,
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    home_odds DECIMAL(10, 4),
    away_odds DECIMAL(10, 4),
    home_prob DECIMAL(6, 4),
    away_prob DECIMAL(6, 4),
    spread_home DECIMAL(5, 1),
    spread_away DECIMAL(5, 1),
    total_line DECIMAL(5, 1),
    over_odds DECIMAL(10, 4),
    under_odds DECIMAL(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Injuries table
CREATE TABLE IF NOT EXISTS injuries (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id),
    player_external_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    position VARCHAR(20),
    injury_type VARCHAR(100),
    injury_status VARCHAR(50),
    injury_date DATE,
    return_date DATE,
    impact_rating DECIMAL(3, 2),
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT injuries_team_player_unique UNIQUE (team_id, player_external_id)
);

-- Weather conditions table
CREATE TABLE IF NOT EXISTS weather_conditions (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id) UNIQUE,
    temperature_f DECIMAL(5, 2),
    feels_like_f DECIMAL(5, 2),
    humidity_pct DECIMAL(5, 2),
    wind_speed_mph DECIMAL(5, 2),
    wind_direction_deg INTEGER,
    precipitation_prob DECIMAL(5, 2),
    weather_code INTEGER,
    weather_description VARCHAR(100),
    is_dome BOOLEAN DEFAULT FALSE,
    forecast_time TIMESTAMP WITH TIME ZONE,
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id),
    model_id VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(30) NOT NULL,
    predicted_at TIMESTAMP WITH TIME ZONE NOT NULL,
    home_win_prob DECIMAL(6, 4),
    away_win_prob DECIMAL(6, 4),
    predicted_spread DECIMAL(5, 1),
    predicted_total DECIMAL(5, 1),
    confidence DECIMAL(6, 4),
    edge DECIMAL(6, 4),
    features_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bets table (paper and real)
CREATE TABLE IF NOT EXISTS bets (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id),
    prediction_id INTEGER REFERENCES predictions(id),
    kalshi_market_id INTEGER REFERENCES kalshi_markets(id),
    bet_type VARCHAR(20) NOT NULL CHECK (bet_type IN ('paper', 'live')),
    market_type VARCHAR(30) NOT NULL,
    side VARCHAR(20) NOT NULL,
    placed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    stake DECIMAL(12, 2) NOT NULL,
    odds_at_placement DECIMAL(10, 4) NOT NULL,
    expected_value DECIMAL(10, 4),
    edge DECIMAL(6, 4),
    kelly_fraction DECIMAL(6, 4),
    status VARCHAR(20) DEFAULT 'open',
    settled_at TIMESTAMP WITH TIME ZONE,
    settlement_price DECIMAL(6, 4),
    pnl DECIMAL(12, 2),
    clv DECIMAL(6, 4),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio snapshots table
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_time TIMESTAMP WITH TIME ZONE NOT NULL,
    bankroll DECIMAL(12, 2) NOT NULL,
    total_exposure DECIMAL(12, 2),
    open_positions INTEGER,
    daily_pnl DECIMAL(12, 2),
    weekly_pnl DECIMAL(12, 2),
    total_pnl DECIMAL(12, 2),
    win_rate DECIMAL(6, 4),
    avg_edge DECIMAL(6, 4),
    avg_clv DECIMAL(6, 4),
    sharpe_ratio DECIMAL(6, 4),
    max_drawdown DECIMAL(6, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API request tracking table
CREATE TABLE IF NOT EXISTS api_requests (
    id SERIAL PRIMARY KEY,
    api_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    request_time TIMESTAMP WITH TIME ZONE NOT NULL,
    response_status INTEGER,
    response_time_ms INTEGER,
    requests_remaining INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_games_datetime ON games(game_datetime);
CREATE INDEX IF NOT EXISTS idx_games_league_season ON games(league, season);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_kalshi_markets_game ON kalshi_markets(game_id);
CREATE INDEX IF NOT EXISTS idx_kalshi_markets_captured ON kalshi_markets(captured_at);
CREATE INDEX IF NOT EXISTS idx_kalshi_markets_status ON kalshi_markets(status);
CREATE INDEX IF NOT EXISTS idx_odds_history_game ON odds_history(game_id);
CREATE INDEX IF NOT EXISTS idx_odds_history_captured ON odds_history(captured_at);
CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_bets_game ON bets(game_id);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
CREATE INDEX IF NOT EXISTS idx_bets_type ON bets(bet_type);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team_id);
CREATE INDEX IF NOT EXISTS idx_injuries_status ON injuries(injury_status);
CREATE INDEX IF NOT EXISTS idx_api_requests_time ON api_requests(request_time);
CREATE INDEX IF NOT EXISTS idx_api_requests_api ON api_requests(api_name);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_time ON portfolio_snapshots(snapshot_time);

-- Trigger function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at column
DROP TRIGGER IF EXISTS update_teams_updated_at ON teams;
CREATE TRIGGER update_teams_updated_at
    BEFORE UPDATE ON teams
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_games_updated_at ON games;
CREATE TRIGGER update_games_updated_at
    BEFORE UPDATE ON games
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_bets_updated_at ON bets;
CREATE TRIGGER update_bets_updated_at
    BEFORE UPDATE ON bets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
