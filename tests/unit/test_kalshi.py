"""Tests for Kalshi API client."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from src.data.kalshi import KalshiClient, KalshiMarket, OrderBook


class TestKalshiClient:
    """Tests for KalshiClient."""

    def test_init(self):
        """Test client initialization."""
        client = KalshiClient()
        assert client.BASE_URL == "https://api.elections.kalshi.com/trade-api/v2"
        assert client._token is None

    def test_init_with_credentials(self):
        """Test client initialization with credentials."""
        client = KalshiClient(email="test@test.com", password="password")
        assert client.email == "test@test.com"
        assert client.password == "password"

    def test_parse_market(self):
        """Test market data parsing."""
        client = KalshiClient()

        raw_data = {
            "ticker": "KXNFL-TEST",
            "event_ticker": "KXNFL-EVENT",
            "series_ticker": "KXNFL",
            "market_type": "binary",
            "title": "Test Market",
            "subtitle": "Test subtitle",
            "yes_bid": 0.55,
            "yes_ask": 0.58,
            "no_bid": 0.42,
            "no_ask": 0.45,
            "last_price": 0.56,
            "volume": 1000,
            "open_interest": 500,
            "status": "open",
            "close_time": "2024-09-05T20:00:00Z",
        }

        market = client._parse_market(raw_data)

        assert isinstance(market, KalshiMarket)
        assert market.ticker == "KXNFL-TEST"
        assert market.event_ticker == "KXNFL-EVENT"
        assert market.yes_bid == 0.55
        assert market.yes_ask == 0.58
        assert market.status == "open"
        assert market.close_time.year == 2024

    def test_parse_orders(self):
        """Test orderbook parsing."""
        client = KalshiClient()

        orders = [
            {"bid": 0.55, "bid_count": 100},
            {"bid": 0.54, "bid_count": 200},
            {"ask": 0.58, "ask_count": 150},
        ]

        bids = client._parse_orders(orders, "bids")
        asks = client._parse_orders(orders, "asks")

        assert len(bids) == 2
        assert bids[0] == (0.55, 100)
        assert len(asks) == 1
        assert asks[0] == (0.58, 150)

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = KalshiClient()

        with patch.object(client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"status": "ok"}
            result = await client.health_check()
            assert result is True
            mock_get.assert_called_once_with("/exchange/status")

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        client = KalshiClient()

        with patch.object(client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Connection error")
            result = await client.health_check()
            assert result is False


class TestKalshiMarket:
    """Tests for KalshiMarket model."""

    def test_market_creation(self):
        """Test market model creation."""
        market = KalshiMarket(
            ticker="TEST-TICKER",
            event_ticker="TEST-EVENT",
            market_type="binary",
            title="Test Market",
            status="open",
        )

        assert market.ticker == "TEST-TICKER"
        assert market.yes_bid is None
        assert market.status == "open"

    def test_market_with_all_fields(self):
        """Test market model with all fields."""
        close_time = datetime(2024, 9, 5, 20, 0, tzinfo=timezone.utc)

        market = KalshiMarket(
            ticker="TEST-TICKER",
            event_ticker="TEST-EVENT",
            series_ticker="TEST-SERIES",
            market_type="binary",
            title="Test Market",
            subtitle="Test Subtitle",
            yes_bid=0.55,
            yes_ask=0.58,
            no_bid=0.42,
            no_ask=0.45,
            last_price=0.56,
            volume=1000,
            open_interest=500,
            status="open",
            close_time=close_time,
        )

        assert market.series_ticker == "TEST-SERIES"
        assert market.yes_bid == 0.55
        assert market.volume == 1000
        assert market.close_time == close_time


class TestOrderBook:
    """Tests for OrderBook model."""

    def test_orderbook_creation(self):
        """Test orderbook model creation."""
        orderbook = OrderBook(
            ticker="TEST-TICKER",
            yes_bids=[(0.55, 100), (0.54, 200)],
            yes_asks=[(0.58, 150)],
            no_bids=[(0.42, 100)],
            no_asks=[(0.45, 100)],
        )

        assert orderbook.ticker == "TEST-TICKER"
        assert len(orderbook.yes_bids) == 2
        assert orderbook.yes_bids[0] == (0.55, 100)

    def test_orderbook_empty(self):
        """Test orderbook with empty lists."""
        orderbook = OrderBook(ticker="TEST-TICKER")

        assert orderbook.yes_bids == []
        assert orderbook.yes_asks == []
        assert orderbook.no_bids == []
        assert orderbook.no_asks == []
