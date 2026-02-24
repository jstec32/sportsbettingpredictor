"""Order management for trade execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger


class OrderStatus(Enum):
    """Order status states."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side."""

    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"


@dataclass
class Order:
    """A trading order."""

    order_id: str
    market_ticker: str
    side: OrderSide
    quantity: int
    price: float  # Limit price (0-1 for Kalshi)

    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    filled_quantity: int = 0
    average_fill_price: Optional[float] = None

    external_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "market_ticker": self.market_ticker,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "external_id": self.external_id,
        }


class OrderManager:
    """Manage orders for trade execution.

    This is a placeholder for actual order management that would
    integrate with Kalshi's trading API. For paper trading,
    orders are simulated.
    """

    def __init__(self, paper_mode: bool = True) -> None:
        """Initialize order manager.

        Args:
            paper_mode: Whether to simulate orders (paper trading).
        """
        self.paper_mode = paper_mode
        self._orders: Dict[str, Order] = {}
        self._next_order_id = 1

    def create_order(
        self,
        market_ticker: str,
        side: OrderSide,
        quantity: int,
        price: float,
        metadata: Dict[str, Any] | None = None,
    ) -> Order:
        """Create a new order.

        Args:
            market_ticker: Kalshi market ticker.
            side: Order side.
            quantity: Number of contracts.
            price: Limit price.
            metadata: Optional metadata.

        Returns:
            Created Order object.
        """
        order_id = f"ORD-{self._next_order_id:06d}"
        self._next_order_id += 1

        order = Order(
            order_id=order_id,
            market_ticker=market_ticker,
            side=side,
            quantity=quantity,
            price=price,
            metadata=metadata or {},
        )

        self._orders[order_id] = order

        logger.info(
            f"Created order {order_id}: {side.value} {quantity} @ {price:.2f} "
            f"on {market_ticker}"
        )

        return order

    def submit_order(self, order_id: str) -> bool:
        """Submit an order for execution.

        Args:
            order_id: Order ID to submit.

        Returns:
            True if submitted successfully.
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order {order_id} not found")
            return False

        if order.status != OrderStatus.PENDING:
            logger.error(f"Order {order_id} not in pending state")
            return False

        if self.paper_mode:
            # Simulate immediate fill
            return self._simulate_fill(order)
        else:
            # Would submit to Kalshi API
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now(timezone.utc)
            logger.info(f"Submitted order {order_id}")
            return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully.
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order {order_id} not found")
            return False

        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            logger.error(f"Order {order_id} cannot be cancelled")
            return False

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now(timezone.utc)

        logger.info(f"Cancelled order {order_id}")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID.

        Args:
            order_id: Order ID.

        Returns:
            Order or None if not found.
        """
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all open (unfilled) orders.

        Returns:
            List of open orders.
        """
        return [
            o
            for o in self._orders.values()
            if o.status
            in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)
        ]

    def get_orders_for_market(self, market_ticker: str) -> List[Order]:
        """Get all orders for a market.

        Args:
            market_ticker: Market ticker.

        Returns:
            List of orders.
        """
        return [o for o in self._orders.values() if o.market_ticker == market_ticker]

    def _simulate_fill(self, order: Order) -> bool:
        """Simulate order fill for paper trading.

        Args:
            order: Order to fill.

        Returns:
            True if filled.
        """
        # In paper mode, assume orders fill at limit price
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc)
        order.filled_quantity = order.quantity
        order.average_fill_price = order.price

        logger.info(f"Order {order.order_id} filled: {order.quantity} @ {order.price:.2f}")

        return True

    def calculate_position_value(
        self,
        orders: List[Order],
        current_price: float,
    ) -> Dict[str, float]:
        """Calculate position value from filled orders.

        Args:
            orders: List of filled orders.
            current_price: Current market price.

        Returns:
            Dictionary with position details.
        """
        total_yes_contracts = 0
        total_no_contracts = 0
        total_cost = 0.0

        for order in orders:
            if order.status != OrderStatus.FILLED:
                continue

            fill_price = order.average_fill_price or order.price

            if order.side == OrderSide.BUY_YES:
                total_yes_contracts += order.filled_quantity
                total_cost += order.filled_quantity * fill_price
            elif order.side == OrderSide.SELL_YES:
                total_yes_contracts -= order.filled_quantity
                total_cost -= order.filled_quantity * fill_price
            elif order.side == OrderSide.BUY_NO:
                total_no_contracts += order.filled_quantity
                total_cost += order.filled_quantity * (1 - fill_price)
            elif order.side == OrderSide.SELL_NO:
                total_no_contracts -= order.filled_quantity
                total_cost -= order.filled_quantity * (1 - fill_price)

        # Calculate current value
        yes_value = total_yes_contracts * current_price
        no_value = total_no_contracts * (1 - current_price)
        total_value = yes_value + no_value

        return {
            "yes_contracts": total_yes_contracts,
            "no_contracts": total_no_contracts,
            "total_cost": total_cost,
            "current_value": total_value,
            "unrealized_pnl": total_value - total_cost,
        }

    def clear_filled_orders(self) -> int:
        """Remove filled orders from memory.

        Returns:
            Number of orders removed.
        """
        filled_ids = [
            oid
            for oid, o in self._orders.items()
            if o.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED)
        ]

        for oid in filled_ids:
            del self._orders[oid]

        return len(filled_ids)
