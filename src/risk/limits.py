"""Risk limit checking and enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger

from src.utils.config import get_risk_config


class LimitStatus(Enum):
    """Limit check status."""

    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"
    HALT = "halt"


@dataclass
class LimitCheck:
    """Result of a limit check."""

    limit_name: str
    status: LimitStatus
    current_value: float
    limit_value: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "limit_name": self.limit_name,
            "status": self.status.value,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "message": self.message,
        }


class RiskLimits:
    """Check and enforce risk limits."""

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize risk limits.

        Args:
            config: Optional configuration override.
        """
        self._config = config or get_risk_config()

    def check_all(
        self,
        bankroll: float,
        total_exposure: float,
        daily_bets: int = 0,
        current_drawdown: float = 0,
        pending_bet_size: float = 0,
    ) -> List[LimitCheck]:
        """Run all limit checks.

        Args:
            bankroll: Current bankroll.
            total_exposure: Current total exposure.
            daily_bets: Number of bets today.
            current_drawdown: Current drawdown percentage.
            pending_bet_size: Size of pending bet (if any).

        Returns:
            List of LimitCheck results.
        """
        checks = []

        # Exposure limit
        checks.append(self.check_exposure(bankroll, total_exposure, pending_bet_size))

        # Single bet limit
        if pending_bet_size > 0:
            checks.append(self.check_single_bet(bankroll, pending_bet_size))

        # Daily bet limit
        checks.append(self.check_daily_bets(daily_bets))

        # Drawdown limit
        checks.append(self.check_drawdown(current_drawdown))

        # Minimum bankroll
        checks.append(self.check_minimum_bankroll(bankroll))

        return checks

    def check_exposure(
        self,
        bankroll: float,
        current_exposure: float,
        pending_bet: float = 0,
    ) -> LimitCheck:
        """Check total exposure limit.

        Args:
            bankroll: Current bankroll.
            current_exposure: Current total exposure.
            pending_bet: Pending bet amount.

        Returns:
            LimitCheck result.
        """
        max_exposure_pct = self._config.max_total_exposure
        max_exposure = bankroll * max_exposure_pct

        new_exposure = current_exposure + pending_bet
        exposure_pct = new_exposure / bankroll if bankroll > 0 else 1

        if new_exposure > max_exposure:
            status = LimitStatus.EXCEEDED
            message = f"Exposure {exposure_pct:.1%} would exceed limit {max_exposure_pct:.0%}"
        elif exposure_pct > max_exposure_pct * 0.8:
            status = LimitStatus.WARNING
            message = f"Exposure at {exposure_pct:.1%} approaching limit"
        else:
            status = LimitStatus.OK
            message = f"Exposure {exposure_pct:.1%} within limit"

        return LimitCheck(
            limit_name="total_exposure",
            status=status,
            current_value=exposure_pct,
            limit_value=max_exposure_pct,
            message=message,
        )

    def check_single_bet(
        self,
        bankroll: float,
        bet_size: float,
    ) -> LimitCheck:
        """Check single bet size limit.

        Args:
            bankroll: Current bankroll.
            bet_size: Proposed bet size.

        Returns:
            LimitCheck result.
        """
        max_bet_pct = self._config.max_bet_pct
        max_bet = bankroll * max_bet_pct

        bet_pct = bet_size / bankroll if bankroll > 0 else 1

        if bet_size > max_bet:
            status = LimitStatus.EXCEEDED
            message = f"Bet size {bet_pct:.1%} exceeds limit {max_bet_pct:.0%}"
        else:
            status = LimitStatus.OK
            message = f"Bet size {bet_pct:.1%} within limit"

        return LimitCheck(
            limit_name="single_bet",
            status=status,
            current_value=bet_pct,
            limit_value=max_bet_pct,
            message=message,
        )

    def check_daily_bets(self, daily_bets: int) -> LimitCheck:
        """Check daily bet count limit.

        Args:
            daily_bets: Number of bets placed today.

        Returns:
            LimitCheck result.
        """
        max_daily = self._config.max_daily_bets

        if daily_bets >= max_daily:
            status = LimitStatus.EXCEEDED
            message = f"Daily bet limit reached ({daily_bets}/{max_daily})"
        elif daily_bets >= max_daily * 0.8:
            status = LimitStatus.WARNING
            message = f"Approaching daily limit ({daily_bets}/{max_daily})"
        else:
            status = LimitStatus.OK
            message = f"Daily bets: {daily_bets}/{max_daily}"

        return LimitCheck(
            limit_name="daily_bets",
            status=status,
            current_value=float(daily_bets),
            limit_value=float(max_daily),
            message=message,
        )

    def check_drawdown(self, current_drawdown: float) -> LimitCheck:
        """Check drawdown limits.

        Args:
            current_drawdown: Current drawdown percentage.

        Returns:
            LimitCheck result.
        """
        halt_threshold = self._config.halt_threshold
        reduce_threshold = getattr(self._config, "reduce_size_threshold", halt_threshold * 0.75)
        warning_threshold = getattr(self._config, "warning_threshold", halt_threshold * 0.5)

        if current_drawdown >= halt_threshold:
            status = LimitStatus.HALT
            message = (
                f"HALT: Drawdown {current_drawdown:.1%} exceeds halt threshold {halt_threshold:.0%}"
            )
        elif current_drawdown >= reduce_threshold:
            status = LimitStatus.EXCEEDED
            message = f"Reduce size: Drawdown {current_drawdown:.1%} exceeds threshold"
        elif current_drawdown >= warning_threshold:
            status = LimitStatus.WARNING
            message = f"Warning: Drawdown at {current_drawdown:.1%}"
        else:
            status = LimitStatus.OK
            message = f"Drawdown {current_drawdown:.1%} within limits"

        return LimitCheck(
            limit_name="drawdown",
            status=status,
            current_value=current_drawdown,
            limit_value=halt_threshold,
            message=message,
        )

    def check_minimum_bankroll(self, bankroll: float) -> LimitCheck:
        """Check minimum operational bankroll.

        Args:
            bankroll: Current bankroll.

        Returns:
            LimitCheck result.
        """
        min_bankroll = self._config.min_operational_bankroll

        if bankroll < min_bankroll:
            status = LimitStatus.HALT
            message = f"HALT: Bankroll ${bankroll:.2f} below minimum ${min_bankroll:.2f}"
        elif bankroll < min_bankroll * 1.2:
            status = LimitStatus.WARNING
            message = f"Warning: Bankroll approaching minimum"
        else:
            status = LimitStatus.OK
            message = f"Bankroll ${bankroll:.2f} above minimum"

        return LimitCheck(
            limit_name="minimum_bankroll",
            status=status,
            current_value=bankroll,
            limit_value=min_bankroll,
            message=message,
        )

    def can_place_bet(
        self,
        bankroll: float,
        total_exposure: float,
        bet_size: float,
        daily_bets: int,
        current_drawdown: float,
    ) -> Tuple[bool, str]:
        """Check if a bet can be placed.

        Args:
            bankroll: Current bankroll.
            total_exposure: Current total exposure.
            bet_size: Proposed bet size.
            daily_bets: Number of bets today.
            current_drawdown: Current drawdown.

        Returns:
            Tuple of (can_place, reason).
        """
        checks = self.check_all(
            bankroll=bankroll,
            total_exposure=total_exposure,
            daily_bets=daily_bets,
            current_drawdown=current_drawdown,
            pending_bet_size=bet_size,
        )

        # Check for any halt or exceeded status
        for check in checks:
            if check.status == LimitStatus.HALT:
                return False, f"HALT: {check.message}"
            if check.status == LimitStatus.EXCEEDED:
                return False, f"Limit exceeded: {check.message}"

        return True, "All limits OK"

    def get_position_size_multiplier(self, current_drawdown: float) -> float:
        """Get position size multiplier based on drawdown.

        Args:
            current_drawdown: Current drawdown percentage.

        Returns:
            Multiplier (0.0 to 1.0).
        """
        halt_threshold = self._config.halt_threshold
        reduce_threshold = getattr(self._config, "reduce_size_threshold", halt_threshold * 0.75)

        if current_drawdown >= halt_threshold:
            return 0.0
        elif current_drawdown >= reduce_threshold:
            # Linear reduction from reduce threshold to halt
            reduction_range = halt_threshold - reduce_threshold
            progress = (current_drawdown - reduce_threshold) / reduction_range
            return max(0.25, 1.0 - progress * 0.75)
        else:
            return 1.0
