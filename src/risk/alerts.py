"""Alert management for risk events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from loguru import logger


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""

    DRAWDOWN = "drawdown"
    EXPOSURE = "exposure"
    LOSING_STREAK = "losing_streak"
    DAILY_LOSS = "daily_loss"
    API_ERROR = "api_error"
    DATA_QUALITY = "data_quality"
    MODEL_DRIFT = "model_drift"
    SYSTEM = "system"


@dataclass
class Alert:
    """An alert event."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
        }


class AlertManager:
    """Manage and dispatch alerts."""

    def __init__(self) -> None:
        """Initialize alert manager."""
        self._alerts: List[Alert] = []
        self._next_alert_id = 1
        self._handlers: Dict[AlertSeverity, List[Callable[[Alert], None]]] = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.ERROR: [],
            AlertSeverity.CRITICAL: [],
        }

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any] | None = None,
    ) -> Alert:
        """Create and dispatch an alert.

        Args:
            alert_type: Type of alert.
            severity: Alert severity.
            message: Alert message.
            details: Optional details.

        Returns:
            Created Alert object.
        """
        alert_id = f"ALT-{self._next_alert_id:06d}"
        self._next_alert_id += 1

        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details or {},
        )

        self._alerts.append(alert)

        # Log the alert
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }[severity]

        log_func(f"[{alert_type.value}] {message}")

        # Dispatch to handlers
        self._dispatch(alert)

        return alert

    def register_handler(
        self,
        severity: AlertSeverity,
        handler: Callable[[Alert], None],
    ) -> None:
        """Register an alert handler.

        Args:
            severity: Minimum severity to handle.
            handler: Handler function.
        """
        self._handlers[severity].append(handler)

    def _dispatch(self, alert: Alert) -> None:
        """Dispatch alert to registered handlers.

        Args:
            alert: Alert to dispatch.
        """
        # Get all handlers at or above this severity
        severities = list(AlertSeverity)
        alert_idx = severities.index(alert.severity)

        for sev in severities[alert_idx:]:
            for handler in self._handlers[sev]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge.

        Returns:
            True if acknowledged.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now(timezone.utc)
                return True
        return False

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
    ) -> List[Alert]:
        """Get active (unacknowledged) alerts.

        Args:
            severity: Filter by minimum severity.
            alert_type: Filter by type.

        Returns:
            List of active alerts.
        """
        alerts = [a for a in self._alerts if not a.acknowledged]

        if severity:
            severities = list(AlertSeverity)
            min_idx = severities.index(severity)
            alerts = [a for a in alerts if severities.index(a.severity) >= min_idx]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return alerts

    def get_recent_alerts(
        self,
        count: int = 10,
        include_acknowledged: bool = False,
    ) -> List[Alert]:
        """Get recent alerts.

        Args:
            count: Number of alerts to return.
            include_acknowledged: Include acknowledged alerts.

        Returns:
            List of recent alerts.
        """
        alerts = self._alerts
        if not include_acknowledged:
            alerts = [a for a in alerts if not a.acknowledged]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)[:count]

    # Convenience methods for common alerts
    def alert_drawdown(
        self,
        current_drawdown: float,
        threshold: float,
    ) -> Alert:
        """Create a drawdown alert.

        Args:
            current_drawdown: Current drawdown percentage.
            threshold: Threshold that was exceeded.

        Returns:
            Created alert.
        """
        severity = AlertSeverity.CRITICAL if current_drawdown > 0.15 else AlertSeverity.WARNING

        return self.create_alert(
            alert_type=AlertType.DRAWDOWN,
            severity=severity,
            message=f"Drawdown at {current_drawdown:.1%} (threshold: {threshold:.0%})",
            details={
                "current_drawdown": current_drawdown,
                "threshold": threshold,
            },
        )

    def alert_losing_streak(self, streak_length: int) -> Alert:
        """Create a losing streak alert.

        Args:
            streak_length: Number of consecutive losses.

        Returns:
            Created alert.
        """
        severity = (
            AlertSeverity.ERROR
            if streak_length >= 7
            else AlertSeverity.WARNING
            if streak_length >= 5
            else AlertSeverity.INFO
        )

        return self.create_alert(
            alert_type=AlertType.LOSING_STREAK,
            severity=severity,
            message=f"Losing streak: {streak_length} consecutive losses",
            details={"streak_length": streak_length},
        )

    def alert_api_error(
        self,
        api_name: str,
        error_message: str,
    ) -> Alert:
        """Create an API error alert.

        Args:
            api_name: Name of the API.
            error_message: Error message.

        Returns:
            Created alert.
        """
        return self.create_alert(
            alert_type=AlertType.API_ERROR,
            severity=AlertSeverity.ERROR,
            message=f"API error ({api_name}): {error_message}",
            details={"api_name": api_name, "error": error_message},
        )

    def alert_data_quality(
        self,
        issue: str,
        affected_records: int = 0,
    ) -> Alert:
        """Create a data quality alert.

        Args:
            issue: Description of the issue.
            affected_records: Number of affected records.

        Returns:
            Created alert.
        """
        return self.create_alert(
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            message=f"Data quality issue: {issue}",
            details={"issue": issue, "affected_records": affected_records},
        )

    def clear_alerts(self, before: Optional[datetime] = None) -> int:
        """Clear old alerts.

        Args:
            before: Clear alerts before this time.

        Returns:
            Number of alerts cleared.
        """
        if before is None:
            count = len(self._alerts)
            self._alerts.clear()
            return count

        original_count = len(self._alerts)
        self._alerts = [a for a in self._alerts if a.created_at >= before]
        return original_count - len(self._alerts)
