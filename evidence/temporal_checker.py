"""Temporal validity checks for evidence."""

from datetime import datetime, timezone

from config import EVIDENCE_RECENCY_THRESHOLDS_DAYS
from models.schemas import Evidence


class TemporalChecker:
    """Rule-based temporal validity checker for evidence."""

    def check(self, evidence: Evidence, time_horizon: str) -> str:
        """Return whether evidence is valid, stale, expired, or unknown."""

        del time_horizon
        evidence_time = _parse_timestamp(evidence.timestamp)
        if evidence_time is None:
            return "unknown"

        age_days = (datetime.now(timezone.utc) - evidence_time).days
        source_type = evidence.source_type

        if source_type in {"technical_indicator", "mock_technical_indicator", "backtest_result"}:
            if age_days > EVIDENCE_RECENCY_THRESHOLDS_DAYS["technical_expired"]:
                return "expired"
            if age_days > EVIDENCE_RECENCY_THRESHOLDS_DAYS["technical_stale"]:
                return "stale"
            return "valid"

        if source_type in {"reputable_news", "macro_data"}:
            if age_days > EVIDENCE_RECENCY_THRESHOLDS_DAYS["news_expired"]:
                return "expired"
            if age_days > EVIDENCE_RECENCY_THRESHOLDS_DAYS["news_stale"]:
                return "stale"
            return "valid"

        if source_type in {
            "official_report",
            "financial_data_provider",
            "mock_financial_metric",
            "valuation_model",
            "industry_benchmark",
            "portfolio_model",
        }:
            if age_days > EVIDENCE_RECENCY_THRESHOLDS_DAYS["financial_expired"]:
                return "expired"
            if age_days > EVIDENCE_RECENCY_THRESHOLDS_DAYS["financial_stale"]:
                return "stale"
            return "valid"

        if age_days > EVIDENCE_RECENCY_THRESHOLDS_DAYS["default_stale"]:
            return "stale"
        return "valid"


def check_temporal_status(evidence: Evidence, time_horizon: str = "") -> str:
    """Compatibility wrapper for temporal evidence checks."""

    return TemporalChecker().check(evidence, time_horizon)


def _parse_timestamp(timestamp: str) -> datetime | None:
    """Parse common ISO timestamp forms into timezone-aware UTC datetimes."""

    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
