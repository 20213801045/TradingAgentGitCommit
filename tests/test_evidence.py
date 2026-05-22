"""Tests for evidence quality scoring and temporal validity."""

from datetime import datetime, timedelta, timezone

from evidence.evidence_scorer import EvidenceScorer
from evidence.temporal_checker import TemporalChecker
from models.schemas import Evidence


def test_evidence_scorer_bounds_and_quality_ordering() -> None:
    """Specific financial metric evidence should score above unknown evidence."""

    recent_date = datetime.now(timezone.utc).date().isoformat()
    strong_evidence = Evidence(
        evidence_id="strong",
        content="Revenue growth is 8%.",
        source="Mock Financial Metrics",
        source_type="mock_financial_metric",
        timestamp=recent_date,
        metric_name="revenue_growth_yoy",
        metric_value="8%",
    )
    unknown_evidence = Evidence(
        evidence_id="unknown",
        content="A vague market comment.",
        source="Unknown",
        source_type="unknown",
        timestamp="not-a-date",
    )
    scorer = EvidenceScorer()

    strong_score = scorer.score(strong_evidence, "Revenue growth is positive.")
    unknown_score = scorer.score(unknown_evidence, "Revenue growth is positive.")

    assert 0.0 <= strong_score <= 1.0
    assert 0.0 <= unknown_score <= 1.0
    assert strong_score > unknown_score


def test_temporal_checker_statuses() -> None:
    """TemporalChecker should handle recent, old, and invalid timestamps."""

    checker = TemporalChecker()
    recent_date = datetime.now(timezone.utc).date().isoformat()
    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()

    recent_financial = Evidence(
        evidence_id="recent-financial",
        content="Revenue growth is 8%.",
        source="Mock Financial Metrics",
        source_type="mock_financial_metric",
        timestamp=recent_date,
    )
    old_technical = Evidence(
        evidence_id="old-technical",
        content="RSI is 62.",
        source="Mock Technical Feed",
        source_type="mock_technical_indicator",
        timestamp=old_date,
    )
    invalid_timestamp = Evidence(
        evidence_id="invalid",
        content="RSI is unknown.",
        source="Mock Technical Feed",
        source_type="mock_technical_indicator",
        timestamp="not-a-date",
    )

    assert checker.check(recent_financial, "12 months") == "valid"
    assert checker.check(old_technical, "1-3 months") in {"stale", "expired"}
    assert checker.check(invalid_timestamp, "1-3 months") == "unknown"
