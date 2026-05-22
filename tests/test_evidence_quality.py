"""Tests for evidence scoring and temporal checks."""

from datetime import datetime, timedelta, timezone

from evidence.evidence_scorer import EvidenceScorer
from evidence.temporal_checker import TemporalChecker
from models.schemas import Evidence


def test_evidence_scorer_scores_specific_relevant_recent_evidence() -> None:
    """Specific, relevant, recent financial evidence should score well."""

    evidence = Evidence(
        evidence_id="ev-1",
        content="Year-over-year revenue growth is 8%.",
        source="Mock Financial Metrics",
        source_type="mock_financial_metric",
        timestamp=datetime.now(timezone.utc).date().isoformat(),
        metric_name="revenue_growth_yoy",
        metric_value="8%",
    )

    score = EvidenceScorer().score(
        evidence,
        "The company shows positive revenue growth.",
    )

    assert 0.0 <= score <= 1.0
    assert score >= 0.85


def test_temporal_checker_expires_old_mock_technical_indicator() -> None:
    """Mock technical indicators older than 14 days should be expired."""

    evidence = Evidence(
        evidence_id="ev-2",
        content="RSI is 62.",
        source="Mock Momentum Indicator Feed",
        source_type="mock_technical_indicator",
        timestamp=(datetime.now(timezone.utc) - timedelta(days=15)).date().isoformat(),
        metric_name="rsi",
        metric_value="62",
    )

    assert TemporalChecker().check(evidence, "1-3 months") == "expired"


def test_temporal_checker_accepts_recent_mock_financial_metric() -> None:
    """Recent mock financial metrics should remain valid."""

    evidence = Evidence(
        evidence_id="ev-3",
        content="Forward P/E is 28.",
        source="Mock Valuation Metrics",
        source_type="mock_financial_metric",
        timestamp=datetime.now(timezone.utc).date().isoformat(),
        metric_name="forward_pe",
        metric_value="28",
    )

    assert TemporalChecker().check(evidence, "6-12 months") == "valid"

