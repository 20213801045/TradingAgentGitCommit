"""Deterministic evidence quality scoring logic."""

import re
from datetime import datetime, timezone

from config import EVIDENCE_RECENCY_THRESHOLDS_DAYS
from models.schemas import Evidence


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
}

SOURCE_RELIABILITY_SCORES = {
    "official_report": 0.35,
    "financial_data_provider": 0.30,
    "reputable_news": 0.25,
    "technical_indicator": 0.25,
    "valuation_model": 0.25,
    "industry_benchmark": 0.25,
    "macro_data": 0.25,
    "backtest_result": 0.25,
    "portfolio_model": 0.25,
    "mock_financial_metric": 0.25,
    "mock_technical_indicator": 0.20,
    "social_media": 0.10,
}


class EvidenceScorer:
    """Rule-based evidence quality scorer.

    The score is deterministic and intentionally simple so it can be replaced
    later with a stronger evidence evaluator without changing agent outputs.
    """

    def score(self, evidence: Evidence, claim: str) -> float:
        """Score evidence quality for a claim on a 0.0 to 1.0 scale."""

        score = 0.0
        score += self._score_source_reliability(evidence)
        score += self._score_specificity(evidence)
        score += self._score_relevance(evidence, claim)
        score += self._score_recency(evidence)
        return min(round(score, 2), 1.0)

    def _score_source_reliability(self, evidence: Evidence) -> float:
        """Score evidence source reliability."""

        return SOURCE_RELIABILITY_SCORES.get(evidence.source_type, 0.05)

    def _score_specificity(self, evidence: Evidence) -> float:
        """Score whether evidence contains specific metrics or numbers."""

        if evidence.metric_name and evidence.metric_value:
            return 0.25
        if re.search(r"\d", evidence.content):
            return 0.15
        return 0.05

    def _score_relevance(self, evidence: Evidence, claim: str) -> float:
        """Score token overlap between the claim and evidence content."""

        claim_tokens = _meaningful_tokens(claim)
        evidence_tokens = _meaningful_tokens(evidence.content)
        if claim_tokens.intersection(evidence_tokens):
            return 0.20
        return 0.05

    def _score_recency(self, evidence: Evidence) -> float:
        """Score evidence recency using deterministic date buckets."""

        evidence_time = _parse_timestamp(evidence.timestamp)
        if evidence_time is None:
            return 0.05

        age_days = (datetime.now(timezone.utc) - evidence_time).days
        if age_days <= EVIDENCE_RECENCY_THRESHOLDS_DAYS["news_expired"]:
            return 0.20
        if age_days <= EVIDENCE_RECENCY_THRESHOLDS_DAYS["financial_stale"]:
            return 0.10
        return 0.05


def score_evidence(evidence: Evidence, claim: str = "") -> float:
    """Compatibility wrapper for scoring evidence."""

    return EvidenceScorer().score(evidence, claim)


def _meaningful_tokens(text: str) -> set[str]:
    """Tokenize text and remove common stopwords."""

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def _parse_timestamp(timestamp: str) -> datetime | None:
    """Parse common ISO timestamp forms into timezone-aware UTC datetimes."""

    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
