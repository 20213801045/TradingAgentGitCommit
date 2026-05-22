"""Base class and helpers for deterministic prototype research agents."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from evidence.evidence_scorer import score_evidence
from evidence.temporal_checker import check_temporal_status
from models.schemas import ClaimEvidenceCommit, Evidence


class BaseAgent:
    """Base class for local mock agents that emit claim-evidence commits.

    Subclasses should keep their logic deterministic for now. The `analyze`
    method is the replacement point for future LLM or tool-augmented calls.
    """

    name: str = "BaseAgent"
    role: str = "base-agent"
    branch_name: str = "base-branch"

    @staticmethod
    def _utc_now() -> str:
        """Return the current UTC time in ISO-8601 format."""

        return datetime.now(timezone.utc).isoformat()

    def _make_evidence(
        self,
        content: str,
        source: str,
        source_type: str,
        timestamp: str,
        url: str | None = None,
        metric_name: str | None = None,
        metric_value: str | None = None,
    ) -> Evidence:
        """Create an evidence object with a generated identifier."""

        return Evidence(
            evidence_id=str(uuid4()),
            content=content,
            source=source,
            source_type=source_type,
            timestamp=timestamp,
            url=url,
            metric_name=metric_name,
            metric_value=metric_value,
        )

    def create_commit(
        self,
        claim: str,
        evidence: Evidence,
        confidence: str,
        risk_tag: str,
        time_horizon: str,
    ) -> ClaimEvidenceCommit:
        """Create a valid claim-evidence commit for this agent."""

        return ClaimEvidenceCommit(
            commit_id=str(uuid4()),
            agent_role=self.role,
            branch_name=self.branch_name,
            claim=claim,
            evidence=evidence,
            evidence_quality_score=score_evidence(evidence, claim),
            confidence=confidence,
            risk_tag=risk_tag,
            time_horizon=time_horizon,
            temporal_status=check_temporal_status(evidence, time_horizon),
            counter_evidence=None,
            created_at=self._utc_now(),
        )

    def analyze(
        self,
        input_data: dict[str, Any] | list[ClaimEvidenceCommit],
        workspace: Any,
    ) -> list[ClaimEvidenceCommit]:
        """Analyze input data and return structured claim-evidence commits."""

        raise NotImplementedError
