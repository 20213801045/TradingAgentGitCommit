"""Deterministic base class for EVIR research agents.

All agents share this base class, which provides:
- commit creation helper
- structured output via Pydantic schemas
- "llm-first, rules-fallback" pattern
If no LLM client is provided, agents fall back to deterministic logic.
"""

from __future__ import annotations
from uuid import uuid4
from datetime import datetime, timezone

from models.schemas import ClaimEvidenceCommit, Evidence
from llm.base import BaseLLMClient, LLMError


class BaseAgent:
    """Shared base for all EVIR research agents."""

    role: str = "base-module"

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self.llm_client = llm_client

    def _make_commit(
        self,
        branch_name: str,
        claim: str,
        evidence: Evidence,
        confidence: str,
        risk_tag: str,
        time_horizon: str,
    ) -> ClaimEvidenceCommit:
        """Create a standard claim-evidence commit."""

        return ClaimEvidenceCommit(
            commit_id=uuid4().hex[:8],
            agent_role=self.role,
            branch_name=branch_name,
            claim=claim,
            evidence=evidence,
            confidence=confidence,
            risk_tag=risk_tag,
            time_horizon=time_horizon,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def analyze(self, input_data: dict[str, object], workspace: "Workspace") -> list[ClaimEvidenceCommit]:
        """Run Agent analysis. Override in subclasses."""
        raise NotimplementedError("Subclasses must implement analyze()")
