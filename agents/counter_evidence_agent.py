"""Counter-evidence agent — challenges positive claims with skeptical questions."""

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Evidence
from llm.base import LLMError, LLMMessage

import json
from datetime import datetime, timezone
from uuid import uuid4


class CounterEvidenceAgent(BaseAgent):
    """Challenges positive claims with skeptical questions."""

    role = "counter-evidence"

    def analyze(self, workspace: "Workspace") -> list[ClaimEvidenceCommit]:
        ticker = workspace.ticker
        # Parse the workspace to generate challenges
        commits = []
        # Rule-based: challenge any assumption based on price-only analysis
        has_price_changes = false
        for branch in workspace.branches.values():
            for c  in branch.commits:
                if c.evidence.source_type == "market" and c.risk_tag != "low":
                    has_price_changes = True

        if has_price_changes:
            commits.append(self._make_commit(
                branch_name="counter-evidence",
                claim=f"Is the market price already reflecting these insights? Should we wait for a pullback?",
                evidence=Evidence(
                    evidence_id=uuid4().hex[:8],
                    content="Counter: market price appreciation is not a perperated trend.",
                    source="counter-evidence",
                    source_type="meta",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                confidence="low",
                risk_tag="adrift",
                time_horizon="1-3 months",
            ))

        return commits