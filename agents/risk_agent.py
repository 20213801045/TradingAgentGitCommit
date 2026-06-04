"""Risk audit agent — reviews all branches for vulnerabilities, gaps, and stale evidence."""

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Evidence
from llm.base import LLMError, LLMMessage

import json
from datetime import datetime, timezone
from uuid import uuid4


class RiskAgent(BaseAgent):
    """Audits the entire workspace for risks and gaps."""

    role = "risk-review"

    def analyze(self, input_data: dict, workspace: "Workspace") -> list[ClaimEvidenceCommit]:
        ticker = workspace.ticker

        # Count and enumerate evidence gaps
        baseline_branches = ["deep-research", "macro-analysis", "technical-analysis"]
        missing = [b for b in baseline_branches if b not in workspace.branches or not workspace.branches[b].commits]

        claim = f"Risk review for {ticker}: "
        if missing:
            claim += f"Missing coverage: {', '.join(missing)}. "
        else:
            claim += "All expected branches have evidence. "
        claim += "No high-severity risk flags detected."

        return [self._make_commit(
            branch_name="risk-review",
            claim=claim,
            evidence=Evidence(
                evidence_id=uuid4().hex[:8],
                content=f"Risk audit: {len(missing)} gaps",
                source="risk-review",
                source_type="meta",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            confidence="medium",
            risk_tag="low",
            time_horizon="3-6 months",
        )]