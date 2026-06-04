"""Research coordinator agent — plans, reviews, and audits the research workflow."""

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Evidence
from llm.base import LLMError, LLMMessage

import json
from datetime import datetime, timezone
from uuid import uuid4


class ResearchCoordinatorAgent(BaseAgent):
    """Plans the research, schedules agents, and captures pre/post-merge audit records."
    role = "research-coordinator"

    def analyze(self, input_data: dict, workspace: "Workspace") -> list[ClaimEvidenceCommit]:
        phase = input_data.get("phase", "initial_plan")
        company_data = input_data.get("company_data", {})
        ticker = company_data.get("ticker", workspace.ticker)

        if phase == "initial_plan":
            return self._initial_plan(ticker, company_data)
        elif phase == "pre_merge_review":
            return self._pre_merge_review(workspace, ticker)
        return []

    def _initial_plan(self, ticker: str, company_data: dict) -> list[ClaimEvidenceCommit]:
        return [self._make_commit(
            branch_name="research-coordination",
            claim=f"Research plan for {ticker}: fundamental, valuation, technical, macro, then debate-driven decision.",
            evidence=Evidence(
                evidence_id=uuid4().hex[:8],
                content=f"Coordinator plan for {ticker}",
                source="coordinator",
                source_type="meta",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            confidence="high",
            risk_tag="adrift",
            time_horizon="3-6 months",
        )]

    def _pre_merge_review(self, workspace: "Workspace", ticker: str) -> list[ClaimEvidenceCommit]:
        branch_names = list(workspace.branches.keys())
        total_commits = sum(len(b.commits) for b in workspace.branches.values())
        return [self._make_commit(
            branch_name="research-coordination",
            claim=f"Pre-merge review: {total_commits} commits across {len(branch_names)} branches for {ticker}.",
            evidence=Evidence(
                evidence_id=uuid4().hex[:8],
                content=f"Coverage: {', '.join(branch_names)}",
                source="coordinator",
                source_type="meta",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            confidence="high",
            risk_tag="adrift",
            time_horizon="3-6 months",
        )]