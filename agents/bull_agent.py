"""Bull-case research agent."""

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Workspace


class BullAgent(BaseAgent):
    """Creates bullish commits by reusing prior branch evidence."""

    name = "BullAgent"
    role = "bull-agent"
    branch_name = "bull-case"

    def analyze(
        self,
        input_data: dict[str, Any] | list[ClaimEvidenceCommit],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Build a bullish thesis from fundamental and technical commits."""

        del input_data
        previous_commits = _collect_source_commits(workspace)
        revenue_commit = _find_commit(previous_commits, "revenue")
        momentum_commit = _find_commit(previous_commits, "momentum")
        profitability_commit = _find_commit(previous_commits, "profitability")

        commits: list[ClaimEvidenceCommit] = []
        if revenue_commit is not None:
            commits.append(
                self.create_commit(
                    claim="正向收入增长支持偏多投资假设。",
                    evidence=revenue_commit.evidence,
                    confidence="medium",
                    risk_tag="bullish_growth",
                    time_horizon="12 months",
                )
            )
        if momentum_commit is not None:
            commits.append(
                self.create_commit(
                    claim="技术动量强化了偏多投资设定。",
                    evidence=momentum_commit.evidence,
                    confidence="medium",
                    risk_tag="bullish_momentum",
                    time_horizon="1-3 months",
                )
            )
        if profitability_commit is not None:
            commits.append(
                self.create_commit(
                    claim="稳定盈利能力提升了偏多观点的质量。",
                    evidence=profitability_commit.evidence,
                    confidence="medium",
                    risk_tag="quality_support",
                    time_horizon="12-24 months",
                )
            )

        return commits[:3]


def _collect_source_commits(workspace: Workspace) -> list[ClaimEvidenceCommit]:
    """Collect commits from branches the bull agent is allowed to reuse."""

    source_branches = (
        "fundamental-analysis",
        "financial-statement-analysis",
        "valuation-analysis",
        "industry-comparison",
        "macro-analysis",
        "technical-analysis",
        "backtest-analysis",
        "portfolio-review",
        "llm-analysis",
    )
    commits: list[ClaimEvidenceCommit] = []
    for branch_name in source_branches:
        branch = workspace.branches.get(branch_name)
        if branch is not None:
            commits.extend(branch.commits)
    return commits


def _find_commit(
    commits: list[ClaimEvidenceCommit],
    keyword: str,
) -> ClaimEvidenceCommit | None:
    """Find the first commit whose claim or evidence references a keyword."""

    lowered_keyword = keyword.lower()
    for commit in commits:
        haystack = (
            f"{commit.claim} {commit.risk_tag} {commit.evidence.content} "
            f"{commit.evidence.metric_name or ''}"
        ).lower()
        if lowered_keyword in haystack:
            return commit
    return None
