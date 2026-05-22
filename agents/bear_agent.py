"""Bear-case research agent."""

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Workspace


class BearAgent(BaseAgent):
    """Creates cautious commits by reusing prior branch evidence."""

    name = "BearAgent"
    role = "bear-agent"
    branch_name = "bear-case"

    def analyze(
        self,
        input_data: dict[str, Any] | list[ClaimEvidenceCommit],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Build a bearish or cautious thesis from prior branch commits."""

        del input_data
        previous_commits = _collect_source_commits(workspace)
        valuation_commit = _find_commit(previous_commits, "valuation")
        volatility_commit = _find_commit(previous_commits, "volatility")
        resistance_commit = _find_commit(previous_commits, "resistance")

        commits: list[ClaimEvidenceCommit] = []
        if valuation_commit is not None:
            commits.append(
                self.create_commit(
                    claim="估值风险可能限制正向增长信号带来的上行空间。",
                    evidence=valuation_commit.evidence,
                    confidence="medium",
                    risk_tag="valuation_risk",
                    time_horizon="6-12 months",
                )
            )
        if volatility_commit is not None:
            commits.append(
                self.create_commit(
                    claim="波动率可能削弱短期投资设定。",
                    evidence=volatility_commit.evidence,
                    confidence="medium",
                    risk_tag="volatility_risk",
                    time_horizon="1-3 months",
                )
            )
        if resistance_commit is not None:
            commits.append(
                self.create_commit(
                    claim="交易区间上沿附近的阻力可能限制短期上行空间。",
                    evidence=resistance_commit.evidence,
                    confidence="medium",
                    risk_tag="technical_resistance",
                    time_horizon="1-3 months",
                )
            )

        return commits[:3]


def _collect_source_commits(workspace: Workspace) -> list[ClaimEvidenceCommit]:
    """Collect commits from branches the bear agent is allowed to reuse."""

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
