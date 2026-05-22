"""Tests for deterministic agent outputs."""

from agents import (
    BearAgent,
    BullAgent,
    CounterEvidenceAgent,
    FundamentalAgent,
    RiskAgent,
    TechnicalAgent,
)
from data import get_mock_company_data
from evidence import process_workspace_evidence
from memory.workspace import add_commit, create_branch, create_workspace
from models.schemas import ClaimEvidenceCommit


BRANCHES = {
    "fundamental-analysis": "Fundamental analysis.",
    "technical-analysis": "Technical analysis.",
    "bull-case": "Bull case.",
    "bear-case": "Bear case.",
    "risk-review": "Risk review.",
    "counter-evidence": "Counter-evidence.",
}


def test_agents_return_claim_evidence_commits() -> None:
    """Every research agent should emit structured ClaimEvidenceCommit objects."""

    company_data = get_mock_company_data("AAPL")
    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
    )
    for branch_name, description in BRANCHES.items():
        create_branch(workspace, branch_name, description)

    fundamental_commits = FundamentalAgent().analyze(company_data, workspace)
    technical_commits = TechnicalAgent().analyze(company_data, workspace)
    for commit in [*fundamental_commits, *technical_commits]:
        add_commit(workspace, commit.branch_name, commit)

    bull_commits = BullAgent().analyze({}, workspace)
    bear_commits = BearAgent().analyze({}, workspace)
    for commit in [*bull_commits, *bear_commits]:
        add_commit(workspace, commit.branch_name, commit)

    risk_commits = RiskAgent().analyze({}, workspace)
    for commit in risk_commits:
        add_commit(workspace, commit.branch_name, commit)

    process_workspace_evidence(workspace)
    counter_commits = CounterEvidenceAgent().analyze(workspace)

    all_commits = [
        *fundamental_commits,
        *technical_commits,
        *bull_commits,
        *bear_commits,
        *risk_commits,
        *counter_commits,
    ]

    assert all_commits
    assert all(isinstance(commit, ClaimEvidenceCommit) for commit in all_commits)
    assert all(commit.claim for commit in all_commits)
    assert all(commit.evidence.content for commit in all_commits)
