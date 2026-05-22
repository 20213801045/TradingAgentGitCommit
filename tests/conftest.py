"""Shared test helpers for EVIR."""

from collections.abc import Iterable

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
from models.schemas import ClaimEvidenceCommit, Workspace


BRANCH_DESCRIPTIONS = {
    "fundamental-analysis": "Financial quality, profitability, and cash generation.",
    "technical-analysis": "Price action, trend, and market timing evidence.",
    "bull-case": "Upside catalysts and positive thesis evidence.",
    "bear-case": "Downside thesis and counter-evidence.",
    "risk-review": "Key risks, fragilities, and conditions for revision.",
    "counter-evidence": "Counter-evidence checks for important positive claims.",
}


def build_workspace_through_counter_evidence() -> Workspace:
    """Build a deterministic AAPL workspace through counter-evidence generation."""

    company_data = get_mock_company_data("AAPL")
    workspace = create_workspace(
        ticker=company_data["ticker"],
        company_name=company_data["company_name"],
        research_question=(
            "Should AAPL be considered for inclusion in a long-term investment watchlist?"
        ),
    )
    for branch_name, description in BRANCH_DESCRIPTIONS.items():
        create_branch(workspace, branch_name, description)

    _add_commits(workspace, FundamentalAgent().analyze(company_data, workspace))
    _add_commits(workspace, TechnicalAgent().analyze(company_data, workspace))
    _add_commits(workspace, BullAgent().analyze({}, workspace))
    _add_commits(workspace, BearAgent().analyze({}, workspace))
    _add_commits(workspace, RiskAgent().analyze({}, workspace))

    process_workspace_evidence(workspace)
    _add_commits(workspace, CounterEvidenceAgent().analyze(workspace))
    process_workspace_evidence(workspace)
    return workspace


def _add_commits(
    workspace: Workspace,
    commits: Iterable[ClaimEvidenceCommit],
) -> None:
    """Append commits to their branches."""

    for commit in commits:
        add_commit(workspace, commit.branch_name, commit)

