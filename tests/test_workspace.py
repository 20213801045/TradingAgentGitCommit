"""Tests for workspace and branch memory helpers."""

from agents import FundamentalAgent
from data import get_mock_company_data
from memory.workspace import (
    add_commit,
    create_branch,
    create_workspace,
    load_workspace,
    save_workspace,
)


def test_workspace_branch_commit_save_and_load(tmp_path) -> None:
    """A workspace should round-trip through local JSON storage."""

    company_data = get_mock_company_data("AAPL")
    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
    )
    create_branch(
        workspace,
        "fundamental-analysis",
        "Fundamental analysis.",
    )
    commit = FundamentalAgent().analyze(company_data, workspace)[0]
    add_commit(workspace, "fundamental-analysis", commit)

    workspace_path = save_workspace(workspace, tmp_path)
    loaded_workspace = load_workspace("AAPL", tmp_path)

    assert workspace_path.exists()
    assert loaded_workspace.ticker == "AAPL"
    assert "fundamental-analysis" in loaded_workspace.branches
    assert loaded_workspace.branches["fundamental-analysis"].commits[0].commit_id == (
        commit.commit_id
    )
