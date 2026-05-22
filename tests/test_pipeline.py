"""Full deterministic mock pipeline test."""

from pathlib import Path

from agents import (
    BearAgent,
    BullAgent,
    CounterEvidenceAgent,
    FundamentalAgent,
    MergeAgent,
    ReportAgent,
    RiskAgent,
    TechnicalAgent,
)
from data import get_mock_company_data
from evidence import process_workspace_evidence
from memory.workspace import add_commit, create_branch, create_workspace


def test_full_mock_pipeline_creates_report(tmp_path: Path) -> None:
    """The local mock pipeline should run without LLM or yfinance calls."""

    company_data = get_mock_company_data("AAPL")
    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
    )
    for branch_name in [
        "fundamental-analysis",
        "technical-analysis",
        "bull-case",
        "bear-case",
        "risk-review",
        "counter-evidence",
    ]:
        create_branch(workspace, branch_name, branch_name)

    for agent, input_data in [
        (FundamentalAgent(), company_data),
        (TechnicalAgent(), company_data),
        (BullAgent(), {}),
        (BearAgent(), {}),
        (RiskAgent(), {}),
    ]:
        for commit in agent.analyze(input_data, workspace):
            add_commit(workspace, commit.branch_name, commit)

    process_workspace_evidence(workspace)
    for commit in CounterEvidenceAgent().analyze(workspace):
        add_commit(workspace, commit.branch_name, commit)
    process_workspace_evidence(workspace)

    merge_result = MergeAgent().merge(workspace)
    report = ReportAgent().generate_report(workspace, merge_result)
    markdown_path, _ = ReportAgent().save_report(report, tmp_path)

    assert merge_result.final_recommendation
    assert Path(markdown_path).exists()
    assert "投资研究报告：AAPL" in report.markdown_report
