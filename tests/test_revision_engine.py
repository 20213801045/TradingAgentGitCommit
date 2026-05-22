"""Tests for decision revision behavior."""

from pathlib import Path

from agents import MergeAgent
from data import get_mock_new_evidence
from revision import RevisionEngine, save_revision_report

from tests.conftest import build_workspace_through_counter_evidence


def test_revision_engine_revises_prior_claims_with_new_evidence() -> None:
    """New mock evidence should weaken, contradict, and support prior claims."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)
    revision_result = RevisionEngine().revise(
        workspace,
        merge_result,
        get_mock_new_evidence(),
    )

    statuses_by_claim = {
        record.original_claim: record.revision_status
        for record in revision_result.revision_records
    }

    assert revision_result.previous_recommendation == merge_result.final_recommendation
    assert revision_result.revised_recommendation in {"Buy", "Hold", "Sell", "Avoid"}
    assert any(
        "增长" in claim and status == "weakened"
        for claim, status in statuses_by_claim.items()
    )
    assert any(
        ("动量" in claim or "趋势" in claim)
        and status == "contradicted"
        for claim, status in statuses_by_claim.items()
    )
    assert any(
        "估值" in claim and status == "supported"
        for claim, status in statuses_by_claim.items()
    )
    assert "收入增长证据转弱，削弱成长性判断。" in revision_result.key_changes
    assert "高估值风险仍被新的远期市盈率证据支持。" in revision_result.key_changes


def test_save_revision_report_writes_markdown(tmp_path: Path) -> None:
    """Revision reports should be saved as Markdown artifacts."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)
    revision_result = RevisionEngine().revise(
        workspace,
        merge_result,
        get_mock_new_evidence(),
    )

    report_path = save_revision_report(revision_result, "AAPL", tmp_path)
    markdown = Path(report_path).read_text(encoding="utf-8")

    assert report_path.endswith("AAPL_revision_report.md")
    assert "# 决策修正报告：AAPL" in markdown
    assert "## 新证据摘要" in markdown
    assert "## 关键变化" in markdown
    assert "修正后建议" in markdown
