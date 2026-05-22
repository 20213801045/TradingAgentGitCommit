"""Tests for report generation."""

from pathlib import Path

from agents import MergeAgent, ReportAgent
from models.schemas import InvestmentReport

from tests.conftest import build_workspace_through_counter_evidence


def test_report_agent_generates_and_saves_report(tmp_path: Path) -> None:
    """ReportAgent should generate report content and save artifacts."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)
    report_agent = ReportAgent()
    report = report_agent.generate_report(workspace, merge_result)
    markdown_path, json_path = report_agent.save_report(report, tmp_path)

    assert isinstance(report, InvestmentReport)
    assert Path(markdown_path).exists()
    assert Path(json_path).exists()
    assert "最终结论" in report.markdown_report
    assert "当前买点评分" in report.markdown_report
    assert "风险收益评分" in report.markdown_report
    assert "多维评分卡" in report.markdown_report
    assert "支持性证据" in report.markdown_report
    assert "反对与谨慎证据" in report.markdown_report
    assert "关键冲突" in report.markdown_report
    assert "反证检查" in report.markdown_report
    assert "风险审查" in report.markdown_report
    assert "决策审计轨迹" in report.markdown_report
    assert "完整证据表" in report.markdown_report
