"""Tests for investment report generation."""

import json

from agents import MergeAgent, ReportAgent

from tests.conftest import build_workspace_through_counter_evidence


def test_report_agent_saves_markdown_and_json_reports(tmp_path) -> None:
    """ReportAgent should save human-readable and machine-readable reports."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)
    report_agent = ReportAgent()
    report = report_agent.generate_report(workspace, merge_result)
    markdown_path, json_path = report_agent.save_report(report, tmp_path)

    markdown = tmp_path.joinpath("AAPL_report.md").read_text(encoding="utf-8")
    json_report = json.loads(tmp_path.joinpath("AAPL_investment_report.json").read_text())

    assert markdown_path.endswith("AAPL_report.md")
    assert json_path.endswith("AAPL_investment_report.json")
    assert "## 7. 反证检查" in markdown
    assert "当前买点评分" in markdown
    assert "风险收益评分" in markdown
    assert "## 8. 风险审查" in markdown
    assert "## 9. 决策审计轨迹" in markdown
    assert "## 10. 触发重新评估的条件" in markdown
    assert "## 11. 完整证据表" in markdown
    assert json_report["ticker"] == "AAPL"
    assert json_report["merge_result"]["final_recommendation"] == merge_result.final_recommendation
    assert "decision_scores" in json_report["merge_result"]
    assert "markdown_report" in json_report
