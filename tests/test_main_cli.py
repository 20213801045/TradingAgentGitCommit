"""Tests for CLI-facing pipeline behavior."""

from pathlib import Path
from typing import Any

from data import BaseDataProvider, MockDataProvider
from main import run_pipeline


def test_run_pipeline_accepts_output_and_report_dirs(tmp_path, capsys) -> None:
    """The pipeline should write artifacts to caller-provided directories."""

    output_dir = tmp_path / "outputs"
    report_dir = tmp_path / "reports"

    workspace, merge_result, markdown_path, json_path = run_pipeline(
        ticker="AAPL",
        data_provider="mock",
        llm_provider="none",
        output_dir=output_dir,
        report_dir=report_dir,
        preview_lines=0,
    )
    capsys.readouterr()

    assert workspace.ticker == "AAPL"
    assert merge_result.final_recommendation in {"Buy", "Hold", "Sell", "Avoid"}
    assert Path(markdown_path).exists()
    assert Path(json_path).exists()
    assert (output_dir / "workspaces" / "AAPL_workspace.json").exists()
    assert (report_dir / "AAPL_revision_report.md").exists()


def test_run_pipeline_falls_back_when_deepseek_key_missing(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """DeepSeek mode without a key should warn and continue without LLM."""

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace, _, markdown_path, json_path = run_pipeline(
        ticker="AAPL",
        data_provider="mock",
        llm_provider="deepseek",
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        preview_lines=0,
    )
    captured = capsys.readouterr()

    assert workspace.ticker == "AAPL"
    assert "Falling back to deterministic no-LLM mode" in captured.out
    assert Path(markdown_path).exists()
    assert Path(json_path).exists()


def test_run_pipeline_research_question_uses_requested_ticker(tmp_path, capsys) -> None:
    """The generated report should not hard-code AAPL in the research question."""

    class MUProvider(BaseDataProvider):
        name = "mu-test"

        def fetch_company_data(self, ticker: str) -> dict[str, Any]:
            data = MockDataProvider().fetch_company_data("AAPL")
            data["ticker"] = ticker
            data["company_name"] = "Micron Technology, Inc."
            return data

    workspace, _, markdown_path, _ = run_pipeline(
        ticker="MU",
        data_provider=MUProvider(),
        llm_provider="none",
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        preview_lines=0,
    )
    capsys.readouterr()

    markdown = Path(markdown_path).read_text(encoding="utf-8")
    revision_markdown = (tmp_path / "reports" / "MU_revision_report.md").read_text(
        encoding="utf-8",
    )

    assert workspace.research_question.startswith("MU 是否适合")
    assert "研究问题：MU 是否适合纳入长期投资观察名单？" in markdown
    assert "研究问题：AAPL 是否适合纳入长期投资观察名单？" not in markdown
    assert "# 决策修正报告：MU" in revision_markdown
    assert "Memory demand remains cyclical" in revision_markdown


def test_run_pipeline_can_use_real_revision_evidence_provider(
    tmp_path,
    capsys,
) -> None:
    """Real data pipelines should allow ticker-specific revision evidence."""

    class MUProvider(BaseDataProvider):
        name = "real"

        def fetch_company_data(self, ticker: str) -> dict[str, Any]:
            data = MockDataProvider().fetch_company_data("AAPL")
            data["ticker"] = ticker
            data["company_name"] = "Micron Technology, Inc."
            data["data_source"] = "yfinance"
            data["as_of_date"] = "2026-05-18"
            return data

    def fake_real_new_evidence(ticker: str) -> dict[str, Any]:
        assert ticker == "MU"
        return {
            "ticker": "MU",
            "new_financial_metrics": {
                "revenue_growth_yoy": "45%",
                "net_margin": "30%",
                "forward_pe": "9.5",
            },
            "new_technical_indicators": {
                "rsi": "59",
                "price_trend": "above 20-day and 50-day moving averages",
                "volatility": "moderate",
            },
            "new_events": [
                {
                    "title": "Latest real revision evidence for MU",
                    "source": "Yahoo Finance via yfinance",
                    "source_type": "financial_data_provider",
                    "timestamp": "2026-05-18",
                }
            ],
        }

    run_pipeline(
        ticker="MU",
        data_provider=MUProvider(),
        llm_provider="none",
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        preview_lines=0,
        new_evidence_provider=fake_real_new_evidence,
    )
    capsys.readouterr()

    revision_markdown = (tmp_path / "reports" / "MU_revision_report.md").read_text(
        encoding="utf-8",
    )

    assert "Latest real revision evidence for MU" in revision_markdown
    assert "Yahoo Finance via yfinance" in revision_markdown
