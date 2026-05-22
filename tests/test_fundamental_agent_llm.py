"""Tests for LLM-backed fundamental analysis."""

import json
from pathlib import Path

from agents import FundamentalAgent
from data import get_mock_company_data
from llm import BaseLLMClient, LLMError, LLMMessage, LLMResponse, MockLLMClient
from main import run_pipeline
from memory.workspace import create_branch, create_workspace


def _workspace():
    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
    )
    create_branch(workspace, "fundamental-analysis", "Fundamental")
    return workspace


def test_fundamental_agent_uses_llm_when_available() -> None:
    """FundamentalAgent should turn LLM insights into evidence-linked commits."""

    workspace = _workspace()
    commits = FundamentalAgent(llm_client=MockLLMClient()).analyze(
        get_mock_company_data("AAPL"),
        workspace,
    )

    assert len(commits) == 6
    assert all(commit.branch_name == "fundamental-analysis" for commit in commits)
    assert any("Revenue growth provides" in commit.claim for commit in commits)
    assert {commit.evidence.metric_name for commit in commits} == {
        "revenue_growth_yoy",
        "margin_quality",
        "cash_generation",
        "balance_sheet_health",
        "capital_allocation_capacity",
        "fundamental_valuation_context",
    }


def test_fundamental_agent_falls_back_on_llm_error() -> None:
    """LLM failures should preserve the original deterministic behavior."""

    class FailingClient(MockLLMClient):
        def complete(
            self,
            messages: list[LLMMessage],
            temperature: float = 0.0,
            response_format: dict[str, object] | None = None,
        ) -> LLMResponse:
            del messages
            del temperature
            del response_format
            raise LLMError("simulated failure")

    workspace = _workspace()
    commits = FundamentalAgent(llm_client=FailingClient()).analyze(
        get_mock_company_data("AAPL"),
        workspace,
    )

    assert len(commits) == 6
    assert any(commit.claim == "公司收入呈现正增长。" for commit in commits)


def test_fundamental_agent_validator_rejects_metric_contradictions() -> None:
    """LLM claims that contradict obvious metrics should trigger fallback."""

    class ContradictingClient(BaseLLMClient):
        provider = "contradicting"
        model = "contradicting"

        def complete(
            self,
            messages: list[LLMMessage],
            temperature: float = 0.0,
            response_format: dict[str, object] | None = None,
        ) -> LLMResponse:
            del messages
            del temperature
            del response_format
            return LLMResponse(
                content=json.dumps(
                    {
                        "insights": [
                            {
                                "dimension": "growth_quality",
                                "claim": "Revenue growth is strong and positive.",
                                "confidence": "medium",
                                "risk_tag": "growth_quality",
                                "time_horizon": "12 months",
                            },
                            {
                                "dimension": "margin_quality",
                                "claim": "Profitability appears stable.",
                                "confidence": "medium",
                                "risk_tag": "profitability",
                                "time_horizon": "12-24 months",
                            },
                            {
                                "dimension": "cash_generation",
                                "claim": "Cash generation is durable.",
                                "confidence": "medium",
                                "risk_tag": "cash_generation_support",
                                "time_horizon": "12-24 months",
                            },
                            {
                                "dimension": "balance_sheet",
                                "claim": "Balance sheet resilience is supported.",
                                "confidence": "medium",
                                "risk_tag": "balance_sheet_strength",
                                "time_horizon": "12-24 months",
                            },
                            {
                                "dimension": "capital_allocation",
                                "claim": "Capital allocation capacity is supported.",
                                "confidence": "medium",
                                "risk_tag": "capital_allocation_support",
                                "time_horizon": "12-24 months",
                            },
                            {
                                "dimension": "fundamental_valuation_risk",
                                "claim": "Valuation risk is manageable.",
                                "confidence": "medium",
                                "risk_tag": "valuation_risk",
                                "time_horizon": "6-12 months",
                            },
                        ]
                    }
                ),
                model=self.model,
                provider=self.provider,
            )

    company_data = get_mock_company_data("AAPL")
    company_data["financial_metrics"]["revenue_growth_yoy"] = "-3%"
    workspace = _workspace()
    commits = FundamentalAgent(llm_client=ContradictingClient()).analyze(
        company_data,
        workspace,
    )

    assert len(commits) == 6
    assert any(commit.claim == "收入增长偏弱或为负。" for commit in commits)


def test_run_pipeline_uses_llm_backed_fundamental_agent(
    tmp_path: Path,
    capsys,
) -> None:
    """Main pipeline should pass its LLM client to FundamentalAgent."""

    workspace, _, _, _ = run_pipeline(
        ticker="AAPL",
        data_provider="mock",
        llm_provider=MockLLMClient(),
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        preview_lines=0,
    )
    capsys.readouterr()

    claims = [
        commit.claim
        for commit in workspace.branches["fundamental-analysis"].commits
    ]
    assert any("Revenue growth provides" in claim for claim in claims)
