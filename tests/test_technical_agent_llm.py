"""Tests for LLM-backed technical analysis."""

import json
from pathlib import Path

from agents import TechnicalAgent
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
    create_branch(workspace, "technical-analysis", "Technical")
    return workspace


def test_technical_agent_uses_llm_when_available() -> None:
    """TechnicalAgent should turn LLM insights into evidence-linked commits."""

    workspace = _workspace()
    commits = TechnicalAgent(llm_client=MockLLMClient()).analyze(
        get_mock_company_data("AAPL"),
        workspace,
    )

    assert len(commits) == 4
    assert all(commit.branch_name == "technical-analysis" for commit in commits)
    assert any("price trend remains constructive" in commit.claim for commit in commits)
    assert {commit.evidence.metric_name for commit in commits} == {
        "price_trend",
        "rsi",
        "volatility",
        "support_resistance",
    }


def test_technical_agent_falls_back_on_llm_error() -> None:
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
    commits = TechnicalAgent(llm_client=FailingClient()).analyze(
        get_mock_company_data("AAPL"),
        workspace,
    )

    assert len(commits) == 4
    assert any(commit.claim == "股价处于建设性的价格趋势中。" for commit in commits)


def test_technical_agent_validator_rejects_indicator_contradictions() -> None:
    """LLM claims that contradict obvious indicators should trigger fallback."""

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
                                "dimension": "trend_structure",
                                "claim": "The trend is constructive and bullish.",
                                "confidence": "medium",
                                "risk_tag": "price_trend_support",
                                "time_horizon": "1-3 months",
                            },
                            {
                                "dimension": "momentum",
                                "claim": "Momentum is positive.",
                                "confidence": "medium",
                                "risk_tag": "momentum",
                                "time_horizon": "1-3 months",
                            },
                            {
                                "dimension": "volatility_risk",
                                "claim": "Volatility is a risk.",
                                "confidence": "medium",
                                "risk_tag": "volatility_risk",
                                "time_horizon": "1-3 months",
                            },
                            {
                                "dimension": "support_resistance",
                                "claim": "Levels frame risk-reward.",
                                "confidence": "medium",
                                "risk_tag": "support_resistance",
                                "time_horizon": "1-3 months",
                            },
                        ]
                    }
                ),
                model=self.model,
                provider=self.provider,
            )

    company_data = get_mock_company_data("AAPL")
    company_data["technical_indicators"]["price_trend"] = (
        "below 20-day and 50-day moving averages"
    )
    workspace = _workspace()
    commits = TechnicalAgent(llm_client=ContradictingClient()).analyze(
        company_data,
        workspace,
    )

    assert len(commits) == 4
    assert any(commit.claim == "股价低于关键移动均线。" for commit in commits)


def test_run_pipeline_uses_llm_backed_technical_agent(
    tmp_path: Path,
    capsys,
) -> None:
    """Main pipeline should pass its LLM client to TechnicalAgent."""

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
        for commit in workspace.branches["technical-analysis"].commits
    ]
    assert any("price trend remains constructive" in claim for claim in claims)
