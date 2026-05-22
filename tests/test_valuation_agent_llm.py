"""Tests for LLM-backed valuation analysis."""

import json
from pathlib import Path

from agents import ValuationAgent
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
    create_branch(workspace, "valuation-analysis", "Valuation")
    return workspace


def test_valuation_agent_uses_llm_when_available() -> None:
    """ValuationAgent should turn LLM insights into evidence-linked commits."""

    workspace = _workspace()
    commits = ValuationAgent(llm_client=MockLLMClient()).analyze(
        get_mock_company_data("AAPL"),
        workspace,
    )

    assert len(commits) == 2
    assert all(commit.branch_name == "valuation-analysis" for commit in commits)
    assert any("valuation premium" in commit.claim for commit in commits)
    assert {commit.evidence.metric_name for commit in commits} == {
        "relative_forward_pe",
        "growth_adjusted_valuation",
    }
    assert any(commit.risk_tag == "valuation_risk" for commit in commits)


def test_valuation_agent_falls_back_on_llm_error() -> None:
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
    commits = ValuationAgent(llm_client=FailingClient()).analyze(
        get_mock_company_data("AAPL"),
        workspace,
    )

    assert len(commits) == 2
    assert any(commit.claim == "股票相对行业估值存在溢价。" for commit in commits)


def test_valuation_agent_falls_back_when_llm_omits_required_dimension() -> None:
    """Incomplete structured LLM output should not create partial valuation work."""

    class IncompleteClient(BaseLLMClient):
        provider = "incomplete"
        model = "incomplete"

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
                                "dimension": "relative_valuation",
                                "claim": "The stock trades at a premium.",
                                "confidence": "medium",
                                "risk_tag": "valuation_risk",
                                "time_horizon": "6-12 months",
                            }
                        ]
                    }
                ),
                model=self.model,
                provider=self.provider,
            )

    workspace = _workspace()
    commits = ValuationAgent(llm_client=IncompleteClient()).analyze(
        get_mock_company_data("AAPL"),
        workspace,
    )

    assert len(commits) == 2
    assert any(commit.claim == "增长调整后的估值较为中性。" for commit in commits)


def test_run_pipeline_uses_llm_backed_valuation_agent(
    tmp_path: Path,
    capsys,
) -> None:
    """Main pipeline should pass its LLM client to ValuationAgent."""

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
        for commit in workspace.branches["valuation-analysis"].commits
    ]
    assert any("valuation premium" in claim for claim in claims)
