"""Tests for LLM-backed risk review."""

import json
from pathlib import Path

from agents import (
    BullAgent,
    FundamentalAgent,
    PortfolioAgent,
    RiskAgent,
    TechnicalAgent,
    ValuationAgent,
)
from data import get_mock_company_data
from llm import BaseLLMClient, LLMError, LLMMessage, LLMResponse, MockLLMClient
from main import run_pipeline
from memory.workspace import add_commit, create_branch, create_workspace


BRANCHES = (
    "fundamental-analysis",
    "valuation-analysis",
    "technical-analysis",
    "portfolio-review",
    "bull-case",
    "risk-review",
)


def _workspace_before_risk():
    company_data = get_mock_company_data("AAPL")
    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
    )
    for branch_name in BRANCHES:
        create_branch(workspace, branch_name, branch_name)

    for agent, input_data in [
        (FundamentalAgent(), company_data),
        (ValuationAgent(), company_data),
        (TechnicalAgent(), company_data),
        (PortfolioAgent(), company_data),
        (BullAgent(), {}),
    ]:
        for commit in agent.analyze(input_data, workspace):
            add_commit(workspace, commit.branch_name, commit)
    return workspace


def test_risk_agent_uses_llm_when_available() -> None:
    """RiskAgent should turn LLM risk reviews into evidence-linked commits."""

    workspace = _workspace_before_risk()
    commits = RiskAgent(llm_client=MockLLMClient()).analyze({}, workspace)

    assert len(commits) == 5
    assert all(commit.branch_name == "risk-review" for commit in commits)
    assert any("Valuation risk should cap" in commit.claim for commit in commits)
    assert {commit.risk_tag for commit in commits} >= {
        "valuation_risk",
        "volatility_risk",
        "evidence_gap",
        "temporal_uncertainty",
        "portfolio_risk",
    }
    assert all(commit.evidence.content for commit in commits)


def test_risk_agent_falls_back_on_llm_error() -> None:
    """LLM failures should preserve deterministic risk review behavior."""

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

    workspace = _workspace_before_risk()
    commits = RiskAgent(llm_client=FailingClient()).analyze({}, workspace)

    assert len(commits) == 4
    assert any(commit.claim == "估值风险应降低对上行情景的信心。" for commit in commits)


def test_risk_agent_falls_back_when_llm_omits_required_dimension() -> None:
    """Incomplete LLM risk output should not create partial risk review work."""

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
                                "dimension": "valuation_risk_review",
                                "claim": "Valuation risk should cap upside conviction.",
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

    workspace = _workspace_before_risk()
    commits = RiskAgent(llm_client=IncompleteClient()).analyze({}, workspace)

    assert len(commits) == 4
    assert any(commit.risk_tag == "temporal_uncertainty" for commit in commits)


def test_run_pipeline_uses_llm_backed_risk_agent(
    tmp_path: Path,
    capsys,
) -> None:
    """Main pipeline should pass its LLM client to RiskAgent."""

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
        for commit in workspace.branches["risk-review"].commits
    ]
    assert any("Valuation risk should cap" in claim for claim in claims)
