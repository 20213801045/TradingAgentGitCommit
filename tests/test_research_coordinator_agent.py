"""Tests for the research coordinator agent."""

from pathlib import Path

from agents import FundamentalAgent, ResearchCoordinatorAgent
from data import get_mock_company_data
from llm import LLMError, LLMMessage, LLMResponse, MockLLMClient
from main import run_pipeline
from memory.workspace import add_commit, create_branch, create_workspace


def test_research_coordinator_agent_generates_deterministic_plan() -> None:
    """Coordinator should produce auditable commits without an LLM."""

    company_data = get_mock_company_data("AAPL")
    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
    )
    create_branch(workspace, "research-coordination", "Coordination")
    create_branch(workspace, "fundamental-analysis", "Fundamental")

    commits = ResearchCoordinatorAgent().analyze(
        {"phase": "initial_plan", "company_data": company_data},
        workspace,
    )

    assert len(commits) == 1
    assert commits[0].branch_name == "research-coordination"
    assert commits[0].agent_role == "research-coordinator-agent"
    assert "FundamentalAgent" in commits[0].evidence.content


def test_research_coordinator_agent_uses_mock_llm() -> None:
    """Coordinator should accept structured LLM coordination output."""

    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
    )
    create_branch(workspace, "research-coordination", "Coordination")
    create_branch(workspace, "fundamental-analysis", "Fundamental")
    for commit in FundamentalAgent().analyze(get_mock_company_data("AAPL"), workspace):
        add_commit(workspace, commit.branch_name, commit)

    commits = ResearchCoordinatorAgent(llm_client=MockLLMClient()).analyze(
        {"phase": "pre_merge_review"},
        workspace,
    )

    assert len(commits) == 1
    assert "branch coverage" in commits[0].evidence.content.lower()
    assert commits[0].confidence == "medium"


def test_research_coordinator_agent_falls_back_on_llm_error() -> None:
    """LLM failure should fall back to deterministic coordination output."""

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

    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
    )
    create_branch(workspace, "research-coordination", "Coordination")

    commits = ResearchCoordinatorAgent(llm_client=FailingClient()).analyze(
        {"phase": "initial_plan"},
        workspace,
    )

    assert len(commits) == 1
    assert "initial multi-branch research plan" in commits[0].claim


def test_run_pipeline_adds_research_coordination_branch(
    tmp_path: Path,
    capsys,
) -> None:
    """Main pipeline should include coordinator planning and review commits."""

    workspace, _, _, _ = run_pipeline(
        ticker="AAPL",
        data_provider="mock",
        llm_provider="none",
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        preview_lines=0,
    )
    capsys.readouterr()

    coordination_branch = workspace.branches["research-coordination"]
    assert len(coordination_branch.commits) == 2
    assert coordination_branch.commits[0].risk_tag == "coordination_plan"
    assert coordination_branch.commits[1].risk_tag == "coordination_review"
