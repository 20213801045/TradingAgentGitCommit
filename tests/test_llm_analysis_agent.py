"""Tests for LLM-powered synthesis analysis."""

from pathlib import Path

from agents import LLMAnalysisAgent
from llm import LLMError, LLMMessage, LLMResponse, MockLLMClient
from main import run_pipeline
from memory.workspace import create_branch

from tests.conftest import build_workspace_through_counter_evidence


def test_llm_analysis_agent_generates_evidence_linked_commits() -> None:
    """LLMAnalysisAgent should turn LLM insights into auditable commits."""

    workspace = build_workspace_through_counter_evidence()
    create_branch(workspace, "llm-analysis", "LLM analysis")

    commits = LLMAnalysisAgent(llm_client=MockLLMClient()).analyze(workspace)

    assert commits
    assert all(commit.branch_name == "llm-analysis" for commit in commits)
    assert all(commit.agent_role == "llm-analysis-agent" for commit in commits)
    assert all(commit.evidence.content for commit in commits)
    assert any("LLM 综合判断" in commit.claim for commit in commits)


def test_llm_analysis_prompt_uses_stable_evidence_refs() -> None:
    """LLMAnalysisAgent prompts should avoid random commit ids for cache hits."""

    class CapturingClient(MockLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.user_prompt = ""

        def complete(
            self,
            messages: list[LLMMessage],
            temperature: float = 0.0,
            response_format: dict[str, object] | None = None,
        ) -> LLMResponse:
            self.user_prompt = next(
                message.content for message in messages if message.role == "user"
            )
            return super().complete(messages, temperature, response_format)

    workspace = build_workspace_through_counter_evidence()
    client = CapturingClient()

    LLMAnalysisAgent(llm_client=client).analyze(workspace)

    assert "evidence_ref:" in client.user_prompt
    assert "commit_id:" not in client.user_prompt


def test_llm_analysis_agent_falls_back_on_llm_error() -> None:
    """LLM errors should not break the deterministic pipeline."""

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

    workspace = build_workspace_through_counter_evidence()

    assert LLMAnalysisAgent(llm_client=FailingClient()).analyze(workspace) == []


def test_run_pipeline_uses_mock_llm_for_analysis_branch(
    tmp_path: Path,
    capsys,
) -> None:
    """The main pipeline should include LLM synthesis when an LLM client exists."""

    workspace, _, markdown_path, _ = run_pipeline(
        ticker="AAPL",
        data_provider="mock",
        llm_provider=MockLLMClient(),
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        preview_lines=0,
    )
    capsys.readouterr()

    assert workspace.branches["llm-analysis"].commits
    markdown = Path(markdown_path).read_text(encoding="utf-8")
    assert "大模型洞察" in markdown
