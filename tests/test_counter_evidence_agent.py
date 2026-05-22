"""Tests for the counter-evidence reasoning layer."""

from agents import CounterEvidenceAgent
from llm import LLMError, LLMMessage, LLMResponse, MockLLMClient

from tests.conftest import build_workspace_through_counter_evidence


def test_counter_evidence_agent_generates_structured_counter_commits() -> None:
    """CounterEvidenceAgent should emit deduplicated structured commits."""

    workspace = build_workspace_through_counter_evidence()
    counter_commits = workspace.branches["counter-evidence"].commits
    claims = [commit.claim for commit in counter_commits]

    assert len(counter_commits) >= 4
    assert len(claims) == len(set(claims))
    assert all(commit.branch_name == CounterEvidenceAgent.branch_name for commit in counter_commits)
    assert all(commit.agent_role == CounterEvidenceAgent.role for commit in counter_commits)
    assert all(commit.counter_evidence for commit in counter_commits)
    assert any("收入增长需要验证" in claim for claim in claims)
    assert any("技术动量需要更新指标" in claim for claim in claims)


def test_counter_evidence_commits_reuse_original_evidence_scores() -> None:
    """Counter-evidence commits should preserve copied evidence scores and status."""

    workspace = build_workspace_through_counter_evidence()
    revenue_counter = next(
        commit
        for commit in workspace.branches["counter-evidence"].commits
        if commit.risk_tag == "counter_evidence_growth"
    )
    source_revenue = next(
        commit
        for commit in workspace.branches["fundamental-analysis"].commits
        if commit.evidence.metric_name == "revenue_growth_yoy"
    )

    assert revenue_counter.evidence.evidence_id == source_revenue.evidence.evidence_id
    assert revenue_counter.evidence_quality_score == source_revenue.evidence_quality_score
    assert revenue_counter.temporal_status == source_revenue.temporal_status


def test_counter_evidence_agent_can_use_mock_llm_for_extra_question() -> None:
    """Optional LLM support should add deterministic questions without network calls."""

    workspace = build_workspace_through_counter_evidence()
    workspace.branches["counter-evidence"].commits = []
    agent = CounterEvidenceAgent(llm_client=MockLLMClient())

    counter_commits = agent.analyze(workspace)

    assert counter_commits
    assert any(
        len(commit.counter_evidence or []) > 3
        for commit in counter_commits
    )


def test_counter_evidence_agent_falls_back_on_malformed_llm_json() -> None:
    """Malformed LLM output should not break deterministic counter-evidence."""

    class MalformedJSONClient(MockLLMClient):
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
                content="not json",
                model="bad-json",
                provider="mock",
            )

    workspace = build_workspace_through_counter_evidence()
    workspace.branches["counter-evidence"].commits = []
    counter_commits = CounterEvidenceAgent(
        llm_client=MalformedJSONClient(),
    ).analyze(workspace)

    assert counter_commits
    assert all(
        len(commit.counter_evidence or []) == 3
        for commit in counter_commits
    )


def test_counter_evidence_agent_falls_back_on_llm_error() -> None:
    """LLM errors should leave deterministic questions intact."""

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
    workspace.branches["counter-evidence"].commits = []
    counter_commits = CounterEvidenceAgent(llm_client=FailingClient()).analyze(workspace)

    assert counter_commits
    assert all(commit.counter_evidence for commit in counter_commits)
