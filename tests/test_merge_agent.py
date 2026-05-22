"""Tests for conflict-aware merge behavior."""

from agents import MergeAgent

from tests.conftest import build_workspace_through_counter_evidence


def test_merge_agent_includes_counter_evidence_in_opposition_and_conflicts() -> None:
    """Counter-evidence should affect opposition and evidence-gap conflicts."""

    workspace = build_workspace_through_counter_evidence()
    merge_agent = MergeAgent()
    result = merge_agent.merge(workspace)
    conflict_types = {conflict.conflict_type for conflict in result.key_conflicts}

    assert result.final_recommendation in {"Buy", "Hold", "Sell", "Avoid"}
    assert merge_agent.support_score >= 0.0
    assert merge_agent.risk_score >= 0.0
    assert merge_agent.opposition_score >= 0.0
    assert 0.0 <= result.decision_scores.entry_score <= 100.0
    assert 0.0 <= result.decision_scores.risk_reward_score <= 100.0
    assert 0.0 <= result.decision_scores.conviction_score <= 100.0
    assert result.decision_scores.risk_level in {"low", "medium", "high"}
    assert result.decision_scores.position_sizing_suggestion
    assert len(result.key_conflicts) <= 8
    assert "evidence_gap" in conflict_types
    assert any(
        "需要验证" in claim
        for claim in result.main_opposing_claims
    )
    assert conflict_types <= {
        "direct_conflict",
        "risk_constraint",
        "temporal_warning",
        "evidence_gap",
    }
