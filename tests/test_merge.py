"""Tests for conflict-aware merge output."""

from agents import MergeAgent
from models.schemas import MergeResult

from tests.conftest import build_workspace_through_counter_evidence


def test_merge_agent_returns_valid_merge_result() -> None:
    """MergeAgent should produce deterministic recommendation fields."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)
    conflict_types = {conflict.conflict_type for conflict in merge_result.key_conflicts}

    assert isinstance(merge_result, MergeResult)
    assert merge_result.final_recommendation in {"Buy", "Hold", "Sell", "Avoid"}
    assert merge_result.confidence in {"low", "medium", "high"}
    assert conflict_types <= {
        "direct_conflict",
        "risk_constraint",
        "temporal_warning",
        "evidence_gap",
    }
