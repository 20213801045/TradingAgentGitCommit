"""Tests for EVIR evaluation metrics and evaluator."""

from pathlib import Path

from agents import MergeAgent, ReportAgent
from evaluation import (
    Evaluator,
    audit_completeness_score,
    conflict_coverage_score,
    decision_traceability_score,
    evidence_coverage_score,
    save_evaluation_result,
    temporal_validity_score,
)
from models.schemas import EvaluationResult

from tests.conftest import build_workspace_through_counter_evidence


def test_evidence_coverage_score_is_full_for_complete_commits() -> None:
    """Complete claim-evidence commits should have full evidence coverage."""

    workspace = build_workspace_through_counter_evidence()

    assert evidence_coverage_score(workspace) == 1.0


def test_temporal_validity_score_handles_status_mix() -> None:
    """Temporal score should average valid/stale/expired/unknown credit."""

    workspace = build_workspace_through_counter_evidence()
    commits = [
        commit
        for branch in workspace.branches.values()
        for commit in branch.commits
    ][:4]
    statuses = ["valid", "stale", "expired", "unknown"]
    for commit, status in zip(commits, statuses):
        commit.temporal_status = status

    score = temporal_validity_score(workspace)

    assert 0.0 <= score <= 1.0
    assert score < 1.0


def test_conflict_coverage_score_is_bounded() -> None:
    """Conflict coverage should return a normalized score."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)
    score = conflict_coverage_score(merge_result, workspace)

    assert 0.0 <= score <= 1.0


def test_decision_traceability_score_detects_matched_claims() -> None:
    """Merge claims should trace back to existing commit claims."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)

    assert decision_traceability_score(merge_result, workspace) == 1.0


def test_audit_completeness_score_detects_required_sections() -> None:
    """Full reports should include all required audit sections."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)
    report = ReportAgent().generate_report(workspace, merge_result)

    assert audit_completeness_score(report) == 1.0


def test_evaluator_returns_and_saves_evaluation_result(tmp_path: Path) -> None:
    """Evaluator should return a bounded EvaluationResult and save JSON."""

    workspace = build_workspace_through_counter_evidence()
    merge_result = MergeAgent().merge(workspace)
    report = ReportAgent().generate_report(workspace, merge_result)

    evaluation_result = Evaluator().evaluate(workspace, merge_result, report)
    evaluation_path = save_evaluation_result(evaluation_result, "AAPL", tmp_path)

    assert isinstance(evaluation_result, EvaluationResult)
    assert 0.0 <= evaluation_result.overall_score <= 1.0
    assert Path(evaluation_path).exists()
