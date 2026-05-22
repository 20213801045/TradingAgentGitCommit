"""Smoke tests for EVIR Pydantic schemas."""

from models.schemas import (
    Branch,
    ClaimEvidenceCommit,
    Conflict,
    Evidence,
    InvestmentReport,
    MergeResult,
    RevisionRecord,
    RevisionResult,
    Workspace,
)


def test_core_schemas_can_be_created() -> None:
    """All core schemas should validate representative data."""

    evidence = Evidence(
        evidence_id="ev-1",
        content="Revenue growth is 8%.",
        source="Mock Financial Metrics",
        source_type="mock_financial_metric",
        timestamp="2026-05-01",
        url=None,
        metric_name="revenue_growth_yoy",
        metric_value="8%",
    )
    commit = ClaimEvidenceCommit(
        commit_id="commit-1",
        agent_role="fundamental-agent",
        branch_name="fundamental-analysis",
        claim="The company shows positive revenue growth.",
        evidence=evidence,
        evidence_quality_score=0.9,
        confidence="medium",
        risk_tag="growth_quality",
        time_horizon="12 months",
        temporal_status="valid",
        counter_evidence=None,
        created_at="2026-05-01T00:00:00+00:00",
    )
    branch = Branch(
        branch_name="fundamental-analysis",
        description="Fundamental analysis.",
        commits=[commit],
    )
    workspace = Workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Should AAPL be considered?",
        created_at="2026-05-01T00:00:00+00:00",
        branches={branch.branch_name: branch},
    )
    conflict = Conflict(
        conflict_id="conflict-1",
        conflict_type="risk_constraint",
        claim_a="Growth is positive.",
        claim_b="Volatility is high.",
        explanation="Risk limits the positive implication.",
        severity="medium",
    )
    merge_result = MergeResult(
        final_recommendation="Hold",
        confidence="medium",
        main_supporting_claims=[commit.claim],
        main_opposing_claims=["Volatility is high."],
        key_conflicts=[conflict],
        risk_adjustment="Risk moderates the decision.",
        decision_rationale="Support and risk are mixed.",
        conditions_for_revision=["Refresh evidence."],
    )
    report = InvestmentReport(
        ticker=workspace.ticker,
        company_name=workspace.company_name,
        final_recommendation=merge_result.final_recommendation,
        merge_result=merge_result,
        audit_trail=["Workspace created."],
        evidence_table=[commit],
        markdown_report="# Report",
    )
    revision_record = RevisionRecord(
        revision_id="revision-1",
        original_claim=commit.claim,
        original_branch=commit.branch_name,
        original_commit_id=commit.commit_id,
        new_evidence_summary="New evidence summary.",
        revision_status="supported",
        explanation="New evidence supports the old claim.",
        impact_on_decision="increase_confidence",
    )
    revision_result = RevisionResult(
        previous_recommendation="Hold",
        revised_recommendation="Hold",
        revision_records=[revision_record],
        key_changes=["No material change."],
        revision_rationale="Recommendation stayed the same.",
        updated_conditions_for_revision=["Refresh evidence."],
    )

    assert report.merge_result.key_conflicts[0].conflict_type == "risk_constraint"
    assert revision_result.revision_records[0].revision_status == "supported"
