"""Core Pydantic schemas for evidence-versioned investment research."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A source-backed artifact used to support or challenge a claim."""

    evidence_id: str
    content: str
    source: str
    source_type: str
    timestamp: str
    url: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[str] = None


class ClaimEvidenceCommit(BaseModel):
    """An auditable research claim paired with supporting evidence."""

    commit_id: str
    agent_role: str
    branch_name: str
    claim: str
    evidence: Evidence
    evidence_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: str
    risk_tag: str
    time_horizon: str
    temporal_status: Optional[str] = None
    counter_evidence: Optional[List[str]] = None
    created_at: str


class Branch(BaseModel):
    """A research perspective containing ordered claim-evidence commits."""

    branch_name: str
    description: str
    commits: List[ClaimEvidenceCommit] = Field(default_factory=list)


class Workspace(BaseModel):
    """A ticker-specific research workspace with multiple evidence branches."""

    ticker: str
    company_name: Optional[str] = None
    research_question: str
    created_at: str
    branches: Dict[str, Branch] = Field(default_factory=dict)


class Conflict(BaseModel):
    """A conflict found between claims during merge review."""

    conflict_id: str
    conflict_type: str
    claim_a: str
    claim_b: str
    explanation: str
    severity: str


class DecisionScores(BaseModel):
    """Quantitative investment-decision metrics derived during merge."""

    support_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    opposition_score: float = Field(default=0.0, ge=0.0, le=100.0)
    entry_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_reward_score: float = Field(default=0.0, ge=0.0, le=100.0)
    conviction_score: float = Field(default=0.0, ge=0.0, le=100.0)
    valuation_attractiveness: float = Field(default=0.0, ge=0.0, le=100.0)
    technical_timing_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_level: Literal["low", "medium", "high"] = "medium"
    position_sizing_suggestion: str = "Observation only"


class MergeResult(BaseModel):
    """A conflict-aware synthesis of all research branches."""

    final_recommendation: str
    confidence: str
    decision_scores: DecisionScores = Field(default_factory=DecisionScores)
    main_supporting_claims: List[str]
    main_opposing_claims: List[str]
    key_conflicts: List[Conflict]
    risk_adjustment: str
    decision_rationale: str
    conditions_for_revision: List[str]


class InvestmentReport(BaseModel):
    """Final auditable investment research report."""

    ticker: str
    company_name: Optional[str] = None
    final_recommendation: str
    merge_result: MergeResult
    audit_trail: List[str]
    evidence_table: List[ClaimEvidenceCommit]
    markdown_report: str


class RevisionRecord(BaseModel):
    """How a single prior claim was affected by new evidence."""

    revision_id: str
    original_claim: str
    original_branch: str
    original_commit_id: str
    new_evidence_summary: str
    revision_status: Literal[
        "supported",
        "weakened",
        "contradicted",
        "unchanged",
        "expired",
    ]
    explanation: str
    impact_on_decision: Literal[
        "increase_confidence",
        "decrease_confidence",
        "change_recommendation",
        "no_change",
    ]


class RevisionResult(BaseModel):
    """Revision summary after comparing old claims with new evidence."""

    previous_recommendation: str
    revised_recommendation: str
    revision_records: List[RevisionRecord]
    key_changes: List[str]
    revision_rationale: str
    updated_conditions_for_revision: List[str]


class EvaluationResult(BaseModel):
    """Quantitative evaluation of evidence grounding and auditability."""

    evidence_coverage_score: float = Field(ge=0.0, le=1.0)
    temporal_validity_score: float = Field(ge=0.0, le=1.0)
    conflict_coverage_score: float = Field(ge=0.0, le=1.0)
    decision_traceability_score: float = Field(ge=0.0, le=1.0)
    audit_completeness_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any]
