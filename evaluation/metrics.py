"""Deterministic evaluation metrics for EVIR artifacts."""

from models.schemas import InvestmentReport, MergeResult, Workspace


REQUIRED_AUDIT_SECTIONS = (
    "最终结论",
    "多维评分卡",
    "支持性证据",
    "反对与谨慎证据",
    "关键冲突",
    "反证检查",
    "风险审查",
    "决策审计轨迹",
    "触发重新评估的条件",
    "完整证据表",
)

RISK_TAG_MARKERS = (
    "risk",
    "counter_evidence",
    "uncertainty",
    "evidence_gap",
)

TEMPORAL_CREDITS = {
    "valid": 1.0,
    "stale": 0.5,
    "unknown": 0.3,
    "expired": 0.0,
}


def evidence_coverage_score(workspace: Workspace) -> float:
    """Score the share of commits with non-empty evidence content and source."""

    commits = _all_commits(workspace)
    if not commits:
        return 0.0

    supported_commits = sum(
        bool(commit.evidence.content.strip() and commit.evidence.source.strip())
        for commit in commits
    )
    return _round_score(supported_commits / len(commits))


def temporal_validity_score(workspace: Workspace) -> float:
    """Average temporal validity credit across all commits."""

    commits = _all_commits(workspace)
    if not commits:
        return 0.0

    total_credit = sum(
        TEMPORAL_CREDITS.get(commit.temporal_status or "unknown", 0.3)
        for commit in commits
    )
    return _round_score(total_credit / len(commits))


def conflict_coverage_score(
    merge_result: MergeResult,
    workspace: Workspace,
) -> float:
    """Score conflict coverage relative to risk and counter-evidence commits."""

    risk_commits = [
        commit
        for commit in _all_commits(workspace)
        if _is_risk_related(commit.risk_tag)
    ]
    conflicts = merge_result.key_conflicts
    if not risk_commits:
        return 1.0 if not conflicts else _round_score(
            min(1.0, len({conflict.conflict_type for conflict in conflicts}) / 4.0)
        )
    if not conflicts:
        return 0.0

    conflict_type_count = len({conflict.conflict_type for conflict in conflicts})
    type_coverage = min(1.0, conflict_type_count / 4.0)
    conflict_volume = min(1.0, len(conflicts) / max(1, len(risk_commits)))
    return _round_score(type_coverage * conflict_volume)


def decision_traceability_score(
    merge_result: MergeResult,
    workspace: Workspace,
) -> float:
    """Score whether merge claims can be traced to workspace commits."""

    merge_claims = [
        *merge_result.main_supporting_claims,
        *merge_result.main_opposing_claims,
    ]
    if not merge_claims:
        return 0.0

    commit_claims = {commit.claim for commit in _all_commits(workspace)}
    matched_claims = sum(claim in commit_claims for claim in merge_claims)
    return _round_score(matched_claims / len(merge_claims))


def audit_completeness_score(report: InvestmentReport) -> float:
    """Score whether the Markdown report includes required audit sections."""

    markdown = report.markdown_report
    present_sections = sum(section in markdown for section in REQUIRED_AUDIT_SECTIONS)
    return _round_score(present_sections / len(REQUIRED_AUDIT_SECTIONS))


def _all_commits(workspace: Workspace):
    """Collect all commits from all workspace branches."""

    return [
        commit
        for branch in workspace.branches.values()
        for commit in branch.commits
    ]


def _is_risk_related(risk_tag: str) -> bool:
    """Return whether a risk tag should be evaluated for conflict coverage."""

    normalized_tag = risk_tag.lower()
    return any(marker in normalized_tag for marker in RISK_TAG_MARKERS)


def _round_score(score: float) -> float:
    """Round a metric score consistently."""

    return round(max(0.0, min(1.0, score)), 2)
