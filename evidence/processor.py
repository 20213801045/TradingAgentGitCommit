"""Evidence post-processing utilities for commits and workspaces."""

from evidence.evidence_scorer import EvidenceScorer
from evidence.temporal_checker import TemporalChecker
from models.schemas import ClaimEvidenceCommit, Workspace


def process_commit_evidence(commit: ClaimEvidenceCommit) -> ClaimEvidenceCommit:
    """Compute evidence quality and temporal status for a commit."""

    if (
        commit.branch_name == "counter-evidence"
        and commit.evidence_quality_score is not None
        and commit.temporal_status is not None
    ):
        return commit

    scorer = EvidenceScorer()
    temporal_checker = TemporalChecker()

    commit.evidence_quality_score = scorer.score(commit.evidence, commit.claim)
    commit.temporal_status = temporal_checker.check(
        commit.evidence,
        commit.time_horizon,
    )
    return commit


def process_workspace_evidence(workspace: Workspace) -> Workspace:
    """Process every commit in every workspace branch."""

    for branch in workspace.branches.values():
        branch.commits = [
            process_commit_evidence(commit)
            for commit in branch.commits
        ]
    return workspace
