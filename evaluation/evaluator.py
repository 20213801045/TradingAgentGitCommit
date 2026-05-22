"""High-level evaluator for EVIR research artifacts."""

from pathlib import Path

from evaluation.metrics import (
    audit_completeness_score,
    conflict_coverage_score,
    decision_traceability_score,
    evidence_coverage_score,
    temporal_validity_score,
)
from memory.storage import save_json
from models.schemas import EvaluationResult, InvestmentReport, MergeResult, Workspace


class Evaluator:
    """Compute evidence-grounding, conflict, and auditability scores."""

    def evaluate(
        self,
        workspace: Workspace,
        merge_result: MergeResult,
        report: InvestmentReport,
    ) -> EvaluationResult:
        """Evaluate a complete EVIR research artifact set."""

        scores = {
            "evidence_coverage_score": evidence_coverage_score(workspace),
            "temporal_validity_score": temporal_validity_score(workspace),
            "conflict_coverage_score": conflict_coverage_score(
                merge_result,
                workspace,
            ),
            "decision_traceability_score": decision_traceability_score(
                merge_result,
                workspace,
            ),
            "audit_completeness_score": audit_completeness_score(report),
        }
        overall_score = round(sum(scores.values()) / len(scores), 2)

        return EvaluationResult(
            **scores,
            overall_score=overall_score,
            details={
                "ticker": workspace.ticker,
                "branch_count": len(workspace.branches),
                "commit_count": sum(
                    len(branch.commits)
                    for branch in workspace.branches.values()
                ),
                "conflict_count": len(merge_result.key_conflicts),
                "final_recommendation": merge_result.final_recommendation,
            },
        )


def save_evaluation_result(
    evaluation_result: EvaluationResult,
    ticker: str,
    output_dir: str | Path = "outputs/evaluation",
) -> str:
    """Save an evaluation result as JSON and return the saved path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evaluation_path = output_path / f"{ticker}_evaluation.json"
    save_json(evaluation_path, evaluation_result)
    return str(evaluation_path)
