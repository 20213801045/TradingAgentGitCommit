"""Research coordination agent for EVIR."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from llm import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


REQUIRED_RESEARCH_BRANCHES = (
    "fundamental-analysis",
    "financial-statement-analysis",
    "valuation-analysis",
    "industry-comparison",
    "macro-analysis",
    "technical-analysis",
    "backtest-analysis",
    "portfolio-review",
    "llm-analysis",
    "bull-case",
    "bear-case",
    "risk-review",
    "counter-evidence",
)


class CoordinatorOutput(BaseModel):
    """Validated coordination output from an optional LLM."""

    research_plan: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    follow_up_priorities: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class ResearchCoordinatorAgent(BaseAgent):
    """Creates auditable planning and coverage-review commits."""

    name = "ResearchCoordinatorAgent"
    role = "research-coordinator-agent"
    branch_name = "research-coordination"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate a coordination commit for the requested workflow phase."""

        phase = str(input_data.get("phase", "pre_merge_review"))
        output = self._llm_output(input_data, workspace, phase)
        if output is None:
            output = _deterministic_output(workspace, phase)

        if phase == "initial_plan":
            return [self._plan_commit(output, workspace)]
        return [self._review_commit(output, workspace, phase)]

    def _llm_output(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
        phase: str,
    ) -> CoordinatorOutput | None:
        """Ask an optional LLM for a structured coordination note."""

        if self.llm_client is None:
            return None

        try:
            response = self.llm_client.complete(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "You are the research coordinator for an auditable "
                            "multi-agent investment research workflow. Return "
                            "only valid JSON."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=_build_prompt(input_data, workspace, phase),
                    ),
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parsed = parse_json_response(response.content)
            return CoordinatorOutput.model_validate(parsed)
        except (LLMError, ValidationError):
            return None

    def _plan_commit(
        self,
        output: CoordinatorOutput,
        workspace: Workspace,
    ) -> ClaimEvidenceCommit:
        """Create the initial planning commit."""

        evidence = self._make_evidence(
            content=_coordination_content(
                title="Initial research plan",
                items=output.research_plan,
                quality_checks=output.quality_checks,
                follow_up_priorities=output.follow_up_priorities,
                workspace=workspace,
            ),
            source="EVIR Research Coordinator",
            source_type="official_report",
            timestamp=self._utc_now(),
            metric_name="coordination_phase",
            metric_value="initial_plan",
        )
        return self.create_commit(
            claim="Coordinator created the initial multi-branch research plan.",
            evidence=evidence,
            confidence=output.confidence,
            risk_tag="coordination_plan",
            time_horizon="research cycle",
        )

    def _review_commit(
        self,
        output: CoordinatorOutput,
        workspace: Workspace,
        phase: str,
    ) -> ClaimEvidenceCommit:
        """Create a pre-merge or follow-up coverage-review commit."""

        evidence = self._make_evidence(
            content=_coordination_content(
                title="Coordinator coverage review",
                items=output.research_plan,
                quality_checks=output.quality_checks,
                follow_up_priorities=output.follow_up_priorities,
                workspace=workspace,
            ),
            source="EVIR Research Coordinator",
            source_type="official_report",
            timestamp=self._utc_now(),
            metric_name="coordination_phase",
            metric_value=phase,
        )
        return self.create_commit(
            claim="Coordinator completed the pre-merge coverage review.",
            evidence=evidence,
            confidence=output.confidence,
            risk_tag="coordination_review",
            time_horizon="research cycle",
        )


def _deterministic_output(workspace: Workspace, phase: str) -> CoordinatorOutput:
    """Return a stable no-LLM coordination output."""

    stats = _branch_stats(workspace)
    empty_required = [
        branch_name
        for branch_name in REQUIRED_RESEARCH_BRANCHES
        if stats.get(branch_name, 0) == 0
    ]
    populated_required = [
        branch_name
        for branch_name in REQUIRED_RESEARCH_BRANCHES
        if stats.get(branch_name, 0) > 0
    ]

    if phase == "initial_plan":
        return CoordinatorOutput(
            research_plan=[
                "Run source-analysis agents before thesis, risk, and merge stages.",
                "Keep every specialist conclusion attached to explicit evidence.",
                "Use counter-evidence before final synthesis.",
                "Preserve branch-level auditability for the final report.",
            ],
            quality_checks=[
                "Check evidence source, timestamp, metric name, and metric value.",
                "Flag stale evidence before the merge step.",
                "Keep specialist outputs in their own branches.",
            ],
            follow_up_priorities=[
                "Migrate FundamentalAgent to LLM-first with deterministic fallback.",
                "Add coordinator-driven follow-up requests after branch review.",
            ],
            confidence="medium",
        )

    if empty_required:
        confidence: Literal["low", "medium", "high"] = "medium"
    else:
        confidence = "high"

    return CoordinatorOutput(
        research_plan=[
            f"Completed required branches: {', '.join(populated_required) or 'none'}.",
            f"Branches still empty: {', '.join(empty_required) or 'none'}.",
        ],
        quality_checks=[
            f"Workspace contains {sum(stats.values())} coordination-visible commits.",
            "Final synthesis should keep support, caution, and counter-evidence traceable.",
            "Future coordinator versions should request reruns for incomplete branches.",
        ],
        follow_up_priorities=[
            "Review whether positive claims have explicit counter-evidence questions.",
            "Review whether time-sensitive technical and backtest evidence remains fresh.",
            "Review whether valuation constraints are reflected before recommendation.",
        ],
        confidence=confidence,
    )


def _build_prompt(
    input_data: dict[str, Any],
    workspace: Workspace,
    phase: str,
) -> str:
    """Build a compact coordinator prompt."""

    stats = _branch_stats(workspace)
    company_data = input_data.get("company_data", {})
    return (
        f"Ticker: {workspace.ticker}\n"
        f"Company: {workspace.company_name or 'unknown'}\n"
        f"Research question: {workspace.research_question}\n"
        f"Workflow phase: {phase}\n"
        f"Company data keys: {sorted(company_data.keys()) if isinstance(company_data, dict) else []}\n"
        f"Branch commit counts: {stats}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "research_plan": ["step"],\n'
        '  "quality_checks": ["check"],\n'
        '  "follow_up_priorities": ["priority"],\n'
        '  "confidence": "low|medium|high"\n'
        "}\n"
    )


def _coordination_content(
    title: str,
    items: list[str],
    quality_checks: list[str],
    follow_up_priorities: list[str],
    workspace: Workspace,
) -> str:
    """Build evidence content for a coordinator commit."""

    return "\n".join(
        [
            f"{title} for {workspace.ticker}.",
            "Research plan: " + _join_or_none(items),
            "Quality checks: " + _join_or_none(quality_checks),
            "Follow-up priorities: " + _join_or_none(follow_up_priorities),
        ]
    )


def _branch_stats(workspace: Workspace) -> dict[str, int]:
    """Return commit counts per workspace branch."""

    return {
        branch_name: len(branch.commits)
        for branch_name, branch in workspace.branches.items()
        if branch_name != "research-coordination"
    }


def _join_or_none(items: list[str]) -> str:
    """Join list content into a compact evidence sentence."""

    cleaned = [item.strip() for item in items if item.strip()]
    if not cleaned:
        return "none"
    return " | ".join(cleaned)
