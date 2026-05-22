"""Risk review agent."""

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from llm import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


RiskDimension = Literal[
    "valuation_risk_review",
    "technical_timing_risk",
    "bullish_thesis_challenge",
    "evidence_quality_risk",
    "portfolio_positioning_risk",
]


class RiskInsight(BaseModel):
    """One LLM-generated risk review insight."""

    dimension: RiskDimension
    claim: str = Field(min_length=8)
    confidence: Literal["low", "medium", "high"] = "medium"
    risk_tag: str = "risk_review"
    time_horizon: str = "ongoing"


class RiskAnalysisOutput(BaseModel):
    """Validated LLM output for risk review."""

    insights: list[RiskInsight] = Field(default_factory=list)


class RiskAgent(BaseAgent):
    """Creates financial and reasoning-audit risk commits."""

    name = "RiskAgent"
    role = "risk-agent"
    branch_name = "risk-review"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(
        self,
        input_data: dict[str, Any] | list[ClaimEvidenceCommit],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Review all prior commits for financial and reasoning risks."""

        del input_data
        prior_commits = [
            commit
            for branch_name, branch in workspace.branches.items()
            if branch_name != self.branch_name
            for commit in branch.commits
        ]
        source_commits = _risk_source_commits(prior_commits)
        llm_commits = self._llm_commits(workspace, prior_commits, source_commits)
        if llm_commits:
            return llm_commits
        return self._deterministic_commits(source_commits)

    def _deterministic_commits(
        self,
        source_commits: dict[RiskDimension, ClaimEvidenceCommit],
    ) -> list[ClaimEvidenceCommit]:
        """Return the original deterministic risk review commits."""

        valuation_commit = source_commits.get("valuation_risk_review")
        volatility_commit = source_commits.get("technical_timing_risk")
        bullish_without_counter = source_commits.get("bullish_thesis_challenge")
        stale_or_gap_commit = source_commits.get("evidence_quality_risk")
        risk_commits: list[ClaimEvidenceCommit] = []

        if valuation_commit is not None:
            risk_commits.append(
                self.create_commit(
                    claim="估值风险应降低对上行情景的信心。",
                    evidence=valuation_commit.evidence,
                    confidence="medium",
                    risk_tag="valuation_risk",
                    time_horizon="6-12 months",
                )
            )
        if volatility_commit is not None:
            risk_commits.append(
                self.create_commit(
                    claim="中高波动率会给短期入场带来择时风险。",
                    evidence=volatility_commit.evidence,
                    confidence="medium",
                    risk_tag="volatility_risk",
                    time_horizon="1-3 months",
                )
            )
        if bullish_without_counter is not None:
            risk_commits.append(
                self.create_commit(
                    claim=(
                        "偏多投资假设需要明确反证检查后才能提高置信度。"
                    ),
                    evidence=bullish_without_counter.evidence,
                    confidence="medium",
                    risk_tag="evidence_gap",
                    time_horizon="ongoing",
                )
            )
        if stale_or_gap_commit is not None:
            risk_commits.append(
                self.create_commit(
                    claim="证据时效不确定或证据质量偏弱，需要持续跟踪。",
                    evidence=stale_or_gap_commit.evidence,
                    confidence="low",
                    risk_tag="temporal_uncertainty",
                    time_horizon="ongoing",
                )
            )

        return risk_commits[:4]

    def _llm_commits(
        self,
        workspace: Workspace,
        prior_commits: list[ClaimEvidenceCommit],
        source_commits: dict[RiskDimension, ClaimEvidenceCommit],
    ) -> list[ClaimEvidenceCommit]:
        """Ask an optional LLM for risk review insights."""

        if self.llm_client is None or not source_commits:
            return []

        try:
            response = self.llm_client.complete(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a risk-review agent for an auditable "
                            "investment research workflow. Use only the supplied "
                            "commit packet. Do not invent evidence. Return only "
                            "valid JSON."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=_build_llm_prompt(workspace, prior_commits, source_commits),
                    ),
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parsed = parse_json_response(response.content)
            validated = RiskAnalysisOutput.model_validate(parsed)
        except (LLMError, ValidationError):
            return []

        valid_insights = _validate_risk_insights(validated.insights, source_commits)
        if valid_insights is None:
            return []

        commits: list[ClaimEvidenceCommit] = []
        seen_dimensions: set[RiskDimension] = set()
        for insight in valid_insights:
            if insight.dimension in seen_dimensions:
                continue
            source_commit = source_commits.get(insight.dimension)
            if source_commit is None:
                return []
            seen_dimensions.add(insight.dimension)
            commits.append(
                self.create_commit(
                    claim=insight.claim.strip(),
                    evidence=source_commit.evidence,
                    confidence=insight.confidence,
                    risk_tag=_normalize_llm_risk_tag(insight.risk_tag, insight.dimension),
                    time_horizon=insight.time_horizon.strip() or "ongoing",
                )
            )

        if not all(dimension in seen_dimensions for dimension in source_commits):
            return []
        return commits[:5]


def _find_commit(
    commits: list[ClaimEvidenceCommit],
    keyword: str,
) -> ClaimEvidenceCommit | None:
    """Find the first commit whose claim, tag, or evidence references a keyword."""

    lowered_keyword = keyword.lower()
    for commit in commits:
        haystack = (
            f"{commit.claim} {commit.risk_tag} {commit.evidence.content} "
            f"{commit.evidence.metric_name or ''}"
        ).lower()
        if lowered_keyword in haystack:
            return commit
    return None


def _risk_source_commits(
    prior_commits: list[ClaimEvidenceCommit],
) -> dict[RiskDimension, ClaimEvidenceCommit]:
    """Map risk-review dimensions to the prior commit they audit."""

    sources: dict[RiskDimension, ClaimEvidenceCommit] = {}
    valuation_commit = _find_commit(prior_commits, "valuation")
    volatility_commit = _find_commit(prior_commits, "volatility")
    bullish_without_counter = _find_first_bullish_without_counter(prior_commits)
    stale_or_gap_commit = _find_temporal_or_low_quality_gap(prior_commits)
    portfolio_commit = _find_commit(prior_commits, "portfolio")

    if valuation_commit is not None:
        sources["valuation_risk_review"] = valuation_commit
    if volatility_commit is not None:
        sources["technical_timing_risk"] = volatility_commit
    if bullish_without_counter is not None:
        sources["bullish_thesis_challenge"] = bullish_without_counter
    if stale_or_gap_commit is not None:
        sources["evidence_quality_risk"] = stale_or_gap_commit
    if portfolio_commit is not None:
        sources["portfolio_positioning_risk"] = portfolio_commit

    return sources


def _build_llm_prompt(
    workspace: Workspace,
    prior_commits: list[ClaimEvidenceCommit],
    source_commits: dict[RiskDimension, ClaimEvidenceCommit],
) -> str:
    """Build a compact risk-review packet for the LLM."""

    source_lines = []
    for dimension, commit in source_commits.items():
        source_lines.append(
            "\n".join(
                [
                    f"- dimension: {dimension}",
                    f"  branch: {commit.branch_name}",
                    f"  claim: {commit.claim}",
                    f"  evidence: {commit.evidence.content}",
                    f"  metric: {commit.evidence.metric_name}={commit.evidence.metric_value}",
                    f"  risk_tag: {commit.risk_tag}",
                    f"  confidence: {commit.confidence}",
                    f"  evidence_quality_score: {commit.evidence_quality_score}",
                    f"  temporal_status: {commit.temporal_status}",
                    f"  time_horizon: {commit.time_horizon}",
                ]
            )
        )

    branch_counts: dict[str, int] = {}
    for commit in prior_commits:
        branch_counts[commit.branch_name] = branch_counts.get(commit.branch_name, 0) + 1

    dimensions = "|".join(source_commits)
    return (
        f"Ticker: {workspace.ticker}\n"
        f"Company: {workspace.company_name or 'unknown'}\n"
        f"Research question: {workspace.research_question}\n"
        f"Prior branch counts: {branch_counts}\n\n"
        "Risk review source commits:\n"
        + "\n".join(source_lines)
        + "\n\n"
        "Generate exactly one risk review insight for each listed dimension. "
        "Use concise investment risk language. Do not add facts that are not "
        "present in the source commits. If a dimension audits evidence quality "
        "or freshness, explain why that limits conviction.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "insights": [\n'
        "    {\n"
        f'      "dimension": "{dimensions}",\n'
        '      "claim": "evidence-grounded risk claim",\n'
        '      "confidence": "low|medium|high",\n'
        '      "risk_tag": "stable_snake_case_tag",\n'
        '      "time_horizon": "ongoing"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _validate_risk_insights(
    insights: list[RiskInsight],
    source_commits: dict[RiskDimension, ClaimEvidenceCommit],
) -> list[RiskInsight] | None:
    """Reject incomplete or dimension-inconsistent LLM risk output."""

    if len(insights) < len(source_commits):
        return None

    valid_insights: list[RiskInsight] = []
    seen_dimensions: set[RiskDimension] = set()
    for insight in insights:
        if insight.dimension in seen_dimensions:
            continue
        source_commit = source_commits.get(insight.dimension)
        if source_commit is None:
            continue
        if not _insight_matches_source(insight, source_commit):
            return None
        seen_dimensions.add(insight.dimension)
        valid_insights.append(insight)
        if len(valid_insights) == len(source_commits):
            break

    if not all(dimension in seen_dimensions for dimension in source_commits):
        return None
    return valid_insights


def _insight_matches_source(
    insight: RiskInsight,
    source_commit: ClaimEvidenceCommit,
) -> bool:
    """Check that the LLM risk dimension matches the audited source commit."""

    source_text = (
        f"{source_commit.claim} {source_commit.risk_tag} "
        f"{source_commit.evidence.content} {source_commit.evidence.metric_name or ''}"
    ).lower()
    insight_text = f"{insight.claim} {insight.risk_tag}".lower()

    if insight.dimension == "valuation_risk_review":
        return "valuation" in source_text or "估值" in source_text
    if insight.dimension == "technical_timing_risk":
        return "volatility" in source_text or "波动" in source_text
    if insight.dimension == "bullish_thesis_challenge":
        return source_commit.branch_name == "bull-case" or "bullish" in source_text or "偏多" in source_text
    if insight.dimension == "evidence_quality_risk":
        if source_commit.temporal_status in {"stale", "expired", "unknown"}:
            return True
        if source_commit.evidence_quality_score is not None and source_commit.evidence_quality_score < 0.75:
            return True
        return "evidence" in insight_text or "temporal" in insight_text or "证据" in insight_text
    if insight.dimension == "portfolio_positioning_risk":
        return "portfolio" in source_text or "position" in source_text or "组合" in source_text or "仓位" in source_text
    return False


def _normalize_llm_risk_tag(risk_tag: str, dimension: RiskDimension) -> str:
    """Normalize LLM risk tags so merge rules can still classify risk."""

    normalized = risk_tag.lower().strip().replace("-", "_").replace(" ", "_")
    if dimension == "valuation_risk_review":
        return "valuation_risk"
    if dimension == "technical_timing_risk":
        return "volatility_risk"
    if dimension == "bullish_thesis_challenge":
        return "evidence_gap"
    if dimension == "evidence_quality_risk":
        return "temporal_uncertainty"
    if dimension == "portfolio_positioning_risk":
        if "gap" in normalized or "unknown" in normalized:
            return "portfolio_evidence_gap"
        return "portfolio_risk"
    if not normalized:
        return "risk_review"
    return normalized[:80]


def _find_first_bullish_without_counter(
    commits: list[ClaimEvidenceCommit],
) -> ClaimEvidenceCommit | None:
    """Find a bullish claim that lacks explicit counter-evidence."""

    for commit in commits:
        if commit.branch_name == "bull-case" and not commit.counter_evidence:
            return commit
    return None


def _find_temporal_or_low_quality_gap(
    commits: list[ClaimEvidenceCommit],
) -> ClaimEvidenceCommit | None:
    """Find a commit with stale/unknown timing or lower evidence quality."""

    for commit in commits:
        if commit.temporal_status in {"unknown", "watch_staleness", "stale"}:
            return commit
        if commit.evidence_quality_score is not None and commit.evidence_quality_score < 0.75:
            return commit
    return None
