"""LLM-powered synthesis agent."""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from llm import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


SOURCE_BRANCHES = (
    "fundamental-analysis",
    "financial-statement-analysis",
    "valuation-analysis",
    "industry-comparison",
    "macro-analysis",
    "technical-analysis",
    "backtest-analysis",
    "portfolio-review",
)


class LLMInsight(BaseModel):
    """One structured LLM-generated investment insight."""

    claim: str = Field(min_length=8)
    evidence_ref: str = ""
    evidence_commit_id: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"
    risk_tag: str = "llm_synthesis"
    time_horizon: str = "6-12 months"
    counter_evidence: list[str] = Field(default_factory=list)


class LLMAnalysisOutput(BaseModel):
    """Validated LLM synthesis response."""

    insights: list[LLMInsight] = Field(default_factory=list)


class LLMAnalysisAgent:
    """Uses a real LLM to synthesize multi-dimensional stock analysis."""

    name = "LLMAnalysisAgent"
    role = "llm-analysis-agent"
    branch_name = "llm-analysis"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(self, workspace: Workspace) -> list[ClaimEvidenceCommit]:
        """Generate structured, evidence-linked LLM analysis commits."""

        source_commits = _source_commits(workspace)
        if self.llm_client is None or not source_commits:
            return []

        try:
            response = self.llm_client.complete(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "你是严谨的股票研究分析师。你必须基于给定证据生成中文分析，"
                            "不能编造未给出的事实。只返回 JSON。"
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=_build_prompt(workspace, source_commits),
                    ),
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parsed = parse_json_response(response.content)
            validated = LLMAnalysisOutput.model_validate(parsed)
        except (LLMError, ValidationError):
            return []

        commits_by_ref = {_commit_ref(commit): commit for commit in source_commits}
        commits_by_id = {commit.commit_id: commit for commit in source_commits}
        fallback_commit = _best_evidence_commit(source_commits)
        generated_commits: list[ClaimEvidenceCommit] = []
        seen_claims: set[str] = set()

        for insight in validated.insights[:4]:
            claim = insight.claim.strip()
            if not claim or claim in seen_claims:
                continue
            source_commit = (
                commits_by_ref.get(insight.evidence_ref.strip())
                or commits_by_id.get(insight.evidence_commit_id)
                or fallback_commit
            )
            seen_claims.add(claim)
            generated_commits.append(
                ClaimEvidenceCommit(
                    commit_id=str(uuid4()),
                    agent_role=self.role,
                    branch_name=self.branch_name,
                    claim=claim,
                    evidence=source_commit.evidence,
                    evidence_quality_score=source_commit.evidence_quality_score,
                    confidence=insight.confidence,
                    risk_tag=_normalize_risk_tag(insight.risk_tag),
                    time_horizon=insight.time_horizon,
                    temporal_status=source_commit.temporal_status,
                    counter_evidence=[
                        question.strip()
                        for question in insight.counter_evidence
                        if question.strip()
                    ] or None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        return generated_commits


def _source_commits(workspace: Workspace) -> list[ClaimEvidenceCommit]:
    """Collect deterministic source commits for LLM synthesis."""

    commits: list[ClaimEvidenceCommit] = []
    for branch_name in SOURCE_BRANCHES:
        branch = workspace.branches.get(branch_name)
        if branch is not None:
            commits.extend(branch.commits)
    return commits


def _build_prompt(workspace: Workspace, commits: list[ClaimEvidenceCommit]) -> str:
    """Build a compact evidence packet for the LLM."""

    evidence_lines = []
    for commit in commits[:24]:
        evidence_ref = _commit_ref(commit)
        evidence_lines.append(
            "\n".join(
                [
                    f"- evidence_ref: {evidence_ref}",
                    f"  branch: {commit.branch_name}",
                    f"  claim: {commit.claim}",
                    f"  evidence: {commit.evidence.content}",
                    f"  metric: {commit.evidence.metric_name}={commit.evidence.metric_value}",
                    f"  source_type: {commit.evidence.source_type}",
                    f"  evidence_quality_score: {commit.evidence_quality_score}",
                    f"  temporal_status: {commit.temporal_status}",
                    f"  risk_tag: {commit.risk_tag}",
                ]
            )
        )

    return (
        f"股票：{workspace.ticker}\n"
        f"公司：{workspace.company_name or '未知'}\n"
        f"研究问题：{workspace.research_question}\n\n"
        "请基于以下证据，输出 2 到 4 条中文综合洞察。洞察应覆盖基本面、估值、"
        "行业/宏观、技术面、风险或组合约束中的多个维度。每条洞察必须引用一个最相关的 "
        "evidence_ref。\n\n"
        "返回 JSON，格式必须是：\n"
        "{\n"
        '  "insights": [\n'
        "    {\n"
        '      "claim": "中文洞察",\n'
        '      "evidence_ref": "上方某个 evidence_ref",\n'
        '      "confidence": "low|medium|high",\n'
        '      "risk_tag": "llm_support|llm_risk|llm_mixed|llm_evidence_gap",\n'
        '      "time_horizon": "6-12 months",\n'
        '      "counter_evidence": ["需要继续验证的问题"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "证据：\n"
        + "\n".join(evidence_lines)
    )


def _commit_ref(commit: ClaimEvidenceCommit) -> str:
    """Return a stable evidence reference that does not include random commit ids."""

    metric = commit.evidence.metric_name or "unlabeled_metric"
    evidence_id = commit.evidence.evidence_id or "unlabeled_evidence"
    return f"{commit.branch_name}:{metric}:{evidence_id}"


def _best_evidence_commit(
    commits: list[ClaimEvidenceCommit],
) -> ClaimEvidenceCommit:
    """Pick the strongest available evidence commit as fallback."""

    return max(
        commits,
        key=lambda commit: (
            commit.evidence_quality_score or 0.0,
            1 if commit.temporal_status == "valid" else 0,
        ),
    )


def _normalize_risk_tag(risk_tag: str) -> str:
    """Keep LLM risk tags stable enough for merge classification."""

    normalized = risk_tag.lower().strip().replace("-", "_").replace(" ", "_")
    allowed_prefixes = ("llm_support", "llm_risk", "llm_mixed", "llm_evidence_gap")
    if normalized in allowed_prefixes:
        return normalized
    if "risk" in normalized:
        return "llm_risk"
    if "gap" in normalized:
        return "llm_evidence_gap"
    if "support" in normalized:
        return "llm_support"
    return "llm_mixed"
