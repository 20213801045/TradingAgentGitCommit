"""Counter-evidence search agent."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from llm import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


POSITIVE_CLAIM_KEYWORDS = {
    "growth",
    "strong",
    "stable",
    "bullish",
    "constructive",
    "upward",
    "momentum",
    "improves",
    "support",
    "supports",
    "financial health",
    "增长",
    "强",
    "稳定",
    "偏多",
    "建设性",
    "上行",
    "动量",
    "支持",
    "现金",
    "健康",
    "可控",
    "吸引力",
}

SOURCE_BRANCHES = (
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
)

COUNTER_EVIDENCE_MAX_WORKERS = 4

CounterEvidenceCandidate = tuple[ClaimEvidenceCommit, str, str, list[str]]


class CounterEvidenceLLMOutput(BaseModel):
    """Validated JSON output for one LLM-generated question."""

    question: str


class CounterEvidenceAgent:
    """Generates deterministic counter-evidence checks for positive claims."""

    name = "Counter-Evidence Agent"
    role = "Counter-Evidence Analyst"
    branch_name = "counter-evidence"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(self, workspace: Workspace) -> list[ClaimEvidenceCommit]:
        """Inspect important positive commits and emit counter-evidence commits."""

        generated_claims: set[str] = set()
        candidates: list[CounterEvidenceCandidate] = []

        for source_commit in self._source_commits(workspace):
            if not self._is_important_positive_claim(source_commit):
                continue

            mapping = self._counter_mapping(source_commit.claim)
            if mapping is None:
                continue

            claim, risk_tag, questions = mapping
            if claim in generated_claims:
                continue

            generated_claims.add(claim)
            candidates.append((source_commit, claim, risk_tag, questions))

        if self.llm_client is not None and candidates:
            candidates = self._augment_candidates_with_llm(candidates)

        return [
            self._build_counter_commit(source_commit, claim, risk_tag, questions)
            for source_commit, claim, risk_tag, questions in candidates
        ]

    def _source_commits(self, workspace: Workspace) -> list[ClaimEvidenceCommit]:
        """Collect commits from branches eligible for counter-evidence review."""

        commits: list[ClaimEvidenceCommit] = []
        for branch_name in SOURCE_BRANCHES:
            branch = workspace.branches.get(branch_name)
            if branch is not None:
                commits.extend(branch.commits)
        return commits

    def _is_important_positive_claim(self, commit: ClaimEvidenceCommit) -> bool:
        """Return whether a commit is positive enough to challenge."""

        quality = commit.evidence_quality_score or 0.0
        return quality >= 0.7 and _contains_any(commit.claim, POSITIVE_CLAIM_KEYWORDS)

    def _counter_mapping(
        self,
        claim: str,
    ) -> tuple[str, str, list[str]] | None:
        """Map a positive claim to a deterministic counter-evidence prompt."""

        lowered_claim = claim.lower()

        if "growth" in lowered_claim or "增长" in lowered_claim:
            return (
                "收入增长需要验证是否存在放缓或未来指引转弱。",
                "counter_evidence_growth",
                [
                    "是否有证据显示收入增长正在放缓？",
                    "管理层指引是否弱于预期？",
                    "正向增长信号是否已经被估值充分反映？",
                ],
            )

        if _contains_any(
            lowered_claim,
            {"profitability", "margin", "stable", "盈利", "利润率", "稳定"},
        ):
            return (
                "盈利能力强度需要检查利润率压力或成本通胀。",
                "counter_evidence_profitability",
                [
                    "是否有利润率压缩的证据？",
                    "成本增速是否快于收入增速？",
                    "盈利能力是否由一次性因素支撑？",
                ],
            )

        if _contains_any(
            lowered_claim,
            {"momentum", "trend", "constructive", "upward", "动量", "趋势", "建设性", "上行"},
        ):
            return (
                "技术动量需要更新指标验证，因为过期技术证据可能削弱信号。",
                "counter_evidence_technical",
                [
                    "最新技术指标是否仍然支持该信号？",
                    "价格动量是否已经较上次信号转弱？",
                    "股价是否跌破关键支撑位？",
                ],
            )

        if _contains_any(
            lowered_claim,
            {"financial health", "cash", "balance sheet", "财务健康", "现金", "资产负债表"},
        ):
            return (
                "资产负债表强度需要结合资本配置需求和未来现金流压力评估。",
                "counter_evidence_balance_sheet",
                [
                    "未来现金流需求是否正在上升？",
                    "现金是否被回购、偿债或资本开支消耗？",
                    "资本配置是否可能降低资产负债表灵活性？",
                ],
            )

        if "bullish" in lowered_claim or "偏多" in lowered_claim:
            return (
                "偏多投资假设需要明确下行验证后才能提高置信度。",
                "counter_evidence_bullish",
                [
                    "什么证据会推翻偏多投资假设？",
                    "下行风险是否已经被谨慎分支充分反映？",
                    "偏多观点是否过度依赖单一证据来源？",
                ],
            )

        return None

    def _augment_questions_with_llm(
        self,
        source_commit: ClaimEvidenceCommit,
        counter_claim: str,
        questions: list[str],
    ) -> list[str]:
        """Optionally ask an LLM for one validated counter-evidence question."""

        if self.llm_client is None:
            return questions

        try:
            response = self.llm_client.complete(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "你为投资研究生成简洁的中文反证问题。只返回 JSON，"
                            '格式为：{"question": "..."}。'
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            f"原始观点：{source_commit.claim}\n"
                            f"反证观点：{counter_claim}\n"
                            f"证据：{source_commit.evidence.content}\n"
                            "请生成一个额外的中文问题，用于检验原始观点是否被夸大。"
                            "不要在 JSON 对象之外输出任何文字。"
                        ),
                    ),
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parsed = parse_json_response(response.content)
            validated = CounterEvidenceLLMOutput.model_validate(parsed)
        except (LLMError, ValidationError):
            return questions

        llm_question = validated.question.strip()
        if not llm_question:
            return questions
        if not llm_question.endswith("?"):
            llm_question = f"{llm_question}?"
        if llm_question not in questions:
            return [*questions, llm_question]
        return questions

    def _augment_candidates_with_llm(
        self,
        candidates: list[CounterEvidenceCandidate],
    ) -> list[CounterEvidenceCandidate]:
        """Add optional LLM questions concurrently while preserving source order."""

        max_workers = min(COUNTER_EVIDENCE_MAX_WORKERS, len(candidates))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._augment_one_candidate_with_llm, candidate)
                for candidate in candidates
            ]
            return [future.result() for future in futures]

    def _augment_one_candidate_with_llm(
        self,
        candidate: CounterEvidenceCandidate,
    ) -> CounterEvidenceCandidate:
        """Add one LLM-generated question for a counter-evidence candidate."""

        source_commit, claim, risk_tag, questions = candidate
        augmented_questions = self._augment_questions_with_llm(
            source_commit=source_commit,
            counter_claim=claim,
            questions=questions,
        )
        return source_commit, claim, risk_tag, augmented_questions

    def _build_counter_commit(
        self,
        source_commit: ClaimEvidenceCommit,
        claim: str,
        risk_tag: str,
        questions: list[str],
    ) -> ClaimEvidenceCommit:
        """Build a structured counter-evidence commit from a reviewed claim."""

        return ClaimEvidenceCommit(
            commit_id=str(uuid4()),
            agent_role=self.role,
            branch_name=self.branch_name,
            claim=claim,
            evidence=source_commit.evidence,
            evidence_quality_score=source_commit.evidence_quality_score,
            confidence="medium",
            risk_tag=risk_tag,
            time_horizon=source_commit.time_horizon,
            temporal_status=source_commit.temporal_status,
            counter_evidence=questions,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def _contains_any(text: str, keywords: set[str]) -> bool:
    """Return whether lowercase text contains any keyword."""

    lowered_text = text.lower()
    return any(keyword in lowered_text for keyword in keywords)
