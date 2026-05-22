"""Valuation analysis agent."""

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from config import INVESTMENT_THRESHOLDS
from llm import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


ValuationDimension = Literal[
    "relative_valuation",
    "growth_adjusted_valuation",
]
EvidenceInputs = dict[str, tuple[str, str, str, str]]


class ValuationInsight(BaseModel):
    """One LLM-generated valuation insight."""

    dimension: ValuationDimension
    claim: str = Field(min_length=8)
    confidence: Literal["low", "medium", "high"] = "medium"
    risk_tag: str = "valuation_analysis"
    time_horizon: str = "6-12 months"


class ValuationAnalysisOutput(BaseModel):
    """Validated LLM output for valuation analysis."""

    insights: list[ValuationInsight] = Field(default_factory=list)


class ValuationAgent(BaseAgent):
    """Creates commits about valuation quality and price expectations."""

    name = "ValuationAgent"
    role = "valuation-agent"
    branch_name = "valuation-analysis"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate LLM-backed valuation commits with deterministic fallback."""

        evidence_by_dimension = self._evidence_by_dimension(input_data)
        llm_commits = self._llm_commits(input_data, workspace, evidence_by_dimension)
        if llm_commits:
            return llm_commits
        return self._deterministic_commits(input_data, evidence_by_dimension)

    def _evidence_by_dimension(self, input_data: dict[str, Any]) -> EvidenceInputs:
        """Build reusable evidence inputs keyed by valuation dimension."""

        metrics = input_data.get("valuation_metrics", {})
        timestamp = str(input_data.get("as_of_date", _news_timestamp(input_data)))
        source = _source_from_input(input_data, "Valuation Model")
        forward_pe = _value(metrics, "forward_pe")
        sector_forward_pe = _value(metrics, "sector_forward_pe")
        earnings_growth = _value(metrics, "earnings_growth_yoy")
        fcf_yield = _value(metrics, "free_cash_flow_yield")

        return {
            "relative_valuation": (
                f"远期市盈率为 {forward_pe}，行业基准为 {sector_forward_pe}。",
                "relative_forward_pe",
                f"{forward_pe}/{sector_forward_pe}",
                "6-12 months",
            ),
            "growth_adjusted_valuation": (
                f"盈利增长为 {earnings_growth}，自由现金流收益率为 {fcf_yield}。",
                "growth_adjusted_valuation",
                f"{earnings_growth}/{fcf_yield}",
                "12 months",
            ),
            "_source": ("", source, "valuation_model", timestamp),
        }

    def _deterministic_commits(
        self,
        input_data: dict[str, Any],
        evidence_by_dimension: EvidenceInputs,
    ) -> list[ClaimEvidenceCommit]:
        """Generate deterministic valuation commits from valuation metrics."""

        metrics = input_data.get("valuation_metrics", {})
        forward_pe = _value(metrics, "forward_pe")
        sector_forward_pe = _value(metrics, "sector_forward_pe")
        earnings_growth = _value(metrics, "earnings_growth_yoy")
        fcf_yield = _value(metrics, "free_cash_flow_yield")

        return [
            self.create_commit(
                claim=_relative_pe_claim(forward_pe, sector_forward_pe),
                evidence=self._evidence_for_dimension(
                    "relative_valuation",
                    evidence_by_dimension,
                ),
                confidence=_confidence(forward_pe, sector_forward_pe),
                risk_tag=_relative_pe_risk_tag(forward_pe, sector_forward_pe),
                time_horizon="6-12 months",
            ),
            self.create_commit(
                claim=_cash_yield_claim(earnings_growth, fcf_yield),
                evidence=self._evidence_for_dimension(
                    "growth_adjusted_valuation",
                    evidence_by_dimension,
                ),
                confidence=_confidence(earnings_growth, fcf_yield),
                risk_tag=_cash_yield_risk_tag(earnings_growth, fcf_yield),
                time_horizon="12 months",
            ),
        ]

    def _llm_commits(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
        evidence_by_dimension: EvidenceInputs,
    ) -> list[ClaimEvidenceCommit]:
        """Ask an optional LLM for valuation insights."""

        if self.llm_client is None:
            return []

        try:
            response = self.llm_client.complete(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a valuation-focused equity research agent. "
                            "Use only the supplied valuation metrics. Do not "
                            "invent facts. Return only valid JSON."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=_build_llm_prompt(input_data, workspace),
                    ),
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parsed = parse_json_response(response.content)
            validated = ValuationAnalysisOutput.model_validate(parsed)
        except (LLMError, ValidationError):
            return []

        commits: list[ClaimEvidenceCommit] = []
        seen_dimensions: set[ValuationDimension] = set()
        for insight in validated.insights[:2]:
            if insight.dimension in seen_dimensions:
                continue
            seen_dimensions.add(insight.dimension)
            default_horizon = evidence_by_dimension[insight.dimension][3]
            commits.append(
                self.create_commit(
                    claim=insight.claim.strip(),
                    evidence=self._evidence_for_dimension(
                        insight.dimension,
                        evidence_by_dimension,
                    ),
                    confidence=insight.confidence,
                    risk_tag=_normalize_llm_risk_tag(insight.risk_tag),
                    time_horizon=insight.time_horizon.strip() or default_horizon,
                )
            )

        required_dimensions: tuple[ValuationDimension, ...] = (
            "relative_valuation",
            "growth_adjusted_valuation",
        )
        if not all(dimension in seen_dimensions for dimension in required_dimensions):
            return []
        return commits

    def _evidence_for_dimension(
        self,
        dimension: ValuationDimension,
        evidence_by_dimension: EvidenceInputs,
    ):
        """Create evidence for one valuation dimension."""

        _, source, source_type, timestamp = evidence_by_dimension["_source"]
        content, metric_name, metric_value, _ = evidence_by_dimension[dimension]
        return self._make_evidence(
            content=content,
            source=source,
            source_type=source_type,
            timestamp=timestamp,
            metric_name=metric_name,
            metric_value=metric_value,
        )


def _build_llm_prompt(input_data: dict[str, Any], workspace: Workspace) -> str:
    """Build a structured prompt for valuation analysis."""

    metrics = input_data.get("valuation_metrics", {})
    metric_lines = [
        f"- forward_pe: {_value(metrics, 'forward_pe')}",
        f"- sector_forward_pe: {_value(metrics, 'sector_forward_pe')}",
        f"- earnings_growth_yoy: {_value(metrics, 'earnings_growth_yoy')}",
        f"- free_cash_flow_yield: {_value(metrics, 'free_cash_flow_yield')}",
        f"- dividend_yield: {_value(metrics, 'dividend_yield')}",
    ]
    return (
        f"Ticker: {workspace.ticker}\n"
        f"Company: {workspace.company_name or 'unknown'}\n"
        f"Research question: {workspace.research_question}\n"
        f"Data source: {input_data.get('data_source', 'mock')}\n"
        f"As-of date: {input_data.get('as_of_date', _news_timestamp(input_data))}\n\n"
        "Valuation metrics:\n"
        + "\n".join(metric_lines)
        + "\n\n"
        "Generate exactly two valuation insights, one for each dimension: "
        "relative_valuation and growth_adjusted_valuation. Use concise "
        "investment research language. Mention uncertainty when fields are "
        "unknown. Do not introduce facts outside the supplied metrics.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "insights": [\n'
        "    {\n"
        '      "dimension": "relative_valuation|growth_adjusted_valuation",\n'
        '      "claim": "evidence-grounded valuation claim",\n'
        '      "confidence": "low|medium|high",\n'
        '      "risk_tag": "stable_snake_case_tag",\n'
        '      "time_horizon": "6-12 months"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _normalize_llm_risk_tag(risk_tag: str) -> str:
    """Normalize LLM risk tags so merge rules can still classify valuation risk."""

    normalized = risk_tag.lower().strip().replace("-", "_").replace(" ", "_")
    if not normalized:
        return "valuation_analysis"
    if any(token in normalized for token in ("risk", "premium", "expensive", "overvalued")):
        return "valuation_risk"
    if any(token in normalized for token in ("gap", "unknown", "missing", "incomplete")):
        return "valuation_evidence_gap"
    if any(token in normalized for token in ("support", "discount", "attractive", "cheap")):
        return "valuation_support"
    if "cash" in normalized and "risk" in normalized:
        return "cash_yield_risk"
    if "cash" in normalized and "support" in normalized:
        return "cash_yield_support"
    return normalized[:80]


def _relative_pe_claim(forward_pe: str, sector_forward_pe: str) -> str:
    pe = _parse_float(forward_pe)
    sector_pe = _parse_float(sector_forward_pe)
    if pe is None:
        return "由于远期市盈率不可用，估值证据不完整。"
    if sector_pe is not None and pe > sector_pe * 1.15:
        return "股票相对行业估值存在溢价。"
    if pe > INVESTMENT_THRESHOLDS["high_valuation_pe"]:
        return "绝对估值偏高。"
    if pe < INVESTMENT_THRESHOLDS["low_valuation_pe"]:
        return "绝对估值具有吸引力。"
    return "相对同行来看，估值整体较为均衡。"


def _cash_yield_claim(earnings_growth: str, fcf_yield: str) -> str:
    growth = _parse_percent(earnings_growth)
    yield_value = _parse_percent(fcf_yield)
    if growth is None or yield_value is None:
        return "增长调整后的估值证据不完整。"
    if growth < INVESTMENT_THRESHOLDS["weak_growth_percent"] and yield_value < 4:
        return "低增长和一般现金收益率削弱估值支撑。"
    if growth >= INVESTMENT_THRESHOLDS["healthy_growth_percent"] or yield_value >= 5:
        return "增长或现金收益率为估值提供支撑。"
    return "增长调整后的估值较为中性。"


def _relative_pe_risk_tag(forward_pe: str, sector_forward_pe: str) -> str:
    claim = _relative_pe_claim(forward_pe, sector_forward_pe).lower()
    if "premium" in claim or "elevated" in claim or "溢价" in claim or "偏高" in claim:
        return "valuation_risk"
    if "incomplete" in claim or "不完整" in claim:
        return "valuation_evidence_gap"
    return "valuation_support"


def _cash_yield_risk_tag(earnings_growth: str, fcf_yield: str) -> str:
    claim = _cash_yield_claim(earnings_growth, fcf_yield).lower()
    if "weaken" in claim or "削弱" in claim:
        return "cash_yield_risk"
    if "incomplete" in claim or "不完整" in claim:
        return "valuation_evidence_gap"
    return "cash_yield_support"


def _news_timestamp(input_data: dict[str, Any]) -> str:
    news = input_data.get("news", [])
    if news:
        return str(news[0].get("timestamp", "2026-05-14"))
    return "2026-05-14"


def _source_from_input(input_data: dict[str, Any], fallback: str) -> str:
    if input_data.get("data_source") == "yfinance":
        return "Yahoo Finance via yfinance"
    if fallback == "Valuation Model":
        return "估值模型"
    return fallback


def _value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key, "unknown")
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value)


def _confidence(*values: str) -> str:
    if any(value.lower() == "unknown" for value in values):
        return "low"
    return "medium"


def _parse_percent(value: str) -> float | None:
    if value.lower() == "unknown":
        return None
    try:
        return float(value.replace("%", "").strip())
    except ValueError:
        return None


def _parse_float(value: str) -> float | None:
    if value.lower() == "unknown":
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None
