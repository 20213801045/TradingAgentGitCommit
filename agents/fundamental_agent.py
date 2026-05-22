"""Fundamental analysis agent."""

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from config import INVESTMENT_THRESHOLDS
from llm import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


FundamentalDimension = Literal[
    "growth_quality",
    "margin_quality",
    "cash_generation",
    "balance_sheet",
    "capital_allocation",
    "fundamental_valuation_risk",
]
EvidenceInputs = dict[str, tuple[str, str, str, Any]]
REQUIRED_FUNDAMENTAL_DIMENSIONS: tuple[FundamentalDimension, ...] = (
    "growth_quality",
    "margin_quality",
    "cash_generation",
    "balance_sheet",
    "capital_allocation",
    "fundamental_valuation_risk",
)


class FundamentalInsight(BaseModel):
    """One LLM-generated fundamental insight."""

    dimension: FundamentalDimension
    claim: str = Field(min_length=8)
    confidence: Literal["low", "medium", "high"] = "medium"
    risk_tag: str = "fundamental_analysis"
    time_horizon: str = "12 months"


class FundamentalAnalysisOutput(BaseModel):
    """Validated LLM output for fundamental analysis."""

    insights: list[FundamentalInsight] = Field(default_factory=list)


class FundamentalAgent(BaseAgent):
    """Creates commits about company fundamentals."""

    name = "FundamentalAgent"
    role = "fundamental-agent"
    branch_name = "fundamental-analysis"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate LLM-backed fundamental commits with deterministic fallback."""

        evidence_by_dimension = self._evidence_by_dimension(input_data)
        llm_commits = self._llm_commits(input_data, workspace, evidence_by_dimension)
        if llm_commits:
            return llm_commits
        return self._deterministic_commits(input_data, evidence_by_dimension)

    def _evidence_by_dimension(
        self,
        input_data: dict[str, Any],
    ) -> EvidenceInputs:
        """Build reusable evidence inputs keyed by fundamental dimension."""

        metrics = input_data.get("financial_metrics", input_data)
        timestamp = _timestamp_from_input(input_data)
        source = _source_from_input(input_data, "financial")
        source_type = _source_type_from_input(input_data, "financial")
        revenue_growth = _metric(metrics, "revenue_growth_yoy")
        gross_margin = _metric(metrics, "gross_margin")
        net_margin = _metric(metrics, "net_margin")
        forward_pe = _metric(metrics, "forward_pe")
        cash_position = _metric(metrics, "cash_position")
        debt_to_equity = _metric(metrics, "debt_to_equity")
        free_cash_flow_margin = _metric(metrics, "free_cash_flow_margin")
        return_on_equity = _metric(metrics, "return_on_equity")
        earnings_growth = _metric(metrics, "earnings_growth_yoy")
        free_cash_flow_yield = _metric(metrics, "free_cash_flow_yield")
        dividend_yield = _metric(metrics, "dividend_yield")
        market_cap = _metric(metrics, "market_cap")
        sector = _metric(metrics, "sector")

        return {
            "growth_quality": (
                f"同比收入增长为 {revenue_growth}。",
                "revenue_growth_yoy",
                revenue_growth,
                revenue_growth,
            ),
            "margin_quality": (
                (
                    f"毛利率为 {gross_margin}，净利率为 {net_margin}，"
                    f"ROE 为 {return_on_equity}。"
                ),
                "margin_quality",
                f"{gross_margin}/{net_margin}/{return_on_equity}",
                net_margin,
            ),
            "cash_generation": (
                (
                    f"自由现金流率为 {free_cash_flow_margin}，收入增长为 "
                    f"{revenue_growth}，ROE 为 {return_on_equity}。"
                ),
                "cash_generation",
                f"{free_cash_flow_margin}/{revenue_growth}/{return_on_equity}",
                free_cash_flow_margin,
            ),
            "balance_sheet": (
                f"现金状况被描述为 {cash_position}，债务权益比为 {debt_to_equity}。",
                "balance_sheet_health",
                f"{cash_position}/{debt_to_equity}",
                cash_position,
            ),
            "capital_allocation": (
                (
                    f"现金状况为 {cash_position}，自由现金流率为 "
                    f"{free_cash_flow_margin}，股息率为 {dividend_yield}。"
                ),
                "capital_allocation_capacity",
                f"{cash_position}/{free_cash_flow_margin}/{dividend_yield}",
                free_cash_flow_margin,
            ),
            "fundamental_valuation_risk": (
                (
                    f"远期市盈率为 {forward_pe}，盈利增长为 {earnings_growth}，"
                    f"自由现金流收益率为 {free_cash_flow_yield}，市值为 "
                    f"{market_cap}，行业为 {sector}。"
                ),
                "fundamental_valuation_context",
                f"{forward_pe}/{earnings_growth}/{free_cash_flow_yield}",
                forward_pe,
            ),
        } | {
            "_source": ("", source, source_type, timestamp),
        }

    def _deterministic_commits(
        self,
        input_data: dict[str, Any],
        evidence_by_dimension: EvidenceInputs,
    ) -> list[ClaimEvidenceCommit]:
        """Generate deterministic commits from financial metrics."""

        metrics = input_data.get("financial_metrics", input_data)
        revenue_growth = _metric(metrics, "revenue_growth_yoy")
        gross_margin = _metric(metrics, "gross_margin")
        net_margin = _metric(metrics, "net_margin")
        forward_pe = _metric(metrics, "forward_pe")
        cash_position = _metric(metrics, "cash_position")
        debt_to_equity = _metric(metrics, "debt_to_equity")
        free_cash_flow_margin = _metric(metrics, "free_cash_flow_margin")
        return_on_equity = _metric(metrics, "return_on_equity")

        return [
            self.create_commit(
                claim=_growth_claim(revenue_growth),
                evidence=self._evidence_for_dimension(
                    "growth_quality",
                    evidence_by_dimension,
                ),
                confidence=_confidence_for_metric(revenue_growth),
                risk_tag=_growth_risk_tag(revenue_growth),
                time_horizon="12 months",
            ),
            self.create_commit(
                claim=_margin_quality_claim(gross_margin, net_margin, return_on_equity),
                evidence=self._evidence_for_dimension(
                    "margin_quality",
                    evidence_by_dimension,
                ),
                confidence=_confidence_for_metric(net_margin),
                risk_tag=_margin_quality_risk_tag(net_margin),
                time_horizon="12-24 months",
            ),
            self.create_commit(
                claim=_cash_generation_claim(free_cash_flow_margin),
                evidence=self._evidence_for_dimension(
                    "cash_generation",
                    evidence_by_dimension,
                ),
                confidence=_confidence_for_metric(free_cash_flow_margin),
                risk_tag=_cash_generation_risk_tag(free_cash_flow_margin),
                time_horizon="12-24 months",
            ),
            self.create_commit(
                claim=_balance_sheet_claim(cash_position, debt_to_equity),
                evidence=self._evidence_for_dimension(
                    "balance_sheet",
                    evidence_by_dimension,
                ),
                confidence=_confidence(cash_position, debt_to_equity),
                risk_tag=_balance_sheet_risk_tag(cash_position, debt_to_equity),
                time_horizon="12-24 months",
            ),
            self.create_commit(
                claim=_capital_allocation_claim(cash_position, free_cash_flow_margin),
                evidence=self._evidence_for_dimension(
                    "capital_allocation",
                    evidence_by_dimension,
                ),
                confidence=_confidence(cash_position, free_cash_flow_margin),
                risk_tag=_capital_allocation_risk_tag(
                    cash_position,
                    free_cash_flow_margin,
                ),
                time_horizon="12-24 months",
            ),
            self.create_commit(
                claim=_valuation_claim(forward_pe),
                evidence=self._evidence_for_dimension(
                    "fundamental_valuation_risk",
                    evidence_by_dimension,
                ),
                confidence=_confidence_for_metric(forward_pe),
                risk_tag=_valuation_risk_tag(forward_pe),
                time_horizon="6-12 months",
            ),
        ]

    def _llm_commits(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
        evidence_by_dimension: EvidenceInputs,
    ) -> list[ClaimEvidenceCommit]:
        """Ask an optional LLM for fundamental insights."""

        if self.llm_client is None:
            return []

        try:
            response = self.llm_client.complete(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a fundamental equity research agent. "
                            "Use only the supplied metrics. Do not invent facts. "
                            "Return only valid JSON."
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
            validated = FundamentalAnalysisOutput.model_validate(parsed)
        except (LLMError, ValidationError):
            return []

        valid_insights = _validate_fundamental_insights(
            validated.insights,
            input_data,
        )
        if valid_insights is None:
            return []

        commits: list[ClaimEvidenceCommit] = []
        seen_dimensions: set[FundamentalDimension] = set()
        for insight in valid_insights:
            if insight.dimension in seen_dimensions:
                continue
            seen_dimensions.add(insight.dimension)
            commits.append(
                self.create_commit(
                    claim=insight.claim.strip(),
                    evidence=self._evidence_for_dimension(
                        insight.dimension,
                        evidence_by_dimension,
                    ),
                    confidence=insight.confidence,
                    risk_tag=_normalize_llm_risk_tag(insight.risk_tag, insight.dimension),
                    time_horizon=(
                        insight.time_horizon.strip()
                        or _default_time_horizon(insight.dimension)
                    ),
                )
            )

        if not all(
            dimension in seen_dimensions
            for dimension in REQUIRED_FUNDAMENTAL_DIMENSIONS
        ):
            return []
        return commits

    def _evidence_for_dimension(
        self,
        dimension: FundamentalDimension,
        evidence_by_dimension: EvidenceInputs,
    ):
        """Create evidence for one fundamental dimension."""

        source_data = evidence_by_dimension["_source"]
        _, source, source_type, timestamp = source_data
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
    """Build a structured prompt for fundamental analysis."""

    metrics = input_data.get("financial_metrics", input_data)
    metric_lines = [
        f"- revenue_growth_yoy: {_metric(metrics, 'revenue_growth_yoy')}",
        f"- gross_margin: {_metric(metrics, 'gross_margin')}",
        f"- net_margin: {_metric(metrics, 'net_margin')}",
        f"- forward_pe: {_metric(metrics, 'forward_pe')}",
        f"- cash_position: {_metric(metrics, 'cash_position')}",
        f"- debt_to_equity: {_metric(metrics, 'debt_to_equity')}",
        f"- free_cash_flow_margin: {_metric(metrics, 'free_cash_flow_margin')}",
        f"- return_on_equity: {_metric(metrics, 'return_on_equity')}",
        f"- earnings_growth_yoy: {_metric(metrics, 'earnings_growth_yoy')}",
        f"- free_cash_flow_yield: {_metric(metrics, 'free_cash_flow_yield')}",
        f"- dividend_yield: {_metric(metrics, 'dividend_yield')}",
        f"- market_cap: {_metric(metrics, 'market_cap')}",
        f"- sector: {_metric(metrics, 'sector')}",
    ]
    return (
        f"Ticker: {workspace.ticker}\n"
        f"Company: {workspace.company_name or 'unknown'}\n"
        f"Research question: {workspace.research_question}\n"
        f"Data source: {input_data.get('data_source', 'mock')}\n"
        f"As-of date: {_timestamp_from_input(input_data)}\n\n"
        "Fundamental metrics:\n"
        + "\n".join(metric_lines)
        + "\n\n"
        "Generate exactly six fundamental insights, one for each dimension: "
        "growth_quality, margin_quality, cash_generation, balance_sheet, "
        "capital_allocation, fundamental_valuation_risk. Use concise "
        "investment research language. Mention uncertainty when fields are "
        "unknown. Do not introduce facts outside the supplied metrics.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "insights": [\n'
        "    {\n"
        '      "dimension": "growth_quality|margin_quality|cash_generation|'
        'balance_sheet|capital_allocation|fundamental_valuation_risk",\n'
        '      "claim": "evidence-grounded claim",\n'
        '      "confidence": "low|medium|high",\n'
        '      "risk_tag": "stable_snake_case_tag",\n'
        '      "time_horizon": "12 months"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _normalize_llm_risk_tag(
    risk_tag: str,
    dimension: FundamentalDimension,
) -> str:
    """Normalize LLM risk tags so merge rules can still classify them."""

    normalized = risk_tag.lower().strip().replace("-", "_").replace(" ", "_")
    if not normalized:
        return f"{dimension}_llm"
    if "valuation" in normalized and "risk" in normalized:
        return "valuation_risk"
    if "growth" in normalized and "risk" in normalized:
        return "growth_risk"
    if ("profit" in normalized or "margin" in normalized) and "risk" in normalized:
        return "profitability_risk"
    if ("cash" in normalized or "generation" in normalized) and "risk" in normalized:
        return "cash_generation_risk"
    if ("balance" in normalized or "leverage" in normalized) and "risk" in normalized:
        return "balance_sheet_risk"
    if "allocation" in normalized and "risk" in normalized:
        return "capital_allocation_risk"
    if "gap" in normalized or "unknown" in normalized or "missing" in normalized:
        return f"{dimension}_evidence_gap"
    if "support" in normalized or "quality" in normalized or "strength" in normalized:
        return f"{dimension}_support"
    return normalized[:80]


def _validate_fundamental_insights(
    insights: list[FundamentalInsight],
    input_data: dict[str, Any],
) -> list[FundamentalInsight] | None:
    """Reject LLM outputs that contradict obvious metric direction."""

    if len(insights) < len(REQUIRED_FUNDAMENTAL_DIMENSIONS):
        return None

    metrics = input_data.get("financial_metrics", input_data)
    valid_insights: list[FundamentalInsight] = []
    seen_dimensions: set[FundamentalDimension] = set()
    for insight in insights:
        if insight.dimension in seen_dimensions:
            continue
        if insight.dimension not in REQUIRED_FUNDAMENTAL_DIMENSIONS:
            continue
        if not _insight_matches_metrics(insight, metrics):
            return None
        seen_dimensions.add(insight.dimension)
        valid_insights.append(insight)
        if len(valid_insights) == len(REQUIRED_FUNDAMENTAL_DIMENSIONS):
            break

    if not all(
        dimension in seen_dimensions
        for dimension in REQUIRED_FUNDAMENTAL_DIMENSIONS
    ):
        return None
    return valid_insights


def _insight_matches_metrics(
    insight: FundamentalInsight,
    metrics: dict[str, Any],
) -> bool:
    """Check that a claim does not obviously contradict supplied metrics."""

    text = f"{insight.claim} {insight.risk_tag}".lower()
    revenue_growth = _parse_percent(_metric(metrics, "revenue_growth_yoy"))
    net_margin = _parse_percent(_metric(metrics, "net_margin"))
    fcf_margin = _parse_percent(_metric(metrics, "free_cash_flow_margin"))
    forward_pe = _parse_float(_metric(metrics, "forward_pe"))
    debt_to_equity = _parse_float(_metric(metrics, "debt_to_equity"))

    if insight.dimension == "growth_quality":
        if revenue_growth is not None and revenue_growth < 0 and _has_positive_language(text):
            return False
    if insight.dimension == "margin_quality":
        if (
            net_margin is not None
            and net_margin < INVESTMENT_THRESHOLDS["weak_net_margin_percent"]
            and _has_positive_language(text)
        ):
            return False
    if insight.dimension == "cash_generation":
        if fcf_margin is not None and fcf_margin < 8 and _has_positive_language(text):
            return False
    if insight.dimension == "balance_sheet":
        if debt_to_equity is not None and debt_to_equity > 2.0 and _has_positive_language(text):
            return False
    if insight.dimension == "fundamental_valuation_risk":
        if (
            forward_pe is not None
            and forward_pe > INVESTMENT_THRESHOLDS["high_valuation_pe"]
            and _has_attractive_valuation_language(text)
        ):
            return False
    return True


def _has_positive_language(text: str) -> bool:
    """Return whether text uses clearly positive fundamental language."""

    return any(
        marker in text
        for marker in (
            "strong",
            "stable",
            "positive",
            "healthy",
            "support",
            "durable",
            "high quality",
            "attractive",
            "强",
            "稳定",
            "健康",
            "支持",
            "优质",
        )
    )


def _has_attractive_valuation_language(text: str) -> bool:
    """Return whether text describes valuation as attractive or cheap."""

    return any(
        marker in text
        for marker in (
            "cheap",
            "attractive valuation",
            "undervalued",
            "discount",
            "low valuation",
            "估值有吸引力",
            "低估",
            "折价",
        )
    )


def _default_time_horizon(dimension: FundamentalDimension) -> str:
    """Return a default time horizon by dimension."""

    if dimension == "fundamental_valuation_risk":
        return "6-12 months"
    return "12-24 months"


def _timestamp_from_input(input_data: dict[str, Any]) -> str:
    """Return a timestamp from provider input data."""

    if input_data.get("as_of_date"):
        return str(input_data["as_of_date"])
    news = input_data.get("news", [])
    if news:
        return str(news[0].get("timestamp", "2026-05-14"))
    return "2026-05-14"


def _source_from_input(input_data: dict[str, Any], data_kind: str) -> str:
    """Return provider-specific evidence source text."""

    if input_data.get("data_source") == "yfinance":
        return "Yahoo Finance via yfinance"
    if data_kind == "financial":
        return "模拟财务指标"
    return "模拟公司基本面"


def _source_type_from_input(input_data: dict[str, Any], data_kind: str) -> str:
    """Return provider-specific evidence source type."""

    del data_kind
    if input_data.get("data_source") == "yfinance":
        return "financial_data_provider"
    return "mock_financial_metric"


def _metric(metrics: dict[str, Any], key: str) -> str:
    """Return a normalized metric string."""

    value = metrics.get(key, "unknown")
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value)


def _growth_claim(value: str) -> str:
    """Create a growth claim that tolerates missing or weak data."""

    growth = _parse_percent(value)
    if growth is None:
        return "收入增长证据不可用，需要谨慎处理。"
    if growth < 0:
        return "收入增长偏弱或为负。"
    return "公司收入呈现正增长。"


def _margin_quality_claim(
    gross_margin: str,
    net_margin: str,
    return_on_equity: str,
) -> str:
    """Create a margin-quality claim that tolerates missing data."""

    del gross_margin
    del return_on_equity
    margin = _parse_percent(net_margin)
    if margin is None:
        return "盈利能力和利润率证据不完整，需要谨慎处理。"
    if margin < INVESTMENT_THRESHOLDS["weak_net_margin_percent"]:
        return "利润率质量可能承压。"
    return "盈利能力和利润率质量保持稳定。"


def _cash_generation_claim(free_cash_flow_margin: str) -> str:
    """Create a cash-generation claim from free-cash-flow margin."""

    margin = _parse_percent(free_cash_flow_margin)
    if margin is None:
        return "现金生成证据不完整，需要谨慎处理。"
    if margin < 8:
        return "自由现金流转化偏弱，现金生成质量需要审查。"
    if margin >= 15:
        return "自由现金流转化支持现金生成质量。"
    return "现金生成质量表现中性。"


def _balance_sheet_claim(cash_position: str, debt_to_equity: str) -> str:
    """Create a balance-sheet claim from cash and leverage metrics."""

    leverage = _parse_float(debt_to_equity)
    if cash_position.lower() == "unknown" and leverage is None:
        return "资产负债表证据不完整，需要谨慎处理。"
    if leverage is not None and leverage > 2.0:
        return "杠杆水平偏高，可能削弱资产负债表灵活性。"
    if cash_position.lower() == "strong":
        return "现金状况和杠杆水平支持资产负债表韧性。"
    return "资产负债表质量需要进一步审查。"


def _capital_allocation_claim(
    cash_position: str,
    free_cash_flow_margin: str,
) -> str:
    """Create a capital-allocation capacity claim."""

    fcf_margin = _parse_percent(free_cash_flow_margin)
    if cash_position.lower() == "unknown" or fcf_margin is None:
        return "资本配置能力证据不完整，需要谨慎处理。"
    if cash_position.lower() == "strong" and fcf_margin >= 15:
        return "现金和自由现金流为资本配置提供支持。"
    if fcf_margin < 8:
        return "自由现金流转化偏弱，可能限制资本配置弹性。"
    return "资本配置能力表现中性。"


def _profitability_claim(value: str) -> str:
    """Create a profitability claim that tolerates missing data."""

    margin = _parse_percent(value)
    if margin is None:
        return "盈利能力证据不完整，需要谨慎处理。"
    if margin < INVESTMENT_THRESHOLDS["weak_net_margin_percent"]:
        return "盈利能力可能承压。"
    return "盈利能力保持稳定。"


def _valuation_claim(value: str) -> str:
    """Create a valuation claim from forward P/E."""

    forward_pe = _parse_float(value)
    if forward_pe is None:
        return "估值证据不可用，需要谨慎处理。"
    if forward_pe > INVESTMENT_THRESHOLDS["high_valuation_pe"]:
        return "估值可能相对偏贵。"
    if forward_pe < INVESTMENT_THRESHOLDS["low_valuation_pe"]:
        return "估值可能相对有吸引力。"
    return "估值整体较为均衡。"


def _cash_claim(value: str) -> str:
    """Create a financial health claim from cash-position label."""

    if value.lower() == "unknown":
        return "由于现金状况未知，财务健康证据不完整。"
    if value.lower() == "strong":
        return "基于现金状况，财务健康度较强。"
    return "基于现金状况，财务健康度需要进一步审查。"


def _growth_risk_tag(value: str) -> str:
    """Return a risk tag for growth evidence."""

    growth = _parse_percent(value)
    if growth is None:
        return "growth_evidence_gap"
    if growth < 0:
        return "growth_risk"
    return "growth_quality"


def _margin_quality_risk_tag(value: str) -> str:
    """Return a risk tag for margin-quality evidence."""

    margin = _parse_percent(value)
    if margin is None:
        return "margin_quality_evidence_gap"
    if margin < INVESTMENT_THRESHOLDS["weak_net_margin_percent"]:
        return "profitability_risk"
    return "profitability"


def _cash_generation_risk_tag(value: str) -> str:
    """Return a risk tag for cash-generation evidence."""

    margin = _parse_percent(value)
    if margin is None:
        return "cash_generation_evidence_gap"
    if margin < 8:
        return "cash_generation_risk"
    return "cash_generation_support"


def _balance_sheet_risk_tag(cash_position: str, debt_to_equity: str) -> str:
    """Return a risk tag for balance-sheet evidence."""

    leverage = _parse_float(debt_to_equity)
    if cash_position.lower() == "unknown" and leverage is None:
        return "balance_sheet_evidence_gap"
    if leverage is not None and leverage > 2.0:
        return "balance_sheet_risk"
    if cash_position.lower() == "strong":
        return "balance_sheet_strength"
    return "balance_sheet_review"


def _capital_allocation_risk_tag(
    cash_position: str,
    free_cash_flow_margin: str,
) -> str:
    """Return a risk tag for capital-allocation evidence."""

    margin = _parse_percent(free_cash_flow_margin)
    if cash_position.lower() == "unknown" or margin is None:
        return "capital_allocation_evidence_gap"
    if margin < 8:
        return "capital_allocation_risk"
    if cash_position.lower() == "strong" and margin >= 15:
        return "capital_allocation_support"
    return "capital_allocation"


def _profitability_risk_tag(value: str) -> str:
    """Return a risk tag for profitability evidence."""

    margin = _parse_percent(value)
    if margin is None:
        return "profitability_evidence_gap"
    if margin < INVESTMENT_THRESHOLDS["weak_net_margin_percent"]:
        return "profitability_risk"
    return "profitability"


def _valuation_risk_tag(value: str) -> str:
    """Return a risk tag for valuation evidence."""

    forward_pe = _parse_float(value)
    if forward_pe is None:
        return "valuation_evidence_gap"
    if forward_pe > INVESTMENT_THRESHOLDS["high_valuation_pe"]:
        return "valuation_risk"
    return "valuation"


def _cash_risk_tag(value: str) -> str:
    """Return a risk tag for financial health evidence."""

    if value.lower() == "unknown":
        return "balance_sheet_evidence_gap"
    if value.lower() == "strong":
        return "balance_sheet_strength"
    return "balance_sheet_risk"


def _confidence_for_metric(value: str) -> str:
    """Use low confidence for unknown provider fields."""

    if value.lower() == "unknown":
        return "low"
    return "medium"


def _confidence(*values: str) -> str:
    """Use low confidence if any required field is unknown."""

    if any(value.lower() == "unknown" for value in values):
        return "low"
    return "medium"


def _parse_percent(value: str) -> float | None:
    """Parse a percent string such as `8.2%` into a float."""

    if value.lower() == "unknown":
        return None
    try:
        return float(value.replace("%", "").strip())
    except ValueError:
        return None


def _parse_float(value: str) -> float | None:
    """Parse a numeric string into a float."""

    if value.lower() == "unknown":
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None
