"""Technical analysis agent."""

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from config import INVESTMENT_THRESHOLDS
from llm import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


TechnicalDimension = Literal[
    "trend_structure",
    "momentum",
    "volatility_risk",
    "support_resistance",
]
EvidenceInputs = dict[str, tuple[str, str, str, Any]]
REQUIRED_TECHNICAL_DIMENSIONS: tuple[TechnicalDimension, ...] = (
    "trend_structure",
    "momentum",
    "volatility_risk",
    "support_resistance",
)


class TechnicalInsight(BaseModel):
    """One LLM-generated technical insight."""

    dimension: TechnicalDimension
    claim: str = Field(min_length=8)
    confidence: Literal["low", "medium", "high"] = "medium"
    risk_tag: str = "technical_analysis"
    time_horizon: str = "1-3 months"


class TechnicalAnalysisOutput(BaseModel):
    """Validated LLM output for technical analysis."""

    insights: list[TechnicalInsight] = Field(default_factory=list)


class TechnicalAgent(BaseAgent):
    """Creates commits about price action and technical indicators."""

    name = "TechnicalAgent"
    role = "technical-agent"
    branch_name = "technical-analysis"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate LLM-backed technical commits with deterministic fallback."""

        evidence_by_dimension = self._evidence_by_dimension(input_data)
        llm_commits = self._llm_commits(input_data, workspace, evidence_by_dimension)
        if llm_commits:
            return llm_commits
        return self._deterministic_commits(input_data, evidence_by_dimension)

    def _evidence_by_dimension(self, input_data: dict[str, Any]) -> EvidenceInputs:
        """Build reusable evidence inputs keyed by technical dimension."""

        indicators = input_data.get("technical_indicators", input_data)
        timestamp = _timestamp_from_input(input_data)
        source = _source_from_input(input_data)
        source_type = _source_type_from_input(input_data)
        price_trend = _indicator(indicators, "price_trend")
        rsi = _indicator(indicators, "rsi")
        volatility = _indicator(indicators, "volatility")
        support_level = _indicator(indicators, "support_level")
        resistance_level = _indicator(indicators, "resistance_level")

        return {
            "trend_structure": (
                f"价格趋势为 {price_trend}。",
                "price_trend",
                price_trend,
                price_trend,
            ),
            "momentum": (
                f"相对强弱指标 RSI 为 {rsi}。",
                "rsi",
                rsi,
                rsi,
            ),
            "volatility_risk": (
                f"波动率被描述为 {volatility}。",
                "volatility",
                volatility,
                volatility,
            ),
            "support_resistance": (
                f"支撑位接近 {support_level}，阻力位接近 {resistance_level}。",
                "support_resistance",
                f"{support_level}/{resistance_level}",
                f"{support_level}/{resistance_level}",
            ),
            "_source": ("", source, source_type, timestamp),
        }

    def _deterministic_commits(
        self,
        input_data: dict[str, Any],
        evidence_by_dimension: EvidenceInputs,
    ) -> list[ClaimEvidenceCommit]:
        """Generate deterministic commits from technical indicators."""

        indicators = input_data.get("technical_indicators", input_data)
        price_trend = _indicator(indicators, "price_trend")
        rsi = _indicator(indicators, "rsi")
        volatility = _indicator(indicators, "volatility")
        support_level = _indicator(indicators, "support_level")
        resistance_level = _indicator(indicators, "resistance_level")

        return [
            self.create_commit(
                claim=_trend_claim(price_trend),
                evidence=self._evidence_for_dimension(
                    "trend_structure",
                    evidence_by_dimension,
                ),
                confidence=_confidence_for_indicator(price_trend),
                risk_tag=_trend_risk_tag(price_trend),
                time_horizon="1-3 months",
            ),
            self.create_commit(
                claim=_momentum_claim(rsi),
                evidence=self._evidence_for_dimension(
                    "momentum",
                    evidence_by_dimension,
                ),
                confidence=_confidence_for_indicator(rsi),
                risk_tag=_momentum_risk_tag(rsi),
                time_horizon="1-3 months",
            ),
            self.create_commit(
                claim=_volatility_claim(volatility),
                evidence=self._evidence_for_dimension(
                    "volatility_risk",
                    evidence_by_dimension,
                ),
                confidence=_confidence_for_indicator(volatility),
                risk_tag=_volatility_risk_tag(volatility),
                time_horizon="1-3 months",
            ),
            self.create_commit(
                claim=_support_resistance_claim(support_level, resistance_level),
                evidence=self._evidence_for_dimension(
                    "support_resistance",
                    evidence_by_dimension,
                ),
                confidence=_levels_confidence(support_level, resistance_level),
                risk_tag=_levels_risk_tag(support_level, resistance_level),
                time_horizon="1-3 months",
            ),
        ]

    def _llm_commits(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
        evidence_by_dimension: EvidenceInputs,
    ) -> list[ClaimEvidenceCommit]:
        """Ask an optional LLM for technical insights."""

        if self.llm_client is None:
            return []

        try:
            response = self.llm_client.complete(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a technical-analysis agent for an auditable "
                            "investment workflow. Use only the supplied indicators. "
                            "Do not invent prices or indicators. Return only valid JSON."
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
            validated = TechnicalAnalysisOutput.model_validate(parsed)
        except (LLMError, ValidationError):
            return []

        valid_insights = _validate_technical_insights(validated.insights, input_data)
        if valid_insights is None:
            return []

        commits: list[ClaimEvidenceCommit] = []
        seen_dimensions: set[TechnicalDimension] = set()
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
                    time_horizon=insight.time_horizon.strip() or "1-3 months",
                )
            )

        if not all(
            dimension in seen_dimensions
            for dimension in REQUIRED_TECHNICAL_DIMENSIONS
        ):
            return []
        return commits

    def _evidence_for_dimension(
        self,
        dimension: TechnicalDimension,
        evidence_by_dimension: EvidenceInputs,
    ):
        """Create evidence for one technical dimension."""

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
    """Build a structured prompt for technical analysis."""

    indicators = input_data.get("technical_indicators", input_data)
    indicator_lines = [
        f"- price_trend: {_indicator(indicators, 'price_trend')}",
        f"- rsi: {_indicator(indicators, 'rsi')}",
        f"- volatility: {_indicator(indicators, 'volatility')}",
        f"- support_level: {_indicator(indicators, 'support_level')}",
        f"- resistance_level: {_indicator(indicators, 'resistance_level')}",
    ]
    return (
        f"Ticker: {workspace.ticker}\n"
        f"Company: {workspace.company_name or 'unknown'}\n"
        f"Research question: {workspace.research_question}\n"
        f"Data source: {input_data.get('data_source', 'mock')}\n"
        f"As-of date: {_timestamp_from_input(input_data)}\n\n"
        "Technical indicators:\n"
        + "\n".join(indicator_lines)
        + "\n\n"
        "Generate exactly four technical insights, one for each dimension: "
        "trend_structure, momentum, volatility_risk, support_resistance. "
        "Use concise investment research language. Mention uncertainty when "
        "fields are unknown. Do not introduce prices outside the supplied "
        "support and resistance levels.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "insights": [\n'
        "    {\n"
        '      "dimension": "trend_structure|momentum|volatility_risk|support_resistance",\n'
        '      "claim": "evidence-grounded technical claim",\n'
        '      "confidence": "low|medium|high",\n'
        '      "risk_tag": "stable_snake_case_tag",\n'
        '      "time_horizon": "1-3 months"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _validate_technical_insights(
    insights: list[TechnicalInsight],
    input_data: dict[str, Any],
) -> list[TechnicalInsight] | None:
    """Reject LLM outputs that contradict obvious technical indicators."""

    if len(insights) < len(REQUIRED_TECHNICAL_DIMENSIONS):
        return None

    indicators = input_data.get("technical_indicators", input_data)
    valid_insights: list[TechnicalInsight] = []
    seen_dimensions: set[TechnicalDimension] = set()
    for insight in insights:
        if insight.dimension in seen_dimensions:
            continue
        if insight.dimension not in REQUIRED_TECHNICAL_DIMENSIONS:
            continue
        if not _insight_matches_indicators(insight, indicators):
            return None
        seen_dimensions.add(insight.dimension)
        valid_insights.append(insight)
        if len(valid_insights) == len(REQUIRED_TECHNICAL_DIMENSIONS):
            break

    if not all(
        dimension in seen_dimensions
        for dimension in REQUIRED_TECHNICAL_DIMENSIONS
    ):
        return None
    return valid_insights


def _insight_matches_indicators(
    insight: TechnicalInsight,
    indicators: dict[str, Any],
) -> bool:
    """Check that a claim does not obviously contradict supplied indicators."""

    text = f"{insight.claim} {insight.risk_tag}".lower()
    price_trend = _indicator(indicators, "price_trend").lower()
    rsi = _parse_float(_indicator(indicators, "rsi"))
    volatility = _indicator(indicators, "volatility").lower()
    support_level = _indicator(indicators, "support_level").lower()
    resistance_level = _indicator(indicators, "resistance_level").lower()

    if insight.dimension == "trend_structure":
        if "below" in price_trend and _has_constructive_trend_language(text):
            return False
        if "above" in price_trend and _has_broken_trend_language(text):
            return False
    if insight.dimension == "momentum":
        if rsi is not None and rsi < INVESTMENT_THRESHOLDS["weak_rsi"] and _has_bullish_momentum_language(text):
            return False
        if rsi is not None and rsi >= INVESTMENT_THRESHOLDS["bullish_rsi"] and _has_weak_momentum_language(text):
            return False
    if insight.dimension == "volatility_risk":
        if volatility == "high" and _has_low_volatility_language(text):
            return False
        if volatility == "low" and _has_high_volatility_language(text):
            return False
    if insight.dimension == "support_resistance":
        if "unknown" in {support_level, resistance_level} and _has_precise_level_language(text):
            return False
    return True


def _normalize_llm_risk_tag(
    risk_tag: str,
    dimension: TechnicalDimension,
) -> str:
    """Normalize LLM risk tags so merge rules can still classify them."""

    normalized = risk_tag.lower().strip().replace("-", "_").replace(" ", "_")
    if not normalized:
        return f"{dimension}_technical"
    if "trend" in normalized and "risk" in normalized:
        return "price_trend_risk"
    if "momentum" in normalized and "risk" in normalized:
        return "momentum_risk"
    if "volatility" in normalized and "risk" in normalized:
        return "volatility_risk"
    if "resistance" in normalized and "risk" in normalized:
        return "technical_resistance"
    if "gap" in normalized or "unknown" in normalized or "missing" in normalized:
        return f"{dimension}_evidence_gap"
    if "support" in normalized:
        return "support_resistance" if dimension == "support_resistance" else normalized[:80]
    return normalized[:80]


def _has_constructive_trend_language(text: str) -> bool:
    return any(marker in text for marker in ("constructive", "bullish trend", "uptrend", "above", "建设性", "上行"))


def _has_broken_trend_language(text: str) -> bool:
    return any(marker in text for marker in ("broken", "below", "downtrend", "weak trend", "跌破", "下行"))


def _has_bullish_momentum_language(text: str) -> bool:
    return any(marker in text for marker in ("bullish", "upward momentum", "strong momentum", "positive momentum", "上行动量", "强动量"))


def _has_weak_momentum_language(text: str) -> bool:
    return any(marker in text for marker in ("weak momentum", "negative momentum", "bearish momentum", "动量偏弱", "弱动量"))


def _has_low_volatility_language(text: str) -> bool:
    return any(marker in text for marker in ("low volatility", "stable volatility", "volatility is low", "波动率较低", "波动稳定"))


def _has_high_volatility_language(text: str) -> bool:
    return any(marker in text for marker in ("high volatility", "volatile", "elevated volatility", "高波动", "波动率升高"))


def _has_precise_level_language(text: str) -> bool:
    return any(marker in text for marker in ("defined", "clear support", "clear resistance", "precise", "明确", "清晰"))


def _timestamp_from_input(input_data: dict[str, Any]) -> str:
    """Return a timestamp from provider input data."""

    if input_data.get("as_of_date"):
        return str(input_data["as_of_date"])
    news = input_data.get("news", [])
    if news:
        return str(news[0].get("timestamp", "2026-05-14"))
    return "2026-05-14"


def _source_from_input(input_data: dict[str, Any]) -> str:
    """Return provider-specific technical evidence source text."""

    if input_data.get("data_source") == "yfinance":
        return "Yahoo Finance 价格历史 via yfinance"
    return "模拟技术指标数据"


def _source_type_from_input(input_data: dict[str, Any]) -> str:
    """Return provider-specific technical evidence source type."""

    if input_data.get("data_source") == "yfinance":
        return "technical_indicator"
    return "mock_technical_indicator"


def _indicator(indicators: dict[str, Any], key: str) -> str:
    """Return a normalized technical indicator value."""

    value = indicators.get(key, "unknown")
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value)


def _trend_claim(value: str) -> str:
    """Create a price trend claim from trend text."""

    lower_value = value.lower()
    if lower_value == "unknown":
        return "价格趋势证据不可用，需要谨慎处理。"
    if "above" in lower_value:
        return "股价处于建设性的价格趋势中。"
    if "below" in lower_value:
        return "股价低于关键移动均线。"
    return "股价趋势较为混合。"


def _momentum_claim(value: str) -> str:
    """Create a momentum claim from RSI."""

    rsi = _parse_float(value)
    if rsi is None:
        return "动量证据不可用，需要谨慎处理。"
    if rsi >= INVESTMENT_THRESHOLDS["bullish_rsi"]:
        return "股票显示短期上行动量。"
    if rsi < INVESTMENT_THRESHOLDS["weak_rsi"]:
        return "短期动量偏弱。"
    return "短期动量中性。"


def _volatility_claim(value: str) -> str:
    """Create a volatility claim from a volatility label."""

    lower_value = value.lower()
    if lower_value == "unknown":
        return "波动率证据不可用，需要谨慎处理。"
    if lower_value == "high":
        return "高波动仍是关键技术面风险。"
    if lower_value == "low":
        return "波动率相对较低。"
    return "波动率仍是关键技术面风险。"


def _support_resistance_claim(support_level: str, resistance_level: str) -> str:
    """Create a support/resistance claim with missing-value tolerance."""

    if "unknown" in {support_level.lower(), resistance_level.lower()}:
        return "支撑位和阻力位证据不完整。"
    return "明确的支撑位和阻力位界定了短期风险回报。"


def _trend_risk_tag(value: str) -> str:
    """Return a risk tag for price trend evidence."""

    lower_value = value.lower()
    if lower_value == "unknown":
        return "technical_evidence_gap"
    if "below" in lower_value:
        return "price_trend_risk"
    return "price_trend"


def _momentum_risk_tag(value: str) -> str:
    """Return a risk tag for RSI evidence."""

    rsi = _parse_float(value)
    if rsi is None:
        return "momentum_evidence_gap"
    if rsi < INVESTMENT_THRESHOLDS["weak_rsi"]:
        return "momentum_risk"
    return "momentum"


def _volatility_risk_tag(value: str) -> str:
    """Return a risk tag for volatility evidence."""

    lower_value = value.lower()
    if lower_value == "unknown":
        return "volatility_evidence_gap"
    if lower_value == "high":
        return "volatility_risk"
    return "volatility"


def _levels_risk_tag(support_level: str, resistance_level: str) -> str:
    """Return a risk tag for support/resistance evidence."""

    if "unknown" in {support_level.lower(), resistance_level.lower()}:
        return "support_resistance_evidence_gap"
    return "support_resistance"


def _confidence_for_indicator(value: str) -> str:
    """Use low confidence for unknown technical fields."""

    if value.lower() == "unknown":
        return "low"
    return "medium"


def _levels_confidence(support_level: str, resistance_level: str) -> str:
    """Use low confidence when either level is unknown."""

    if "unknown" in {support_level.lower(), resistance_level.lower()}:
        return "low"
    return "medium"


def _parse_float(value: str) -> float | None:
    """Parse a numeric indicator string into a float."""

    if value.lower() == "unknown":
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None
