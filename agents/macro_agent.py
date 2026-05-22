"""Macro context agent."""

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Workspace


class MacroAgent(BaseAgent):
    """Creates commits about macro conditions around the company."""

    name = "MacroAgent"
    role = "macro-agent"
    branch_name = "macro-analysis"

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate macro context commits."""

        del workspace
        metrics = input_data.get("macro_context", {})
        timestamp = str(input_data.get("as_of_date", _news_timestamp(input_data)))
        rate_environment = _value(metrics, "rate_environment")
        demand = _value(metrics, "consumer_demand")
        inflation = _value(metrics, "inflation_pressure")
        usd_trend = _value(metrics, "usd_trend")

        demand_evidence = self._make_evidence(
            content=(
                f"利率环境为 {rate_environment}，消费需求为 {demand}，"
                f"通胀压力为 {inflation}。"
            ),
            source="宏观环境模型",
            source_type="macro_data",
            timestamp=timestamp,
            metric_name="macro_demand_conditions",
            metric_value=f"{rate_environment}/{demand}/{inflation}",
        )
        currency_evidence = self._make_evidence(
            content=f"美元趋势为 {usd_trend}。",
            source="宏观环境模型",
            source_type="macro_data",
            timestamp=timestamp,
            metric_name="usd_trend",
            metric_value=usd_trend,
        )

        return [
            self.create_commit(
                claim=_macro_demand_claim(rate_environment, demand, inflation),
                evidence=demand_evidence,
                confidence=_confidence(rate_environment, demand, inflation),
                risk_tag=_macro_demand_risk_tag(rate_environment, demand, inflation),
                time_horizon="6-12 months",
            ),
            self.create_commit(
                claim=_currency_claim(usd_trend),
                evidence=currency_evidence,
                confidence=_confidence(usd_trend),
                risk_tag=_currency_risk_tag(usd_trend),
                time_horizon="6-12 months",
            ),
        ]


def _macro_demand_claim(rate_environment: str, demand: str, inflation: str) -> str:
    combined = f"{rate_environment} {demand} {inflation}".lower()
    if "unknown" in combined:
        return "宏观背景不完整，应限制结论置信度。"
    if "elevated" in combined or "weak" in combined:
        return "宏观条件带来需求或估值压力。"
    if "stable" in combined and "moderate" in combined:
        return "宏观条件看起来可控。"
    return "宏观条件表现混合。"


def _currency_claim(usd_trend: str) -> str:
    value = usd_trend.lower()
    if value == "unknown":
        return "汇率背景不可用。"
    if "strong" in value:
        return "美元走强可能压制海外收入折算。"
    return "汇率背景没有实质改变投资判断。"


def _macro_demand_risk_tag(rate_environment: str, demand: str, inflation: str) -> str:
    claim = _macro_demand_claim(rate_environment, demand, inflation).lower()
    if "pressure" in claim or "压力" in claim:
        return "macro_risk"
    if "incomplete" in claim or "不完整" in claim:
        return "macro_evidence_gap"
    return "macro_context"


def _currency_risk_tag(usd_trend: str) -> str:
    claim = _currency_claim(usd_trend).lower()
    if "pressure" in claim or "压制" in claim:
        return "currency_risk"
    if "unavailable" in claim or "不可用" in claim:
        return "macro_evidence_gap"
    return "currency_context"


def _news_timestamp(input_data: dict[str, Any]) -> str:
    news = input_data.get("news", [])
    if news:
        return str(news[0].get("timestamp", "2026-05-14"))
    return "2026-05-14"


def _value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key, "unknown")
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value)


def _confidence(*values: str) -> str:
    if any(value.lower() == "unknown" for value in values):
        return "low"
    return "medium"
