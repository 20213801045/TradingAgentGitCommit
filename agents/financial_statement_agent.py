"""Financial statement deep-dive agent."""

from typing import Any

from agents.base_agent import BaseAgent
from config import INVESTMENT_THRESHOLDS
from models.schemas import ClaimEvidenceCommit, Workspace


class FinancialStatementAgent(BaseAgent):
    """Creates commits about financial statement quality."""

    name = "FinancialStatementAgent"
    role = "financial-statement-agent"
    branch_name = "financial-statement-analysis"

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate commits from statement-level metrics."""

        del workspace
        metrics = input_data.get("financial_statements", {})
        timestamp = str(input_data.get("as_of_date", _news_timestamp(input_data)))
        source = _source_from_input(input_data)
        revenue_growth_3y = _value(metrics, "revenue_growth_3y")
        fcf_margin = _value(metrics, "free_cash_flow_margin")
        debt_to_equity = _value(metrics, "debt_to_equity")
        roe = _value(metrics, "return_on_equity")

        cash_generation_evidence = self._make_evidence(
            content=(
                f"三年收入增长为 {revenue_growth_3y}，自由现金流率为 {fcf_margin}，"
                f"ROE 为 {roe}。"
            ),
            source=source,
            source_type="financial_data_provider",
            timestamp=timestamp,
            metric_name="statement_quality",
            metric_value=f"{revenue_growth_3y}/{fcf_margin}/{roe}",
        )
        leverage_evidence = self._make_evidence(
            content=f"资产负债相关的债务权益比为 {debt_to_equity}。",
            source=source,
            source_type="financial_data_provider",
            timestamp=timestamp,
            metric_name="debt_to_equity",
            metric_value=debt_to_equity,
        )

        return [
            self.create_commit(
                claim=_cash_generation_claim(revenue_growth_3y, fcf_margin),
                evidence=cash_generation_evidence,
                confidence=_confidence(revenue_growth_3y, fcf_margin),
                risk_tag=_cash_generation_risk_tag(revenue_growth_3y, fcf_margin),
                time_horizon="12-24 months",
            ),
            self.create_commit(
                claim=_leverage_claim(debt_to_equity),
                evidence=leverage_evidence,
                confidence=_confidence(debt_to_equity),
                risk_tag=_leverage_risk_tag(debt_to_equity),
                time_horizon="12-24 months",
            ),
        ]


def _cash_generation_claim(revenue_growth_3y: str, fcf_margin: str) -> str:
    growth = _parse_percent(revenue_growth_3y)
    margin = _parse_percent(fcf_margin)
    if growth is None or margin is None:
        return "由于增长或现金流数据缺失，财报深度分析受限。"
    if growth >= INVESTMENT_THRESHOLDS["weak_growth_percent"] and margin >= 15:
        return "财务报表显示增长和现金生成能力具有持续性。"
    if margin < 8:
        return "自由现金流转化偏弱，降低财务质量。"
    return "财务报表质量表现混合。"


def _leverage_claim(debt_to_equity: str) -> str:
    leverage = _parse_float(debt_to_equity)
    if leverage is None:
        return "杠杆证据不可用，需要谨慎处理。"
    if leverage > 2.0:
        return "杠杆水平偏高，可能削弱资产负债表灵活性。"
    return "杠杆水平看起来可控。"


def _cash_generation_risk_tag(revenue_growth_3y: str, fcf_margin: str) -> str:
    claim = _cash_generation_claim(revenue_growth_3y, fcf_margin).lower()
    if "durable" in claim or "持续" in claim:
        return "statement_quality"
    if "missing" in claim or "缺失" in claim or "受限" in claim:
        return "statement_evidence_gap"
    return "cash_generation_risk"


def _leverage_risk_tag(debt_to_equity: str) -> str:
    claim = _leverage_claim(debt_to_equity).lower()
    if "elevated" in claim or "偏高" in claim:
        return "leverage_risk"
    if "unavailable" in claim or "不可用" in claim:
        return "leverage_evidence_gap"
    return "leverage"


def _news_timestamp(input_data: dict[str, Any]) -> str:
    news = input_data.get("news", [])
    if news:
        return str(news[0].get("timestamp", "2026-05-14"))
    return "2026-05-14"


def _source_from_input(input_data: dict[str, Any]) -> str:
    if input_data.get("data_source") == "yfinance":
        return "Yahoo Finance via yfinance"
    return "模拟财务报表"


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
