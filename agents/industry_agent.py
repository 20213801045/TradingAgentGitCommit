"""Industry comparison agent."""

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Workspace


class IndustryComparisonAgent(BaseAgent):
    """Creates commits comparing a company with coarse sector benchmarks."""

    name = "IndustryComparisonAgent"
    role = "industry-comparison-agent"
    branch_name = "industry-comparison"

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate industry-relative commits."""

        del workspace
        metrics = input_data.get("industry_comparison", {})
        timestamp = str(input_data.get("as_of_date", _news_timestamp(input_data)))
        sector = _value(metrics, "sector")
        growth = _value(metrics, "revenue_growth_yoy")
        sector_growth = _value(metrics, "sector_revenue_growth_yoy")
        margin = _value(metrics, "net_margin")
        sector_margin = _value(metrics, "sector_net_margin")
        pe = _value(metrics, "forward_pe")
        sector_pe = _value(metrics, "sector_forward_pe")

        operating_evidence = self._make_evidence(
            content=(
                f"所属行业为 {sector}。公司收入增长为 {growth}，行业基准为 {sector_growth}；"
                f"公司净利率为 {margin}，行业基准为 {sector_margin}。"
            ),
            source="行业基准模型",
            source_type="industry_benchmark",
            timestamp=timestamp,
            metric_name="industry_operating_comparison",
            metric_value=f"{growth}/{sector_growth}; {margin}/{sector_margin}",
        )
        valuation_evidence = self._make_evidence(
            content=f"公司远期市盈率为 {pe}，行业基准为 {sector_pe}。",
            source="行业基准模型",
            source_type="industry_benchmark",
            timestamp=timestamp,
            metric_name="industry_valuation_comparison",
            metric_value=f"{pe}/{sector_pe}",
        )

        return [
            self.create_commit(
                claim=_operating_claim(growth, sector_growth, margin, sector_margin),
                evidence=operating_evidence,
                confidence=_confidence(growth, sector_growth, margin, sector_margin),
                risk_tag=_operating_risk_tag(growth, sector_growth, margin, sector_margin),
                time_horizon="12 months",
            ),
            self.create_commit(
                claim=_valuation_claim(pe, sector_pe),
                evidence=valuation_evidence,
                confidence=_confidence(pe, sector_pe),
                risk_tag=_valuation_risk_tag(pe, sector_pe),
                time_horizon="6-12 months",
            ),
        ]


def _operating_claim(
    growth: str,
    sector_growth: str,
    margin: str,
    sector_margin: str,
) -> str:
    growth_value = _parse_percent(growth)
    sector_growth_value = _parse_percent(sector_growth)
    margin_value = _parse_percent(margin)
    sector_margin_value = _parse_percent(sector_margin)
    if None in {growth_value, sector_growth_value, margin_value, sector_margin_value}:
        return "由于缺少行业基准数据，行业比较不完整。"
    if growth_value >= sector_growth_value and margin_value >= sector_margin_value:
        return "公司在收入增长和利润率方面优于行业基准。"
    if growth_value < sector_growth_value and margin_value < sector_margin_value:
        return "公司在收入增长和利润率方面落后于行业基准。"
    return "公司相对行业的增长和利润率表现较为混合。"


def _valuation_claim(pe: str, sector_pe: str) -> str:
    pe_value = _parse_float(pe)
    sector_pe_value = _parse_float(sector_pe)
    if pe_value is None or sector_pe_value is None:
        return "行业估值比较不完整。"
    if pe_value > sector_pe_value * 1.15:
        return "行业估值比较显示公司存在估值溢价。"
    if pe_value < sector_pe_value * 0.85:
        return "行业估值比较显示公司存在估值折价。"
    return "行业估值比较显示公司接近同行基准。"


def _operating_risk_tag(
    growth: str,
    sector_growth: str,
    margin: str,
    sector_margin: str,
) -> str:
    claim = _operating_claim(growth, sector_growth, margin, sector_margin).lower()
    if "favorably" in claim or "优于" in claim:
        return "industry_support"
    if "lags" in claim or "落后" in claim:
        return "industry_risk"
    if "incomplete" in claim or "不完整" in claim:
        return "industry_evidence_gap"
    return "industry_mixed"


def _valuation_risk_tag(pe: str, sector_pe: str) -> str:
    claim = _valuation_claim(pe, sector_pe).lower()
    if "premium" in claim or "溢价" in claim:
        return "valuation_risk"
    if "discount" in claim or "折价" in claim:
        return "valuation_support"
    if "incomplete" in claim or "不完整" in claim:
        return "industry_evidence_gap"
    return "valuation"


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
