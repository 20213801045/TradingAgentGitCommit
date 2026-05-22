"""Portfolio construction agent."""

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Workspace


class PortfolioAgent(BaseAgent):
    """Creates commits about position sizing and portfolio fit."""

    name = "PortfolioAgent"
    role = "portfolio-agent"
    branch_name = "portfolio-review"

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate portfolio construction commits."""

        del workspace
        metrics = input_data.get("portfolio_context", {})
        timestamp = str(input_data.get("as_of_date", _news_timestamp(input_data)))
        position_size = _value(metrics, "position_size")
        max_position_size = _value(metrics, "max_position_size")
        correlation = _value(metrics, "correlation_to_market")
        liquidity = _value(metrics, "liquidity")
        role = _value(metrics, "portfolio_role")

        sizing_evidence = self._make_evidence(
            content=(
                f"当前仓位为 {position_size}，最大仓位为 {max_position_size}，"
                f"与市场相关性为 {correlation}，流动性为 {liquidity}，"
                f"组合角色为 {role}。"
            ),
            source="组合构建模型",
            source_type="portfolio_model",
            timestamp=timestamp,
            metric_name="portfolio_fit",
            metric_value=f"{position_size}/{max_position_size}/{correlation}",
        )

        return [
            self.create_commit(
                claim=_portfolio_claim(position_size, max_position_size, correlation, liquidity),
                evidence=sizing_evidence,
                confidence=_confidence(position_size, max_position_size, correlation, liquidity),
                risk_tag=_portfolio_risk_tag(position_size, max_position_size, correlation, liquidity),
                time_horizon="ongoing",
            )
        ]


def _portfolio_claim(
    position_size: str,
    max_position_size: str,
    correlation: str,
    liquidity: str,
) -> str:
    size = _parse_percent(position_size)
    max_size = _parse_percent(max_position_size)
    corr = _parse_float(correlation)
    if "watchlist" in position_size.lower():
        return "在仓位规则明确前，该标的仅适合作为观察名单对象。"
    if size is None or max_size is None or corr is None:
        return "由于缺少仓位或相关性数据，组合适配性不完整。"
    if size <= max_size and corr <= 0.85 and liquidity.lower() == "high":
        return "在当前约束下，组合仓位看起来可控。"
    return "组合构建约束限制了持仓信心。"


def _portfolio_risk_tag(
    position_size: str,
    max_position_size: str,
    correlation: str,
    liquidity: str,
) -> str:
    claim = _portfolio_claim(position_size, max_position_size, correlation, liquidity).lower()
    if "manageable" in claim or "可控" in claim:
        return "portfolio_fit"
    if "watchlist" in claim or "观察名单" in claim or "incomplete" in claim or "不完整" in claim:
        return "portfolio_evidence_gap"
    return "portfolio_risk"


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
