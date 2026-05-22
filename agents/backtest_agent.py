"""Backtest summary agent."""

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Workspace


class BacktestAgent(BaseAgent):
    """Creates commits about simple strategy backtest behavior."""

    name = "BacktestAgent"
    role = "backtest-agent"
    branch_name = "backtest-analysis"

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Generate commits from a precomputed backtest summary."""

        del workspace
        metrics = input_data.get("backtest_summary", {})
        timestamp = str(input_data.get("as_of_date", _news_timestamp(input_data)))
        strategy = _value(metrics, "strategy")
        win_rate = _value(metrics, "win_rate")
        max_drawdown = _value(metrics, "max_drawdown")
        annualized_return = _value(metrics, "annualized_return")
        total_return = _value(metrics, "total_return")
        buy_hold_return = _value(metrics, "buy_hold_return")
        excess_return = _value(metrics, "excess_return")

        performance_evidence = self._make_evidence(
            content=(
                f"策略 {strategy} 的胜率为 {win_rate}，最大回撤为 {max_drawdown}，"
                f"年化收益为 {annualized_return}，总收益为 {total_return}，"
                f"买入持有收益为 {buy_hold_return}，超额收益为 {excess_return}。"
            ),
            source="回测摘要模型",
            source_type="backtest_result",
            timestamp=timestamp,
            metric_name="backtest_performance",
            metric_value=f"{win_rate}/{max_drawdown}/{annualized_return}/{excess_return}",
        )

        return [
            self.create_commit(
                claim=_performance_claim(win_rate, max_drawdown, annualized_return),
                evidence=performance_evidence,
                confidence=_confidence(win_rate, max_drawdown, annualized_return),
                risk_tag=_performance_risk_tag(win_rate, max_drawdown, annualized_return),
                time_horizon="1-6 months",
            )
        ]


def _performance_claim(win_rate: str, max_drawdown: str, annualized_return: str) -> str:
    win = _parse_percent(win_rate)
    drawdown = _parse_percent(max_drawdown)
    annual_return = _parse_percent(annualized_return)
    if win is None or drawdown is None or annual_return is None:
        return "回测证据不完整，不应主导决策。"
    if annual_return > 8 and drawdown <= 15 and win >= 50:
        return "回测表现支持当前技术面设置。"
    if drawdown > 20 or annual_return < 0:
        return "回测表现显示风险调整后结果不利。"
    return "回测表现较为混合。"


def _performance_risk_tag(win_rate: str, max_drawdown: str, annualized_return: str) -> str:
    claim = _performance_claim(win_rate, max_drawdown, annualized_return).lower()
    if "supports" in claim or "支持" in claim:
        return "backtest_support"
    if "unfavorable" in claim or "不利" in claim:
        return "backtest_risk"
    if "incomplete" in claim or "不完整" in claim:
        return "backtest_evidence_gap"
    return "backtest_mixed"


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
