"""Decision revision engine for new evidence."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from config import INVESTMENT_THRESHOLDS
from models.schemas import (
    ClaimEvidenceCommit,
    MergeResult,
    RevisionRecord,
    RevisionResult,
    Workspace,
)


class RevisionEngine:
    """Compare new evidence with prior commits and revise the decision."""

    def revise(
        self,
        workspace: Workspace,
        previous_merge_result: MergeResult,
        new_evidence_data: dict[str, Any],
    ) -> RevisionResult:
        """Generate a revision result from new evidence."""

        revision_records = [
            self._revise_commit(commit, new_evidence_data)
            for commit in self._collect_commits(workspace)
        ]
        revised_recommendation = self._revised_recommendation(
            previous_merge_result.final_recommendation,
            revision_records,
        )
        key_changes = self._key_changes(revision_records)

        return RevisionResult(
            previous_recommendation=previous_merge_result.final_recommendation,
            revised_recommendation=revised_recommendation,
            revision_records=revision_records,
            key_changes=key_changes,
            revision_rationale=self._revision_rationale(
                previous_merge_result.final_recommendation,
                revised_recommendation,
                revision_records,
            ),
            updated_conditions_for_revision=_updated_conditions(new_evidence_data),
        )

    def _collect_commits(self, workspace: Workspace) -> list[ClaimEvidenceCommit]:
        """Collect all prior claim-evidence commits from the workspace."""

        return [
            commit
            for branch in workspace.branches.values()
            for commit in branch.commits
        ]

    def _revise_commit(
        self,
        commit: ClaimEvidenceCommit,
        new_evidence_data: dict[str, Any],
    ) -> RevisionRecord:
        """Compare one commit against new evidence."""

        claim_text = commit.claim.lower()
        risk_tag = commit.risk_tag.lower()
        financials = new_evidence_data.get("new_financial_metrics", {})
        technicals = new_evidence_data.get("new_technical_indicators", {})
        events = new_evidence_data.get("new_events", [])

        revenue_growth = _parse_percent(financials.get("revenue_growth_yoy"))
        net_margin = _parse_percent(financials.get("net_margin"))
        forward_pe = _parse_float(financials.get("forward_pe"))
        rsi = _parse_float(technicals.get("rsi"))
        price_trend = str(technicals.get("price_trend", "")).lower()
        volatility = str(technicals.get("volatility", "")).lower()

        summary = _new_evidence_summary(new_evidence_data)

        if _contains_any(claim_text, {"growth", "增长", "成长"}) and revenue_growth is not None:
            if revenue_growth < 0:
                return self._record(
                    commit,
                    summary,
                    "contradicted",
                    "新的收入增长证据为负，否定了原有增长观点。",
                    "change_recommendation",
                )
            if revenue_growth < INVESTMENT_THRESHOLDS["weak_growth_percent"]:
                return self._record(
                    commit,
                    summary,
                    "weakened",
                    "新的收入增长证据弱于原有增长观点。",
                    "decrease_confidence",
                )
            if _is_positive_claim_context(claim_text, risk_tag):
                return self._record(
                    commit,
                    summary,
                    "supported",
                    "新的收入增长证据较强，支持原有增长质量或成长性观点。",
                    "increase_confidence",
                )

        if (
            _contains_any(
                claim_text,
                {"profitability", "margin", "stable", "盈利", "利润率", "稳定"},
            )
            and net_margin is not None
        ):
            if net_margin < INVESTMENT_THRESHOLDS["revision_weak_net_margin_percent"]:
                return self._record(
                    commit,
                    summary,
                    "weakened",
                    "新的盈利能力证据显示净利率低于原稳定性阈值。",
                    "decrease_confidence",
                )
            if _is_risk_or_challenge_context(risk_tag):
                return self._record(
                    commit,
                    summary,
                    "weakened",
                    "新的盈利能力证据较强，削弱了原有利润率压力或反证担忧。",
                    "increase_confidence",
                )
            return self._record(
                commit,
                summary,
                "supported",
                "新的盈利能力证据显示净利率保持在较强水平，支持原有盈利质量观点。",
                "increase_confidence",
            )

        if _contains_any(
            claim_text,
            {"momentum", "trend", "constructive", "upward", "动量", "趋势", "建设性", "上行"},
        ):
            if rsi is not None and rsi < INVESTMENT_THRESHOLDS["weak_rsi"]:
                return self._record(
                    commit,
                    summary,
                    "contradicted",
                    "更新后的 RSI 低于 50，否定了原有动量或趋势观点。",
                    "decrease_confidence",
                )
            if "below" in price_trend:
                return self._record(
                    commit,
                    summary,
                    "contradicted",
                    "更新后的价格趋势跌破关键均线，否定了原有技术面观点。",
                    "decrease_confidence",
                )
            if (
                rsi is not None
                and rsi >= INVESTMENT_THRESHOLDS["bullish_rsi"]
                and "above" in price_trend
            ):
                return self._record(
                    commit,
                    summary,
                    "supported",
                    "新的 RSI 和价格趋势证据支持原有动量或趋势观点。",
                    "increase_confidence",
                )

        if (
            (_contains_any(claim_text, {"valuation", "估值"}) or "valuation_risk" in risk_tag)
            and forward_pe is not None
        ):
            if forward_pe > INVESTMENT_THRESHOLDS["revision_high_valuation_pe"]:
                return self._record(
                    commit,
                    summary,
                    "supported",
                    "新的估值证据仍然支持高估值风险。",
                    "decrease_confidence",
                )
            if forward_pe < INVESTMENT_THRESHOLDS["low_valuation_pe"]:
                if _is_risk_or_challenge_context(risk_tag) or _contains_any(
                    claim_text,
                    {"expensive", "premium", "risk", "pressure", "偏贵", "溢价", "高估", "风险"},
                ):
                    return self._record(
                        commit,
                        summary,
                        "weakened",
                        "新的估值证据显示远期市盈率较低，削弱了原有估值风险观点。",
                        "increase_confidence",
                    )
                return self._record(
                    commit,
                    summary,
                    "supported",
                    "新的估值证据显示远期市盈率较低，支持估值吸引力。",
                    "increase_confidence",
                )

        if _contains_any(claim_text, {"volatility", "波动"}) and volatility == "high":
            return self._record(
                commit,
                summary,
                "supported",
                "新的技术面证据支持原有高波动风险观点。",
                "decrease_confidence",
            )

        if events and _contains_any(
            " ".join(str(event.get("title", "")) for event in events).lower(),
            {"weakened", "slowing", "slowdown"},
        ) and _contains_any(claim_text, {"bullish", "growth", "偏多", "增长", "成长"}):
            return self._record(
                commit,
                summary,
                "weakened",
                "新的事件证据指向指引转弱或需求放缓。",
                "decrease_confidence",
            )

        if commit.temporal_status == "expired":
            return self._record(
                commit,
                summary,
                "expired",
                "原观点依赖的证据当前已经过期。",
                "decrease_confidence",
            )

        return self._record(
            commit,
            summary,
            "unchanged",
            "新的证据没有触发明确的修正规则。",
            "no_change",
        )

    def _record(
        self,
        commit: ClaimEvidenceCommit,
        summary: str,
        status: str,
        explanation: str,
        impact: str,
    ) -> RevisionRecord:
        """Create one revision record."""

        return RevisionRecord(
            revision_id=str(uuid4()),
            original_claim=commit.claim,
            original_branch=commit.branch_name,
            original_commit_id=commit.commit_id,
            new_evidence_summary=summary,
            revision_status=status,
            explanation=explanation,
            impact_on_decision=impact,
        )

    def _revised_recommendation(
        self,
        previous_recommendation: str,
        records: list[RevisionRecord],
    ) -> str:
        """Determine revised recommendation from revision records."""

        contradicted = sum(record.revision_status == "contradicted" for record in records)
        weakened = sum(
            record.revision_status == "weakened"
            and record.impact_on_decision in {"decrease_confidence", "change_recommendation"}
            for record in records
        )
        supported_risk = sum(
            record.revision_status == "supported"
            and record.impact_on_decision == "decrease_confidence"
            for record in records
        )
        positive_support = sum(
            record.impact_on_decision == "increase_confidence"
            for record in records
        )
        negative_pressure = contradicted + weakened + supported_risk

        if contradicted >= 2 and supported_risk >= 2:
            return "Avoid"
        if contradicted >= 1 and weakened >= 2:
            return "Sell"
        if positive_support >= 4 and negative_pressure <= 3:
            return _upgrade_recommendation(previous_recommendation)
        if positive_support >= 6 and negative_pressure <= 5:
            return _upgrade_recommendation(previous_recommendation)
        if weakened >= 2:
            return "Hold"
        return previous_recommendation

    def _key_changes(self, records: list[RevisionRecord]) -> list[str]:
        """Generate key changes from revision records."""

        changes: list[str] = []
        if any(
            _contains_any(record.original_claim.lower(), {"growth", "增长", "成长"})
            and record.impact_on_decision == "increase_confidence"
            for record in records
        ):
            changes.append("新的收入增长证据增强了成长性判断。")
        if any(
            _contains_any(record.original_claim.lower(), {"growth", "增长", "成长"})
            and record.revision_status in {"weakened", "contradicted"}
            and record.impact_on_decision in {"decrease_confidence", "change_recommendation"}
            for record in records
        ):
            changes.append("收入增长证据转弱，削弱成长性判断。")
        if any(
            _contains_any(record.original_claim.lower(), {"profitability", "margin", "盈利", "利润率"})
            and record.impact_on_decision == "increase_confidence"
            for record in records
        ):
            changes.append("新的盈利能力证据增强了盈利质量判断。")
        if any(
            _contains_any(record.original_claim.lower(), {"momentum", "trend", "动量", "趋势"})
            and record.revision_status == "contradicted"
            for record in records
        ):
            changes.append("技术动量被更新后的 RSI 或价格趋势否定。")
        if any(
            _contains_any(record.original_claim.lower(), {"momentum", "trend", "动量", "趋势"})
            and record.impact_on_decision == "increase_confidence"
            for record in records
        ):
            changes.append("新的技术面证据增强了趋势或动量判断。")
        if any(
            _contains_any(record.original_claim.lower(), {"valuation", "估值"})
            and record.revision_status == "supported"
            and record.impact_on_decision == "decrease_confidence"
            for record in records
        ):
            changes.append("高估值风险仍被新的远期市盈率证据支持。")
        if any(
            _contains_any(record.original_claim.lower(), {"valuation", "估值"})
            and record.impact_on_decision == "increase_confidence"
            for record in records
        ):
            changes.append("新的估值证据增强了估值吸引力。")
        if any(
            _contains_any(record.original_claim.lower(), {"volatility", "波动"})
            and record.revision_status == "supported"
            for record in records
        ):
            changes.append("高波动风险仍需要控制。")

        if not changes:
            changes.append("新的证据没有实质改变原有观点。")
        return changes[:5]

    def _revision_rationale(
        self,
        previous_recommendation: str,
        revised_recommendation: str,
        records: list[RevisionRecord],
    ) -> str:
        """Explain why the recommendation changed or stayed the same."""

        contradicted = sum(record.revision_status == "contradicted" for record in records)
        weakened = sum(
            record.revision_status == "weakened"
            and record.impact_on_decision in {"decrease_confidence", "change_recommendation"}
            for record in records
        )
        supported = sum(record.revision_status == "supported" for record in records)
        unchanged = sum(record.revision_status == "unchanged" for record in records)
        positive_support = sum(
            record.impact_on_decision == "increase_confidence"
            for record in records
        )

        if revised_recommendation == previous_recommendation:
            action = "修正后建议保持不变"
        else:
            action = (
                "修正后建议从 "
                f"{_translate_recommendation(previous_recommendation)} "
                f"调整为 {_translate_recommendation(revised_recommendation)}"
            )

        return (
            f"{action}。本次新证据修正中，{contradicted} 条观点被否定，"
            f"{weakened} 条观点被削弱，{positive_support} 条观点提升决策置信度，"
            f"{supported} 条观点获得支持，{unchanged} 条观点保持不变。"
            "修正主要由增长、盈利质量、估值、动量和波动率证据共同驱动。"
        )


def save_revision_report(
    revision_result: RevisionResult,
    ticker: str,
    output_dir: str | Path = "outputs/reports",
) -> str:
    """Save a Markdown revision report and return its path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"{ticker}_revision_report.md"
    report_path.write_text(
        _revision_markdown(revision_result, ticker),
        encoding="utf-8",
    )
    return str(report_path)


def _revision_markdown(revision_result: RevisionResult, ticker: str) -> str:
    """Build a concise Markdown revision report."""

    records = "\n".join(
        (
            f"- {_translate_revision_status(record.revision_status)}：{record.original_claim}\n"
            f"  - 分支：{record.original_branch}\n"
            f"  - 对决策影响：{_translate_impact(record.impact_on_decision)}\n"
            f"  - 解释：{record.explanation}"
        )
        for record in revision_result.revision_records
        if record.revision_status != "unchanged"
    )
    if not records:
        records = "- 没有观点发生实质性修正。"

    key_changes = "\n".join(f"- {change}" for change in revision_result.key_changes)
    conditions = "\n".join(
        f"- {condition}"
        for condition in revision_result.updated_conditions_for_revision
    )
    evidence_summary = (
        revision_result.revision_records[0].new_evidence_summary
        if revision_result.revision_records
        else "没有新的证据被纳入复核。"
    )

    return (
        f"# 决策修正报告：{ticker}\n\n"
        f"原建议：{_translate_recommendation(revision_result.previous_recommendation)}\n\n"
        f"修正后建议：{_translate_recommendation(revision_result.revised_recommendation)}\n\n"
        "## 新证据摘要\n"
        f"{evidence_summary}\n\n"
        "## 关键变化\n"
        f"{key_changes}\n\n"
        "## 修正记录\n"
        f"{records}\n\n"
        "## 修正理由\n"
        f"{revision_result.revision_rationale}\n\n"
        "## 后续重新评估条件\n"
        f"{conditions}\n"
    )


def _new_evidence_summary(new_evidence_data: dict[str, Any]) -> str:
    """Summarize new evidence data into a compact string."""

    financials = new_evidence_data.get("new_financial_metrics", {})
    technicals = new_evidence_data.get("new_technical_indicators", {})
    events = new_evidence_data.get("new_events", [])
    event_titles = "; ".join(
        (
            f"{event.get('title', '')} "
            f"({event.get('source', 'unknown source')})"
        ).strip()
        for event in events
    )
    return (
        f"财务指标={financials}; 技术指标={technicals}; 事件={event_titles}"
    )


def _parse_percent(value: Any) -> float | None:
    """Parse a percentage-like value into a float."""

    if value is None:
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    """Parse a numeric-like value into a float."""

    if value is None:
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return None


def _contains_any(text: str, keywords: set[str]) -> bool:
    """Return whether text contains any keyword."""

    lowered_text = text.lower()
    return any(keyword in lowered_text for keyword in keywords)


def _is_positive_claim_context(claim_text: str, risk_tag: str) -> bool:
    """Return whether a claim is meant to support the investment thesis."""

    if _is_risk_or_challenge_context(risk_tag):
        return False
    return not _contains_any(
        claim_text,
        {
            "risk",
            "pressure",
            "weaken",
            "limit",
            "constrain",
            "validation",
            "checked",
            "evaluated",
            "风险",
            "压力",
            "削弱",
            "限制",
            "反证",
        },
    )


def _is_risk_or_challenge_context(risk_tag: str) -> bool:
    """Return whether a risk tag describes caution or counter-evidence."""

    return _contains_any(
        risk_tag,
        {
            "risk",
            "counter",
            "evidence_gap",
            "uncertainty",
            "llm_risk",
            "llm_mixed",
            "gap",
        },
    )


def _upgrade_recommendation(previous_recommendation: str) -> str:
    """Upgrade a recommendation by one conservative step."""

    if previous_recommendation == "Avoid":
        return "Hold"
    if previous_recommendation == "Sell":
        return "Hold"
    if previous_recommendation == "Hold":
        return "Buy"
    return previous_recommendation


def _updated_conditions(new_evidence_data: dict[str, Any]) -> list[str]:
    """Create evidence-aware future revision conditions."""

    financials = new_evidence_data.get("new_financial_metrics", {})
    technicals = new_evidence_data.get("new_technical_indicators", {})
    revenue_growth = _parse_percent(financials.get("revenue_growth_yoy"))
    net_margin = _parse_percent(financials.get("net_margin"))
    forward_pe = _parse_float(financials.get("forward_pe"))
    rsi = _parse_float(technicals.get("rsi"))
    volatility = str(technicals.get("volatility", "unknown")).lower()

    conditions: list[str] = []
    if revenue_growth is not None:
        if revenue_growth >= INVESTMENT_THRESHOLDS["healthy_growth_percent"]:
            conditions.append("如果收入增长明显回落，需要重新评估成长性假设。")
        else:
            conditions.append("如果收入增长恢复到健康区间，可以上调成长性判断。")
    if net_margin is not None:
        if net_margin >= INVESTMENT_THRESHOLDS["revision_weak_net_margin_percent"]:
            conditions.append("如果净利率跌破当前强势区间，需要下调盈利质量判断。")
        else:
            conditions.append("如果净利率恢复到更健康水平，可以提高盈利质量置信度。")
    if forward_pe is not None:
        if forward_pe < INVESTMENT_THRESHOLDS["low_valuation_pe"]:
            conditions.append("如果远期市盈率快速上升，需要重新评估估值吸引力。")
        elif forward_pe > INVESTMENT_THRESHOLDS["revision_high_valuation_pe"]:
            conditions.append("如果估值风险下降且增长证据保持强劲，可以上调结论。")
        else:
            conditions.append("如果估值与增长出现明显背离，需要重新评估风险回报。")
    if rsi is not None:
        if rsi >= INVESTMENT_THRESHOLDS["bullish_rsi"]:
            conditions.append("如果 RSI 跌破 50 或价格跌破关键均线，需要刷新技术面判断。")
        else:
            conditions.append("如果 RSI 回到 55 以上且趋势证据有效，可以提高技术面置信度。")
    if volatility == "high":
        conditions.append("如果高波动持续，需要降低仓位或提高风险折扣。")
    else:
        conditions.append("如果波动率显著升高，需要重新评估持仓风险。")

    conditions.append("如果新的财报、指引或行业周期数据与当前判断相反，需要重新运行修正流程。")
    return conditions[:6]


def _translate_recommendation(value: str) -> str:
    """Translate recommendation labels to Chinese."""

    mapping = {
        "Buy": "买入",
        "Hold": "持有/观察",
        "Sell": "卖出",
        "Avoid": "回避",
    }
    return mapping.get(value, value)


def _translate_revision_status(value: str) -> str:
    """Translate revision status labels to Chinese."""

    mapping = {
        "supported": "获得支持",
        "weakened": "被削弱",
        "contradicted": "被否定",
        "unchanged": "不变",
        "expired": "证据过期",
    }
    return mapping.get(value, value)


def _translate_impact(value: str) -> str:
    """Translate revision impact labels to Chinese."""

    mapping = {
        "increase_confidence": "提高信心",
        "decrease_confidence": "降低信心",
        "change_recommendation": "改变建议",
        "no_change": "无变化",
    }
    return mapping.get(value, value)
