"""Trade-plan report agent — replaces audit-style reports with actionable trading plans.

Output format changes:
  BEFORE: 11 sections of audit trail, evidence tables, scores
  AFTER:  Investor-ready trading plan with entry/stop/target/position sizing
"""

from __future__ import annotations

from pathlib import Path

from memory.storage import save_json
from models.schemas import (
    ClaimEvidenceCommit,
    InvestmentReport,
    MergeResult,
    Workspace,
)

# ── section builders ──────────────────────────────────────────────────────



class TradePlanReportAgent:
    """Builds an actionable trading-plan report (replaces the old ReportAgent)."""

    def generate_report(
        self,
        workspace: Workspace,
        merge_result: MergeResult,
    ) -> InvestmentReport:
        """Build an InvestmentReport whose markdown is a trade plan."""

        evidence_table = self._collect_commits(workspace)
        markdown = self._build_trade_plan(workspace, merge_result, evidence_table)
        audit = self._build_audit_slim(workspace, merge_result)

        return InvestmentReport(
            ticker=workspace.ticker,
            company_name=workspace.company_name,
            final_recommendation=merge_result.final_recommendation,
            merge_result=merge_result,
            audit_trail=audit,
            evidence_table=evidence_table,
            markdown_report=markdown,
        )

    def save_report(
        self,
        report: InvestmentReport,
        output_dir: str | Path = "outputs/reports",
    ) -> tuple[str, str]:
        """Save Markdown + JSON reports."""

        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        md_path = report_dir / f"{report.ticker}_trade_plan.md"
        json_path = report_dir / f"{report.ticker}_trade_plan.json"
        md_path.write_text(report.markdown_report, encoding="utf-8")
        save_json(json_path, report)
        return str(md_path), str(json_path)

    # ── helpers ───────────────────────────────────────────────────────────
    def _collect_commits(self, workspace: Workspace) -> list[ClaimEvidenceCommit]:
        commits: list[ClaimEvidenceCommit] = []
        for branch in workspace.branches.values():
            commits.extend(branch.commits)
        return commits

    # ── main trade plan builder ────────────────────────────────────────────
    def _build_trade_plan(
        self,
        workspace: Workspace,
        result: MergeResult,
        evidence_table: list[ClaimEvidenceCommit],
    ) -> str:
        """Assemble the full trade-plan markdown."""

        company = workspace.company_name or workspace.ticker
        ds = result.decision_scores
        rec_cn = _cn_recommendation(result.final_recommendation)

        sections: list[str] = []

        # ── 1. Decision header ──────────────────────────────────────────
        icon = {"Buy": "🟢", "Sell": "🔴", "Hold": "🟡"}.get(result.final_recommendation, "⚪")
        sections.append(
            f"# {icon} {workspace.ticker} 交易计划\n\n"
            f"**{company}** | 生成时间: {workspace.created_at}\n\n"
            f"## 核心决策\n\n"
            f"| 项目 | 内容 |\n"
            f"|---|---|\n"
            f"| **建议** | **{rec_cn}** |\n"
            f"| **置信度** | {_cn_confidence(result.confidence)} |\n"
            f"| **风险等级** | {_cn_risk(ds.risk_level)} |\n"
            f"| **仓位建议** | {ds.position_sizing_suggestion} |\n"
            f"| **时间框架** | 3-6 个月，下次财报后重评 |"
        )

        # ── 2. Trade plan table ─────────────────────────────────────────
        sections.append(self._build_trade_table(result))

        # ── 3. Investment thesis ────────────────────────────────────────
        sections.append(self._build_thesis(result))

        # ── 4. Key evidence summary ─────────────────────────────────────
        sections.append(self._build_evidence_summary(result, evidence_table))

        # ── 5. Risk & exit triggers ─────────────────────────────────────
        sections.append(self._build_risk_exit(result))

        # ── 6. Revision conditions ──────────────────────────────────────
        if result.conditions_for_revision:
            sections.append(self._build_revision_conditions(result))

        # ── 7. Decision rationale (from debate) ─────────────────────────
        sections.append(
            "## 📋 决策逻辑\n\n"
            f"{result.decision_rationale}\n\n"
            f"*{result.risk_adjustment}*"
        )

        # ── 8. Quick evidence reference ─────────────────────────────────
        sections.append(self._build_quick_reference(evidence_table))

        return "\n\n".join(sections) + "\n"

    # ── subsection builders ───────────────────────────────────────────────

    def _build_trade_table(self, result: MergeResult) -> str:
        """Build the concrete trade plan table.

        Tries to extract price levels from the decision rationale
        (which the DebateAgent embeds).  Falls back to score-based guidance.
        """

        rationale = result.decision_rationale

        # try to extract numeric levels from debate rationale
        entry_low = _extract_number(rationale, "entry_price_low")
        entry_high = _extract_number(rationale, "entry_price_high")
        stop = _extract_number(rationale, "stop_loss")
        target1 = _extract_number(rationale, "target_1")
        target2 = _extract_number(rationale, "target_2")

        def price_or_na(val: float | None) -> str:
            return f"${val:.2f}" if val is not None else "待定"

        entry_str = (
            f"{price_or_na(entry_low)} – {price_or_na(entry_high)}"
            if entry_low or entry_high
            else "等待技术面确认"
        )

        return (
            "## 📊 交易参数\n\n"
            "| 参数 | 数值 | 说明 |\n"
            "|---|---|---|\n"
            f"| **入场区间** | {entry_str} | 建议分批建仓 |\n"
            f"| **止损价** | {price_or_na(stop)} | 跌破则无条件退出 |\n"
            f"| **第一目标** | {price_or_na(target1)} | 达到后可移动止盈 |\n"
            f"| **第二目标** | {price_or_na(target2)} | 视市场环境决定是否持有 |\n"
            f"| **仓位占比** | {result.decision_scores.position_sizing_suggestion} | 控制单票风险 |\n"
            f"| **买点评分** | {result.decision_scores.entry_timing:.0f}/100 | "
            f"{'适合入场' if result.decision_scores.entry_timing >= 60 else '等待更好时机' if result.decision_scores.entry_timing >= 40 else '不建议入场'} |"
        )

    def _build_thesis(self, result: MergeResult) -> str:
        """Build the bull/bear thesis summary."""

        parts = ["## 🎯 投资逻辑\n"]

        # extract bull/bear strongest from rationale
        rationale = result.decision_rationale

        parts.append("### 看多逻辑")
        supporting = result.main_supporting_claims
        if supporting:
            for c in supporting[:3]:
                parts.append(f"- ✅ {c}")
        else:
            parts.append("- (无明确看多证据)")

        parts.append("\n### 看空逻辑")
        opposing = result.main_opposing_claims
        if opposing:
            for c in opposing[:3]:
                parts.append(f"- ⚠️ {c}")
        else:
            parts.append("- (无明确看空证据)")

        parts.append(f"\n### 综合判断\n{result.decision_rationale}")
        return "\n".join(parts)

    def _build_evidence_summary(
        self,
        result: MergeResult,
        evidence_table: list[ClaimEvidenceCommit],
    ) -> str:
        """Summarise evidence quality by branch."""

        # group by branch
        by_branch: dict[str, list[ClaimEvidenceCommit]] = {}
        for c in evidence_table:
            by_branch.setdefault(c.branch_name, []).append(c)

        rows: list[str] = [
            "## 📑 证据质量概览\n",
            "| 分支 | 条数 | 平均质量 | 时效 |",
            "|---|---|---|---|",
        ]
        for branch_name, commits in sorted(by_branch.items()):
            avg_q = (
                sum(c.evidence_quality_score or 0 for c in commits) / len(commits)
                if commits
                else 0
            )
            temporal = _worst_temporal(commits)
            rows.append(
                f"| {branch_name} | {len(commits)} | {avg_q:.2f} | {_cn_temporal(temporal)} |"
            )

        return "\n".join(rows)

    def _build_risk_exit(self, result: MergeResult) -> str:
        """Build the risk management and exit trigger section."""

        return (
            "## ⛔ 风险管理与退出规则\n\n"
            "| 规则 | 条件 | 动作 |\n"
            "|---|---|---|\n"
            "| **硬止损** | 价格跌破止损位 | 无条件全部卖出 |\n"
            "| **移动止盈** | 达到第一目标后 | 将止损上移至成本价，回撤 5% 卖出 |\n"
            "| **时间止损** | 持有超过时间框架未达目标 | 重新评估，减仓 50% |\n"
            "| **基本面止损** | 核心假设被证伪（见下方条件） | 立即退出 |\n"
            "| **仓位纪律** | 单票不超过组合 15% | 超过部分逐步减仓 |"
        )

    def _build_revision_conditions(self, result: MergeResult) -> str:
        """Build the conditions that would trigger a revision."""

        lines = ["## 🔄 触发重评的条件"]
        for i, cond in enumerate(result.conditions_for_revision, 1):
            lines.append(f"{i}. {cond}")
        return "\n".join(lines)

    def _build_quick_reference(
        self,
        evidence_table: list[ClaimEvidenceCommit],
    ) -> str:
        """Build a compact evidence reference table."""

        if not evidence_table:
            return "## 📎 证据附录\n\n*无证据记录*"

        rows = [
            "## 📎 证据附录（精简）\n",
            "| 来源 | 观点摘要 | 质量 | 时效 |",
            "|---|---|---|---|",
        ]
        for c in evidence_table[:20]:  # cap at 20 for readability
            claim_short = c.claim[:60] + ("..." if len(c.claim) > 60 else "")
            rows.append(
                f"| {c.branch_name} | {claim_short} | "
                f"{c.evidence_quality_score or '?'} | "
                f"{_cn_temporal(c.temporal_status)} |"
            )

        return "\n".join(rows)

    def _build_audit_slim(
        self,
        workspace: Workspace,
        result: MergeResult,
    ) -> list[str]:
        """Slim audit trail (kept for traceability, not the main output)."""

        return [
            f"Ticker: {workspace.ticker}",
            f"Recommendation: {result.final_recommendation} (confidence: {result.confidence})",
            f"Branches with evidence: {[b for b, br in workspace.branches.items() if br.commits]}",
            f"Supporting claims: {len(result.main_supporting_claims)}",
            f"Opposing claims: {len(result.main_opposing_claims)}",
            f"Key conflicts: {len(result.key_conflicts)}",
            f"Risk adjustment: {result.risk_adjustment}",
        ]


# ── Chinese translation helpers ────────────────────────────────────────────

def _cn_recommendation(val: str) -> str:
    return {"Buy": "买入 🟢", "Sell": "卖出 🔴", "Hold": "持有 🟡"}.get(val, val)


def _cn_confidence(val: str) -> str:
    return {"high": "高 ✅", "medium": "中 ⚠️", "low": "低 ❌"}.get(val, val)


def _cn_risk(val: str) -> str:
    return {"high": "高 🔴", "medium": "中 🟡", "low": "低 🟢"}.get(val, val)


def _cn_temporal(val: str | None) -> str:
    return {
        "valid": "✅ 有效", "stale": "⚠️ 偏旧",
        "expired": "❌ 过期", "unknown": "❓ 未知",
    }.get(val or "", "❓")


def _worst_temporal(commits: list[ClaimEvidenceCommit]) -> str:
    rank = {"expired": 0, "unknown": 1, "stale": 2, "valid": 3}
    return min(
        (c.temporal_status or "unknown" for c in commits),
        key=lambda s: rank.get(s, 1),
    )


def _extract_number(text: str, key: str) -> float | None:
    """Try to extract a number mentioned near a key in text."""
    import re

    # look for patterns like "entry_price_low": 680.5 or entry_price_low=680
    pattern = rf'{key}["\s:=]+([\d.]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None
