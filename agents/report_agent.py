"""Report generation agent."""

from pathlib import Path

from memory.storage import save_json
from models.schemas import ClaimEvidenceCommit, InvestmentReport, MergeResult, Workspace


class ReportAgent:
    """Builds a formal Markdown investment research report."""

    def generate_report(
        self,
        workspace: Workspace,
        merge_result: MergeResult,
    ) -> InvestmentReport:
        """Generate an investment report model with markdown content."""

        evidence_table = self._collect_commits(workspace)
        audit_trail = self._build_audit_trail(workspace, merge_result, evidence_table)
        markdown_report = self._build_markdown(workspace, merge_result, evidence_table)

        return InvestmentReport(
            ticker=workspace.ticker,
            company_name=workspace.company_name,
            final_recommendation=merge_result.final_recommendation,
            merge_result=merge_result,
            audit_trail=audit_trail,
            evidence_table=evidence_table,
            markdown_report=markdown_report,
        )

    def save_report(
        self,
        report: InvestmentReport,
        output_dir: str | Path = "outputs/reports",
    ) -> tuple[str, str]:
        """Save the Markdown and JSON reports and return their paths."""

        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = report_dir / f"{report.ticker}_report.md"
        json_path = report_dir / f"{report.ticker}_investment_report.json"

        markdown_path.write_text(report.markdown_report, encoding="utf-8")
        save_json(json_path, report)
        return str(markdown_path), str(json_path)

    def _collect_commits(self, workspace: Workspace) -> list[ClaimEvidenceCommit]:
        """Collect commits from all branches in insertion order."""

        return [
            commit
            for branch in workspace.branches.values()
            for commit in branch.commits
        ]

    def _build_markdown(
        self,
        workspace: Workspace,
        merge_result: MergeResult,
        evidence_table: list[ClaimEvidenceCommit],
    ) -> str:
        """Build the formal Markdown report."""

        return "\n\n".join(
            [
                f"# 投资研究报告：{workspace.ticker}",
                self._final_recommendation_section(workspace, merge_result),
                self._executive_summary_section(merge_result),
                self._scorecard_section(merge_result, evidence_table),
                self._claim_evidence_section(
                    "## 4. 支持性证据",
                    merge_result.main_supporting_claims,
                    evidence_table,
                ),
                self._claim_evidence_section(
                    "## 5. 反对与谨慎证据",
                    merge_result.main_opposing_claims,
                    evidence_table,
                ),
                self._conflicts_section(merge_result),
                self._counter_evidence_review_section(workspace),
                self._risk_review_section(workspace),
                self._audit_trail_section(workspace, merge_result, evidence_table),
                self._conditions_section(merge_result),
                self._full_evidence_table_section(evidence_table),
            ]
        ).strip() + "\n"

    def _final_recommendation_section(
        self,
        workspace: Workspace,
        merge_result: MergeResult,
    ) -> str:
        """Build the final recommendation section."""

        company_name = workspace.company_name or "Unknown"
        return (
            "## 1. 最终结论\n\n"
            f"- 股票代码：{workspace.ticker}\n"
            f"- 公司名称：{company_name}\n"
            f"- 研究问题：{workspace.research_question}\n"
            f"- 最终建议：{_translate_recommendation(merge_result.final_recommendation)}\n"
            f"- 置信度：{_translate_confidence(merge_result.confidence)}\n"
            f"- 当前买点评分：{merge_result.decision_scores.entry_score:.1f}/100\n"
            f"- 风险收益评分：{merge_result.decision_scores.risk_reward_score:.1f}/100\n"
            f"- 结论确信度评分：{merge_result.decision_scores.conviction_score:.1f}/100\n"
            f"- 估值吸引力：{merge_result.decision_scores.valuation_attractiveness:.1f}/100\n"
            f"- 技术择时评分：{merge_result.decision_scores.technical_timing_score:.1f}/100\n"
            f"- 风险等级：{_translate_risk_level(merge_result.decision_scores.risk_level)}\n"
            f"- 仓位建议：{_translate_position_sizing(merge_result.decision_scores.position_sizing_suggestion)}"
        )

    def _executive_summary_section(self, merge_result: MergeResult) -> str:
        """Build a concise deterministic executive summary."""

        return (
            "## 2. 核心摘要\n\n"
            f"最终建议为 **{_translate_recommendation(merge_result.final_recommendation)}**，"
            f"置信度为 **{_translate_confidence(merge_result.confidence)}**。"
            f"当前买点评分为 **{merge_result.decision_scores.entry_score:.1f}/100**，"
            f"风险收益评分为 **{merge_result.decision_scores.risk_reward_score:.1f}/100**，"
            f"风险等级为 **{_translate_risk_level(merge_result.decision_scores.risk_level)}**。"
            f"{_chinese_decision_rationale(merge_result)} "
            f"{_chinese_risk_adjustment(merge_result)}"
        )

    def _scorecard_section(
        self,
        merge_result: MergeResult,
        evidence_table: list[ClaimEvidenceCommit],
    ) -> str:
        """Build a multi-dimensional scorecard section."""

        dimensions = {
            "基本面": ("fundamental-analysis", "financial-statement-analysis"),
            "估值": ("valuation-analysis",),
            "行业比较": ("industry-comparison",),
            "宏观环境": ("macro-analysis",),
            "技术面": ("technical-analysis",),
            "回测表现": ("backtest-analysis",),
            "组合约束": ("portfolio-review",),
            "大模型洞察": ("llm-analysis",),
            "风险与反证": ("risk-review", "counter-evidence", "bear-case"),
        }
        lines = [
            "## 3. 多维评分卡",
            "",
            "### 3.1 决策量化指标",
            "",
            "| 指标 | 分数/等级 | 含义 |",
            "|---|---|---|",
            (
                f"| 当前买点评分 | {merge_result.decision_scores.entry_score:.1f}/100 | "
                "当前是否适合作为新买点 |"
            ),
            (
                f"| 风险收益评分 | {merge_result.decision_scores.risk_reward_score:.1f}/100 | "
                "潜在上行与风险约束的综合平衡 |"
            ),
            (
                f"| 结论确信度评分 | {merge_result.decision_scores.conviction_score:.1f}/100 | "
                "证据质量、时效和冲突后的综合确信度 |"
            ),
            (
                f"| 估值吸引力 | {merge_result.decision_scores.valuation_attractiveness:.1f}/100 | "
                "估值证据对买入的支持程度 |"
            ),
            (
                f"| 技术择时评分 | {merge_result.decision_scores.technical_timing_score:.1f}/100 | "
                "趋势、动量、波动和时效后的入场条件 |"
            ),
            (
                f"| 风险等级 | {_translate_risk_level(merge_result.decision_scores.risk_level)} | "
                "风险、反证和关键冲突的综合等级 |"
            ),
            "",
            "### 3.2 分支证据质量",
            "",
            "| 维度 | 平均证据分 | 时效状态 | 主要结论数量 |",
            "|---|---|---|---|",
        ]
        for label, branches in dimensions.items():
            commits = [
                commit
                for commit in evidence_table
                if commit.branch_name in branches
            ]
            if not commits:
                lines.append(f"| {label} | 无数据 | 无数据 | 0 |")
                continue
            average_score = round(
                sum(commit.evidence_quality_score or 0.0 for commit in commits)
                / len(commits),
                2,
            )
            temporal_status = _summarize_temporal_status(commits)
            lines.append(
                f"| {label} | {average_score:.2f} | {temporal_status} | {len(commits)} |"
            )
        return "\n".join(lines)

    def _claim_evidence_section(
        self,
        title: str,
        claims: list[str],
        commits: list[ClaimEvidenceCommit],
    ) -> str:
        """Build a detailed evidence section for a list of claims."""

        if not claims:
            return f"{title}\n\n本节暂无可归类观点。"

        claim_to_commit = {commit.claim: commit for commit in commits}
        lines = [title]
        for claim in claims:
            commit = claim_to_commit.get(claim)
            lines.append("")
            if commit is None:
                lines.append(f"- 观点：{claim}")
                lines.append("  - 关联提交：未找到")
                continue

            lines.extend(
                [
                    f"- 观点：{commit.claim}",
                    f"  - 证据内容：{commit.evidence.content}",
                    f"  - 来源：{commit.evidence.source}",
                    f"  - 来源类型：{commit.evidence.source_type}",
                    f"  - 证据质量分：{commit.evidence_quality_score}",
                    f"  - 时效状态：{_translate_temporal_status(commit.temporal_status)}",
                    f"  - 置信度：{_translate_confidence(commit.confidence)}",
                    f"  - 时间范围：{commit.time_horizon}",
                ]
            )
        return "\n".join(lines)

    def _conflicts_section(self, merge_result: MergeResult) -> str:
        """Build the key conflicts section."""

        lines = ["## 6. 关键冲突"]
        if not merge_result.key_conflicts:
            lines.extend(["", "未发现关键冲突。"])
            return "\n".join(lines)

        for conflict in merge_result.key_conflicts:
            lines.extend(
                [
                    "",
                    f"- 冲突类型：{conflict.conflict_type}",
                    f"  - 严重程度：{_translate_confidence(conflict.severity)}",
                    f"  - 观点 A：{conflict.claim_a}",
                    f"  - 观点 B：{conflict.claim_b}",
                    f"  - 解释：{conflict.explanation}",
                ]
            )
        return "\n".join(lines)

    def _counter_evidence_review_section(self, workspace: Workspace) -> str:
        """Summarize all commits from the counter-evidence branch."""

        counter_branch = workspace.branches.get("counter-evidence")
        counter_commits = counter_branch.commits if counter_branch else []
        lines = ["## 7. 反证检查"]
        if not counter_commits:
            lines.extend(["", "未发现反证检查记录。"])
            return "\n".join(lines)

        for commit in counter_commits:
            questions = commit.counter_evidence or []
            lines.extend(
                [
                    "",
                    f"- 观点：{commit.claim}",
                    f"  - 关联证据：{commit.evidence.content}",
                    f"  - 证据来源：{commit.evidence.source}",
                    f"  - 证据质量分：{commit.evidence_quality_score}",
                    f"  - 时效状态：{_translate_temporal_status(commit.temporal_status)}",
                    "  - 反证问题：",
                ]
            )
            if questions:
                lines.extend(f"    - {question}" for question in questions)
            else:
                lines.append("    - 无")
        return "\n".join(lines)

    def _risk_review_section(self, workspace: Workspace) -> str:
        """Summarize all commits from the risk-review branch."""

        risk_branch = workspace.branches.get("risk-review")
        risk_commits = risk_branch.commits if risk_branch else []
        lines = ["## 8. 风险审查"]
        if not risk_commits:
            lines.extend(["", "未发现风险审查记录。"])
            return "\n".join(lines)

        for commit in risk_commits:
            lines.extend(
                [
                    "",
                    f"- 观点：{commit.claim}",
                    f"  - 证据：{commit.evidence.content}",
                    f"  - 风险标签：{commit.risk_tag}",
                    f"  - 证据质量分：{commit.evidence_quality_score}",
                    f"  - 时效状态：{_translate_temporal_status(commit.temporal_status)}",
                    f"  - 置信度：{_translate_confidence(commit.confidence)}",
                ]
            )
        return "\n".join(lines)

    def _audit_trail_section(
        self,
        workspace: Workspace,
        merge_result: MergeResult,
        evidence_table: list[ClaimEvidenceCommit],
    ) -> str:
        """Build the decision audit trail section."""

        audit_trail = self._build_audit_trail(workspace, merge_result, evidence_table)
        lines = ["## 9. 决策审计轨迹"]
        for index, item in enumerate(audit_trail, start=1):
            lines.append(f"{index}. {item}")
        return "\n".join(lines)

    def _conditions_section(self, merge_result: MergeResult) -> str:
        """Build the conditions for revision section."""

        lines = ["## 10. 触发重新评估的条件"]
        for condition in merge_result.conditions_for_revision:
            lines.append(f"- {_translate_condition(condition)}")
        return "\n".join(lines)

    def _full_evidence_table_section(
        self,
        evidence_table: list[ClaimEvidenceCommit],
    ) -> str:
        """Build a Markdown table containing all claim-evidence commits."""

        header = (
            "| 分支 | Agent 角色 | 观点 | 证据来源 | 来源类型 | "
            "指标 | 证据质量 | 时效状态 | 置信度 | 风险标签 | 时间范围 |"
        )
        separator = (
            "|---|---|---|---|---|---|---|---|---|---|---|"
        )
        rows = [
            "| "
            + " | ".join(
                [
                    _escape_table_cell(commit.branch_name),
                    _escape_table_cell(commit.agent_role),
                    _escape_table_cell(commit.claim),
                    _escape_table_cell(commit.evidence.source),
                    _escape_table_cell(commit.evidence.source_type),
                    _escape_table_cell(commit.evidence.metric_name or ""),
                    _escape_table_cell(str(commit.evidence_quality_score)),
                    _escape_table_cell(_translate_temporal_status(commit.temporal_status)),
                    _escape_table_cell(_translate_confidence(commit.confidence)),
                    _escape_table_cell(commit.risk_tag),
                    _escape_table_cell(commit.time_horizon),
                ]
            )
            + " |"
            for commit in evidence_table
        ]
        return "\n".join(["## 11. 完整证据表", "", header, separator, *rows])

    def _build_audit_trail(
        self,
        workspace: Workspace,
        merge_result: MergeResult,
        evidence_table: list[ClaimEvidenceCommit],
    ) -> list[str]:
        """Build a step-by-step decision audit trail."""

        contributing_branches = [
            branch_name
            for branch_name, branch in workspace.branches.items()
            if branch.commits
        ]
        stale_or_weak = [
            commit
            for commit in evidence_table
            if commit.temporal_status in {"stale", "expired", "unknown"}
            or (commit.evidence_quality_score or 0.0) < 0.6
        ]

        return [
            "证据来自这些分支："
            + ", ".join(contributing_branches)
            + ".",
            f"{len(merge_result.main_supporting_claims)} 条观点支持投资假设。",
            f"{len(merge_result.main_opposing_claims)} 条观点反对或削弱投资假设。",
            f"合并审查中发现 {len(merge_result.key_conflicts)} 个关键冲突。",
            (
                f"{len(stale_or_weak)} 条记录存在过期、未知、时效偏弱或证据偏弱问题，"
                "因此降低了最终结论的置信度。"
            ),
            f"风险调整说明：{_chinese_risk_adjustment(merge_result)}",
            (
                f"最终建议为 {_translate_recommendation(merge_result.final_recommendation)}，"
                f"原因是：{_chinese_decision_rationale(merge_result)}"
            ),
        ]


def _escape_table_cell(value: str) -> str:
    """Escape Markdown table delimiters in a cell value."""

    return value.replace("|", "\\|").replace("\n", " ")


def _translate_recommendation(value: str) -> str:
    """Translate recommendation labels for the Chinese report."""

    mapping = {
        "Buy": "买入",
        "Hold": "持有/观察",
        "Sell": "卖出",
        "Avoid": "回避",
    }
    return mapping.get(value, value)


def _translate_confidence(value: str | None) -> str:
    """Translate confidence and severity labels for the Chinese report."""

    mapping = {
        "high": "高",
        "medium": "中",
        "low": "低",
    }
    return mapping.get(value or "", value or "未知")


def _translate_risk_level(value: str | None) -> str:
    """Translate risk-level labels for the Chinese report."""

    mapping = {
        "high": "高",
        "medium": "中",
        "low": "低",
    }
    return mapping.get(value or "", value or "未知")


def _translate_position_sizing(value: str) -> str:
    """Translate deterministic position-sizing suggestions."""

    mapping = {
        "Core starter position, up to 5-8% of portfolio": (
            "核心试探仓，最多约 5-8% 组合权重"
        ),
        "Small starter position, up to 2-4% of portfolio": (
            "小额试探仓，最多约 2-4% 组合权重"
        ),
        "Watchlist or existing-position hold; new entry only on better setup": (
            "观察名单或已有仓位持有；新买入需等待更好设置"
        ),
        "Reduce or avoid adding exposure": "降低仓位或避免新增暴露",
        "Observation only; avoid new exposure": "仅观察，避免新增暴露",
    }
    return mapping.get(value, value)


def _translate_temporal_status(value: str | None) -> str:
    """Translate temporal status labels for the Chinese report."""

    mapping = {
        "valid": "有效",
        "stale": "偏旧",
        "expired": "过期",
        "unknown": "未知",
    }
    return mapping.get(value or "", value or "未知")


def _summarize_temporal_status(commits: list[ClaimEvidenceCommit]) -> str:
    """Return the weakest temporal status for a group of commits."""

    rank = {"expired": 0, "unknown": 1, "stale": 2, "valid": 3}
    weakest = min(
        (commit.temporal_status or "unknown" for commit in commits),
        key=lambda status: rank.get(status, 1),
    )
    return _translate_temporal_status(weakest)


def _chinese_decision_rationale(merge_result: MergeResult) -> str:
    """Build a Chinese rationale from merge result fields."""

    conflict_count = len(merge_result.key_conflicts)
    supporting_count = len(merge_result.main_supporting_claims)
    opposing_count = len(merge_result.main_opposing_claims)
    return (
        "系统将增长、盈利质量、财务健康、行业比较、技术面、回测表现和组合约束 "
        "与估值风险、波动风险、宏观压力、证据时效和反证缺口一起权衡。"
        f"本次合并中有 {supporting_count} 条支持性观点、"
        f"{opposing_count} 条反对或谨慎观点，并识别出 {conflict_count} 个关键冲突。"
        "因此结论不是单纯看好或看空，而是根据证据强度和风险约束做出的综合判断。"
    )


def _chinese_risk_adjustment(merge_result: MergeResult) -> str:
    """Build a Chinese risk-adjustment sentence."""

    conflict_types = {conflict.conflict_type for conflict in merge_result.key_conflicts}
    reasons: list[str] = []
    if "risk_constraint" in conflict_types:
        reasons.append("风险因素限制了正面观点可以转化为买入结论的力度")
    if "temporal_warning" in conflict_types:
        reasons.append("部分证据已经偏旧或过期，尤其会影响技术面和择时判断")
    if "evidence_gap" in conflict_types:
        reasons.append("部分正面观点仍缺少足够反证验证")
    if not reasons:
        return "当前风险调整不明显。"
    return "风险调整：" + "；".join(reasons) + "。"


def _translate_condition(condition: str) -> str:
    """Translate deterministic revision conditions."""

    mapping = {
        "Upgrade if valuation risk decreases and growth evidence remains strong.": (
            "如果估值风险下降且增长证据仍然强劲，可以上调结论。"
        ),
        "Downgrade if revenue growth weakens.": "如果收入增长转弱，需要下调结论。",
        "Downgrade if volatility increases.": "如果波动率进一步升高，需要下调结论。",
        "Refresh technical view if stale technical indicators are updated.": (
            "如果过期的技术指标得到更新，需要重新刷新技术面判断。"
        ),
        "Increase confidence if counter-evidence search finds no strong contradiction.": (
            "如果反证检查没有发现强冲突，可以提高结论置信度。"
        ),
    }
    return mapping.get(condition, condition)
