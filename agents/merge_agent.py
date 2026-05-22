"""Conflict-aware merge agent for research branches."""

import re
from uuid import uuid4

from config import INVESTMENT_THRESHOLDS
from models.schemas import ClaimEvidenceCommit, Conflict, DecisionScores, MergeResult, Workspace


POSITIVE_KEYWORDS = {
    "positive",
    "strong",
    "stable",
    "constructive",
    "upward",
    "growth",
    "bullish",
    "improves",
    "support",
    "supports",
    "momentum",
    "financial health",
    "durable",
    "favorably",
    "supportive",
    "manageable",
    "discount",
    "增长",
    "强劲",
    "稳定",
    "支持",
    "有利",
    "改善",
    "优势",
    "健康",
    "可控",
    "看多",
    "上行",
    "动能",
    "现金流",
    "低估",
    "折价",
    "吸引力",
}

NEGATIVE_KEYWORDS = {
    "risk",
    "expensive",
    "valuation",
    "volatile",
    "volatility",
    "weaken",
    "constrain",
    "limit",
    "uncertainty",
    "stale",
    "gap",
    "overconfidence",
    "downside",
    "bearish",
    "validation",
    "pressure",
    "premium",
    "elevated",
    "lags",
    "unfavorable",
    "constraint",
    "constraints",
    "风险",
    "昂贵",
    "高估",
    "溢价",
    "波动",
    "转弱",
    "放缓",
    "限制",
    "压力",
    "下行",
    "不确定",
    "过期",
    "缺口",
    "谨慎",
    "回避",
    "弱",
}

RISK_TAG_KEYWORDS = {
    "risk",
    "valuation",
    "volatility",
    "uncertainty",
    "evidence_gap",
    "temporal",
    "downside",
    "overconfidence",
    "counter_evidence",
    "llm_risk",
    "llm_evidence_gap",
    "llm_mixed",
}

RISK_CONSTRAINT_TAGS = {
    "valuation_risk",
    "volatility_risk",
    "macro_risk",
    "technical_resistance",
    "downside",
    "uncertainty",
}

EVIDENCE_GAP_TAGS = {
    "evidence_gap",
    "temporal_uncertainty",
    "reasoning_risk",
    "overconfidence",
    "weak_evidence",
    "counter_evidence",
}

THEME_KEYWORDS = {
    "growth": {"growth", "revenue", "bullish", "增长", "收入", "成长", "偏多"},
    "profitability": {"profitability", "profit", "margin", "margins", "盈利", "利润率", "净利率"},
    "valuation": {"valuation", "expensive", "pe", "upside", "multiple", "估值", "偏贵", "市盈率", "溢价", "折价"},
    "momentum": {"momentum", "rsi", "upward", "动量", "上行"},
    "volatility": {"volatility", "volatile", "波动"},
    "technical": {"technical", "trend", "support", "resistance", "price", "技术", "趋势", "支撑", "阻力", "价格", "均线"},
    "balance sheet": {"balance", "cash", "financial health", "capital allocation", "资产负债表", "现金", "财务健康", "资本配置"},
    "temporal": {"temporal", "stale", "expired", "recency", "updated", "时效", "过期", "更新"},
    "evidence quality": {"evidence", "gap", "counter", "contradiction", "证据", "缺口", "反证", "冲突"},
    "llm synthesis": {"llm", "大模型", "综合", "洞察"},
}

DIRECT_CONTRADICTION_TERMS = {
    "growth": (
        {"strong", "positive", "growth", "growing"},
        {"weak", "weakening", "declining", "negative"},
    ),
    "profitability": (
        {"stable", "improving", "strong"},
        {"declining", "weak", "weakening"},
    ),
    "momentum": (
        {"upward", "positive", "constructive", "momentum"},
        {"negative", "weak", "weakening"},
    ),
    "valuation": (
        {"attractive", "cheap", "undervalued"},
        {"expensive", "overvalued"},
    ),
    "volatility": (
        {"low", "stable", "moderate"},
        {"high", "increasing", "volatile"},
    ),
}

TEMPORAL_WEIGHTS = {
    "valid": 1.0,
    "stale": 0.7,
    "expired": 0.3,
    "unknown": 0.5,
}


class MergeAgent:
    """Synthesizes branch commits into a conflict-aware merge result."""

    def __init__(self) -> None:
        self.support_score: float = 0.0
        self.risk_score: float = 0.0
        self.opposition_score: float = 0.0

    def merge(self, workspace: Workspace) -> MergeResult:
        """Merge all branch commits into a deterministic recommendation."""

        all_commits = self._collect_commits(workspace)
        supporting_commits = self._classify_supporting_commits(all_commits)
        opposing_commits = self._classify_opposing_commits(all_commits)
        risk_commits = self._classify_risk_commits(all_commits)

        self.support_score = self._weighted_average_score(supporting_commits)
        self.risk_score = self._weighted_average_score(risk_commits)
        self.opposition_score = self._weighted_average_score(opposing_commits)

        conflicts = self._detect_conflicts(
            supporting_commits,
            self._dedupe_commits(opposing_commits + risk_commits),
        )
        final_recommendation = self._recommendation(all_commits)
        confidence = self._confidence(final_recommendation, all_commits, conflicts)
        decision_scores = self._decision_scores(
            final_recommendation,
            all_commits,
            conflicts,
        )

        return MergeResult(
            final_recommendation=final_recommendation,
            confidence=confidence,
            decision_scores=decision_scores,
            main_supporting_claims=[commit.claim for commit in supporting_commits],
            main_opposing_claims=[commit.claim for commit in opposing_commits],
            key_conflicts=conflicts,
            risk_adjustment=self._risk_adjustment(all_commits, conflicts),
            decision_rationale=self._decision_rationale(
                final_recommendation,
                all_commits,
                conflicts,
            ),
            conditions_for_revision=[
                "Upgrade if valuation risk decreases and growth evidence remains strong.",
                "Downgrade if revenue growth weakens.",
                "Downgrade if volatility increases.",
                "Refresh technical view if stale technical indicators are updated.",
                "Increase confidence if counter-evidence search finds no strong contradiction.",
            ],
        )

    def _collect_commits(self, workspace: Workspace) -> list[ClaimEvidenceCommit]:
        """Collect all commits from all workspace branches."""

        return [
            commit
            for branch in workspace.branches.values()
            for commit in branch.commits
        ]

    def _classify_supporting_commits(
        self,
        commits: list[ClaimEvidenceCommit],
    ) -> list[ClaimEvidenceCommit]:
        """Find commits with positive investment support language."""

        return [
            commit
            for commit in commits
            if _is_supporting_commit(commit)
        ]

    def _classify_opposing_commits(
        self,
        commits: list[ClaimEvidenceCommit],
    ) -> list[ClaimEvidenceCommit]:
        """Find commits with caution, downside, or negative language."""

        return [
            commit
            for commit in commits
            if _contains_any(commit.claim, NEGATIVE_KEYWORDS)
            or _contains_any(commit.risk_tag, {"counter_evidence"})
            or commit.branch_name == "counter-evidence"
        ]

    def _classify_risk_commits(
        self,
        commits: list[ClaimEvidenceCommit],
    ) -> list[ClaimEvidenceCommit]:
        """Find commits whose risk tags indicate review risk."""

        return [
            commit
            for commit in commits
            if _contains_any(commit.risk_tag, RISK_TAG_KEYWORDS)
        ]

    def _detect_conflicts(
        self,
        supporting_commits: list[ClaimEvidenceCommit],
        opposing_or_risk_commits: list[ClaimEvidenceCommit],
    ) -> list[Conflict]:
        """Detect direct conflicts, risk constraints, warnings, and gaps."""

        conflicts: list[Conflict] = []
        seen_pairs: set[tuple[str, str]] = set()

        for support_commit in supporting_commits:
            if support_commit.temporal_status in {"stale", "expired", "unknown"}:
                conflicts.append(self._temporal_warning(support_commit))

        for support_commit in supporting_commits:
            for risk_commit in opposing_or_risk_commits:
                if support_commit.commit_id == risk_commit.commit_id:
                    continue

                pair_key = (support_commit.claim, risk_commit.claim)
                if pair_key in seen_pairs:
                    continue

                if self._is_direct_conflict(support_commit, risk_commit):
                    conflicts.append(
                        self._claim_pair_conflict(
                            "direct_conflict",
                            support_commit,
                            risk_commit,
                            (
                                "The two claims directly oppose each other on "
                                "the same research theme."
                            ),
                        )
                    )
                    seen_pairs.add(pair_key)
                    continue

                if self._is_risk_constraint(support_commit, risk_commit):
                    conflicts.append(
                        self._claim_pair_conflict(
                            "risk_constraint",
                            support_commit,
                            risk_commit,
                            (
                                "The positive claim is not directly contradicted, "
                                "but its investment implication is limited by this "
                                "risk claim."
                            ),
                        )
                    )
                    seen_pairs.add(pair_key)

        for risk_commit in opposing_or_risk_commits:
            if _is_evidence_gap_commit(risk_commit):
                related_support = self._related_supporting_commit(
                    risk_commit,
                    supporting_commits,
                )
                conflicts.append(
                    self._evidence_gap_conflict(
                        risk_commit,
                        related_support,
                    )
                )

        return self._select_key_conflicts(conflicts)

    def _is_direct_conflict(
        self,
        support_commit: ClaimEvidenceCommit,
        risk_commit: ClaimEvidenceCommit,
    ) -> bool:
        """Return whether two claims directly contradict on the same theme."""

        support_text = _commit_text(support_commit)
        risk_text = _commit_text(risk_commit)

        for theme, (positive_terms, negative_terms) in DIRECT_CONTRADICTION_TERMS.items():
            if theme not in _themes_for_commit(support_commit):
                continue
            if theme not in _themes_for_commit(risk_commit):
                continue
            support_positive = _contains_any(support_text, positive_terms)
            support_negative = _contains_any(support_text, negative_terms)
            risk_positive = _contains_any(risk_text, positive_terms)
            risk_negative = _contains_any(risk_text, negative_terms)
            if support_positive and risk_negative:
                return True
            if support_negative and risk_positive:
                return True
        return False

    def _is_risk_constraint(
        self,
        support_commit: ClaimEvidenceCommit,
        risk_commit: ClaimEvidenceCommit,
    ) -> bool:
        """Return whether a risk claim constrains a positive claim."""

        if _is_evidence_gap_commit(risk_commit):
            return False

        if not _contains_any(risk_commit.risk_tag, RISK_CONSTRAINT_TAGS):
            return False

        support_themes = _themes_for_commit(support_commit)
        risk_themes = _themes_for_commit(risk_commit)
        if {"valuation", "volatility"} & risk_themes:
            return bool(
                support_themes
                & {"growth", "profitability", "momentum", "technical", "valuation"}
            )
        if "technical" in risk_themes:
            return bool(support_themes & {"momentum", "technical"})
        return bool(support_themes)

    def _claim_pair_conflict(
        self,
        conflict_type: str,
        support_commit: ClaimEvidenceCommit,
        risk_commit: ClaimEvidenceCommit,
        explanation: str,
    ) -> Conflict:
        """Create a conflict for a support and risk/opposing claim pair."""

        return Conflict(
            conflict_id=str(uuid4()),
            conflict_type=conflict_type,
            claim_a=support_commit.claim,
            claim_b=risk_commit.claim,
            explanation=explanation,
            severity=self._conflict_severity(risk_commit),
        )

    def _temporal_warning(self, support_commit: ClaimEvidenceCommit) -> Conflict:
        """Create a temporal warning for a stale/expired/unknown support claim."""

        status = support_commit.temporal_status or "unknown"
        return Conflict(
            conflict_id=str(uuid4()),
            conflict_type="temporal_warning",
            claim_a=support_commit.claim,
            claim_b=(
                f"Supporting evidence has temporal_status='{status}' for "
                f"time horizon '{support_commit.time_horizon}'."
            ),
            explanation=(
                "This is a temporal warning, not a direct contradiction. "
                "The claim depends on evidence that may be too old for its "
                "intended investment horizon."
            ),
            severity=self._temporal_warning_severity(status),
        )

    def _evidence_gap_conflict(
        self,
        risk_commit: ClaimEvidenceCommit,
        support_commit: ClaimEvidenceCommit | None,
    ) -> Conflict:
        """Create an evidence gap conflict from a reasoning-audit risk commit."""

        affects_bullish_thesis = _contains_any(
            f"{risk_commit.claim} {risk_commit.risk_tag}",
            {"bullish", "thesis", "counter evidence", "counter"},
        )
        if affects_bullish_thesis:
            severity = "high" if (risk_commit.evidence_quality_score or 0.0) >= 0.8 else "medium"
        else:
            severity = "low"

        return Conflict(
            conflict_id=str(uuid4()),
            conflict_type="evidence_gap",
            claim_a=support_commit.claim if support_commit else "Evidence audit",
            claim_b=risk_commit.claim,
            explanation=(
                "This is an evidence gap, not a direct contradiction. The risk "
                "claim indicates missing counter-evidence, weak evidence, "
                "overconfidence, or reasoning uncertainty."
            ),
            severity=severity,
        )

    def _conflict_severity(self, risk_commit: ClaimEvidenceCommit) -> str:
        """Assign conflict severity from opposing/risk evidence strength."""

        quality = risk_commit.evidence_quality_score or 0.0
        if quality >= 0.8 and risk_commit.temporal_status == "valid":
            return "high"
        if quality >= 0.6:
            return "medium"
        return "low"

    def _temporal_warning_severity(self, temporal_status: str) -> str:
        """Assign severity for temporal warnings."""

        if temporal_status == "expired":
            return "high"
        if temporal_status == "stale":
            return "medium"
        return "low"

    def _related_supporting_commit(
        self,
        risk_commit: ClaimEvidenceCommit,
        supporting_commits: list[ClaimEvidenceCommit],
    ) -> ClaimEvidenceCommit | None:
        """Find a supporting commit that shares a theme with a risk commit."""

        risk_themes = _themes_for_commit(risk_commit)
        for support_commit in supporting_commits:
            if _themes_for_commit(support_commit) & risk_themes:
                return support_commit
        return supporting_commits[0] if supporting_commits else None

    def _weighted_average_score(self, commits: list[ClaimEvidenceCommit]) -> float:
        """Average evidence quality after applying temporal validity weights."""

        if not commits:
            return 0.0

        adjusted_scores = [
            (commit.evidence_quality_score or 0.0)
            * TEMPORAL_WEIGHTS.get(commit.temporal_status or "unknown", 0.5)
            for commit in commits
        ]
        return round(sum(adjusted_scores) / len(adjusted_scores), 2)

    def _decision_scores(
        self,
        final_recommendation: str,
        commits: list[ClaimEvidenceCommit],
        conflicts: list[Conflict],
    ) -> DecisionScores:
        """Build normalized decision metrics for clearer recommendation output."""

        support = self.support_score
        risk = max(self.risk_score, self.opposition_score)
        valuation_attractiveness = self._valuation_attractiveness(commits)
        technical_timing = self._technical_timing_score(commits)
        freshness = self._freshness_score(commits)
        conflict_penalty = self._conflict_penalty(conflicts)

        entry_score = _clamp_score(
            100
            * (
                0.40 * support
                + 0.25 * valuation_attractiveness
                + 0.25 * technical_timing
                + 0.10 * freshness
                - 0.30 * risk
                - conflict_penalty
            )
        )
        risk_reward_score = _clamp_score(
            100
            * (
                0.45 * support
                + 0.30 * valuation_attractiveness
                + 0.15 * technical_timing
                - 0.35 * risk
                - 0.50 * conflict_penalty
            )
        )
        conviction_score = _clamp_score(
            100
            * (
                0.45 * max(support, 0.0)
                + 0.35 * freshness
                - 0.25 * risk
                - conflict_penalty
            )
        )
        risk_level = self._risk_level(risk, conflicts)

        return DecisionScores(
            support_score=_percent(support),
            risk_score=_percent(self.risk_score),
            opposition_score=_percent(self.opposition_score),
            entry_score=entry_score,
            risk_reward_score=risk_reward_score,
            conviction_score=conviction_score,
            valuation_attractiveness=_percent(valuation_attractiveness),
            technical_timing_score=_percent(technical_timing),
            risk_level=risk_level,
            position_sizing_suggestion=_position_sizing_suggestion(
                final_recommendation,
                risk_level,
                conviction_score,
            ),
        )

    def _valuation_attractiveness(self, commits: list[ClaimEvidenceCommit]) -> float:
        """Estimate valuation attractiveness from valuation support and risk evidence."""

        valuation_commits = [
            commit
            for commit in commits
            if commit.branch_name in {"valuation-analysis", "industry-comparison"}
            or "valuation" in _commit_text(commit)
            or "估值" in _commit_text(commit)
        ]
        if not valuation_commits:
            return 0.5

        valuation_support = self._weighted_average_score(
            [
                commit
                for commit in valuation_commits
                if _is_supporting_commit(commit)
                and not _contains_any(commit.risk_tag, {"valuation_risk"})
            ]
        )
        valuation_risk = self._weighted_average_score(
            [
                commit
                for commit in valuation_commits
                if _contains_any(_commit_text(commit), NEGATIVE_KEYWORDS)
                or _contains_any(commit.risk_tag, {"valuation_risk", "valuation_mixed"})
            ]
        )
        return _clamp_unit(0.50 + 0.50 * valuation_support - 0.50 * valuation_risk)

    def _technical_timing_score(self, commits: list[ClaimEvidenceCommit]) -> float:
        """Estimate whether current technical evidence supports entry timing."""

        technical_commits = [
            commit
            for commit in commits
            if commit.branch_name in {"technical-analysis", "backtest-analysis"}
        ]
        if not technical_commits:
            return 0.5

        technical_support = self._weighted_average_score(
            [commit for commit in technical_commits if _is_supporting_commit(commit)]
        )
        technical_risk = self._weighted_average_score(
            [
                commit
                for commit in technical_commits
                if _contains_any(_commit_text(commit), NEGATIVE_KEYWORDS)
                or _contains_any(commit.risk_tag, {"volatility_risk", "resistance"})
            ]
        )
        freshness = self._freshness_score(technical_commits)
        return _clamp_unit(
            0.15 + 0.55 * technical_support + 0.30 * freshness - 0.35 * technical_risk
        )

    def _freshness_score(self, commits: list[ClaimEvidenceCommit]) -> float:
        """Return the average temporal validity score for a set of commits."""

        if not commits:
            return 0.0
        return round(
            sum(TEMPORAL_WEIGHTS.get(commit.temporal_status or "unknown", 0.5) for commit in commits)
            / len(commits),
            2,
        )

    def _conflict_penalty(self, conflicts: list[Conflict]) -> float:
        """Convert conflicts into a small normalized penalty."""

        weights = {"high": 0.08, "medium": 0.05, "low": 0.02}
        return min(sum(weights.get(conflict.severity, 0.03) for conflict in conflicts), 0.25)

    def _risk_level(self, risk: float, conflicts: list[Conflict]) -> str:
        """Map risk and conflict severity to a simple risk label."""

        high_conflicts = sum(1 for conflict in conflicts if conflict.severity == "high")
        medium_conflicts = sum(1 for conflict in conflicts if conflict.severity == "medium")
        if risk >= 0.65 or high_conflicts >= 2:
            return "high"
        if risk >= 0.45 or high_conflicts == 1 or medium_conflicts >= 2:
            return "medium"
        return "low"

    def _recommendation(self, all_commits: list[ClaimEvidenceCommit]) -> str:
        """Choose the final recommendation from aggregate scores."""

        if self._evidence_is_too_weak_or_stale(all_commits):
            return "Avoid"
        if (
            self.support_score >= INVESTMENT_THRESHOLDS["high_support_score"]
            and self.risk_score < INVESTMENT_THRESHOLDS["medium_risk_score"]
            and self.opposition_score < INVESTMENT_THRESHOLDS["medium_risk_score"]
        ):
            return "Buy"
        if self.support_score >= INVESTMENT_THRESHOLDS["medium_support_score"] and (
            self.risk_score >= INVESTMENT_THRESHOLDS["medium_risk_score"]
            or self.opposition_score >= INVESTMENT_THRESHOLDS["medium_risk_score"]
        ):
            return "Hold"
        if self.support_score < INVESTMENT_THRESHOLDS["medium_support_score"] and (
            self.risk_score >= INVESTMENT_THRESHOLDS["high_risk_score"]
            or self.opposition_score >= INVESTMENT_THRESHOLDS["high_risk_score"]
        ):
            return "Sell"
        return "Hold"

    def _evidence_is_too_weak_or_stale(
        self,
        commits: list[ClaimEvidenceCommit],
    ) -> bool:
        """Check whether evidence quality and freshness are too weak overall."""

        if not commits:
            return True

        weak_count = sum(
            1
            for commit in commits
            if (commit.evidence_quality_score or 0.0)
            < INVESTMENT_THRESHOLDS["weak_evidence_score"]
            or commit.temporal_status in {"stale", "expired", "unknown"}
        )
        return weak_count / len(commits) > INVESTMENT_THRESHOLDS["weak_or_stale_ratio"]

    def _confidence(
        self,
        final_recommendation: str,
        commits: list[ClaimEvidenceCommit],
        conflicts: list[Conflict],
    ) -> str:
        """Determine confidence from evidence quality, freshness, and conflicts."""

        del final_recommendation
        medium_high_conflicts = [
            conflict
            for conflict in conflicts
            if conflict.severity in {"medium", "high"}
        ]
        stale_or_weak_count = sum(
            1
            for commit in commits
            if commit.temporal_status in {"stale", "expired", "unknown"}
            or (commit.evidence_quality_score or 0.0) < 0.6
        )

        if (
            stale_or_weak_count / max(len(commits), 1)
            > INVESTMENT_THRESHOLDS["low_confidence_stale_or_weak_ratio"]
        ):
            return "low"
        if len(medium_high_conflicts) >= 4:
            return "low"
        if (
            max(self.support_score, self.risk_score, self.opposition_score)
            >= INVESTMENT_THRESHOLDS["high_confidence_score"]
            and len(medium_high_conflicts) < 2
        ):
            return "high"
        return "medium"

    def _risk_adjustment(
        self,
        commits: list[ClaimEvidenceCommit],
        conflicts: list[Conflict],
    ) -> str:
        """Explain whether risk changed the recommendation posture."""

        has_valuation_risk = any(
            _contains_any(commit.risk_tag, {"valuation_risk"})
            for commit in commits
        )
        has_stale_technical = any(
            commit.evidence.source_type in {"technical_indicator", "mock_technical_indicator"}
            and commit.temporal_status in {"stale", "expired", "unknown"}
            for commit in commits
        )
        has_evidence_gap = any(
            conflict.conflict_type == "evidence_gap"
            for conflict in conflicts
        )
        reasons: list[str] = []
        if has_valuation_risk:
            reasons.append("valuation risk limits upside interpretation")
        if has_stale_technical:
            reasons.append("stale technical evidence reduces timing confidence")
        if has_evidence_gap:
            reasons.append("missing counter-evidence keeps thesis confidence capped")

        reason_text = "; ".join(reasons)
        if (
            self.risk_score >= INVESTMENT_THRESHOLDS["high_risk_score"]
            or self.opposition_score >= INVESTMENT_THRESHOLDS["high_risk_score"]
        ):
            return (
                "Risk materially reduces conviction"
                + (f" because {reason_text}." if reason_text else ".")
            )
        if (
            self.risk_score >= INVESTMENT_THRESHOLDS["medium_risk_score"]
            or self.opposition_score >= INVESTMENT_THRESHOLDS["medium_risk_score"]
        ):
            return (
                "Risk moderates the decision"
                + (f" because {reason_text}." if reason_text else ".")
            )
        return "Risk does not materially reduce the recommendation under current evidence."

    def _decision_rationale(
        self,
        final_recommendation: str,
        commits: list[ClaimEvidenceCommit],
        conflicts: list[Conflict],
    ) -> str:
        """Explain the deterministic merge decision."""

        stale_count = sum(
            1
            for commit in commits
            if commit.temporal_status in {"stale", "expired", "unknown"}
        )
        conflict_counts = _count_conflict_types(conflicts)
        direct_count = conflict_counts.get("direct_conflict", 0)
        risk_constraint_count = conflict_counts.get("risk_constraint", 0)
        temporal_warning_count = conflict_counts.get("temporal_warning", 0)
        evidence_gap_count = conflict_counts.get("evidence_gap", 0)

        if direct_count:
            conflict_sentence = (
                f"{direct_count} direct conflicts were detected, alongside "
                f"{risk_constraint_count} risk constraints, "
                f"{temporal_warning_count} temporal warnings, and "
                f"{evidence_gap_count} evidence gaps."
            )
        else:
            conflict_sentence = (
                "No strong direct contradiction was detected, but "
                f"{risk_constraint_count} risk constraints, "
                f"{temporal_warning_count} temporal warnings, and "
                f"{evidence_gap_count} evidence gaps reduce confidence."
            )

        return (
            f"The final recommendation is {final_recommendation} because positive "
            "growth, profitability, financial health, and momentum evidence were "
            "weighed against valuation risk, volatility risk, stale technical "
            "evidence, and evidence gaps. "
            f"Support score is {self.support_score}, risk score is {self.risk_score}, "
            f"and opposition score is {self.opposition_score}. "
            "Supporting evidence was compared against risk and opposing claims "
            "after discounting stale, expired, or unknown evidence. "
            f"{stale_count} of {len(commits)} commits have stale, expired, or "
            "unknown temporal status, which reduces confidence in time-sensitive "
            f"claims. {conflict_sentence} Risk adjustment was applied where "
            "cautionary evidence constrained the supporting thesis."
        )

    def _dedupe_commits(
        self,
        commits: list[ClaimEvidenceCommit],
    ) -> list[ClaimEvidenceCommit]:
        """Remove duplicate commits while preserving order."""

        seen_commit_ids: set[str] = set()
        deduped: list[ClaimEvidenceCommit] = []
        for commit in commits:
            if commit.commit_id in seen_commit_ids:
                continue
            seen_commit_ids.add(commit.commit_id)
            deduped.append(commit)
        return deduped

    def _select_key_conflicts(self, conflicts: list[Conflict]) -> list[Conflict]:
        """Keep the most useful conflicts for the audit trail."""

        severity_rank = {"high": 0, "medium": 1, "low": 2}
        type_rank = {
            "direct_conflict": 0,
            "risk_constraint": 1,
            "temporal_warning": 2,
            "evidence_gap": 3,
        }
        sorted_conflicts = sorted(
            conflicts,
            key=lambda conflict: (
                severity_rank.get(conflict.severity, 3),
                type_rank.get(conflict.conflict_type, 9),
            ),
        )

        selected: list[Conflict] = []
        type_counts: dict[str, int] = {}
        seen_pairs: set[tuple[str, str]] = set()
        for conflict in sorted_conflicts:
            pair_key = (conflict.claim_a, conflict.claim_b)
            if pair_key in seen_pairs:
                continue
            if type_counts.get(conflict.conflict_type, 0) >= 2:
                continue
            seen_pairs.add(pair_key)
            selected.append(conflict)
            type_counts[conflict.conflict_type] = (
                type_counts.get(conflict.conflict_type, 0) + 1
            )
            if len(selected) >= 8:
                break
        return selected


def _contains_any(text: str, keywords: set[str]) -> bool:
    """Return whether text contains any keyword with word-aware matching."""

    lowered_text = _normalize_match_text(text)
    for keyword in keywords:
        lowered_keyword = _normalize_match_text(keyword)
        if _should_use_substring_match(lowered_keyword):
            if lowered_keyword in lowered_text:
                return True
            continue
        if " " in lowered_keyword:
            if lowered_keyword in lowered_text:
                return True
            continue
        if re.search(rf"\b{re.escape(lowered_keyword)}\b", lowered_text):
            return True
    return False


def _is_supporting_commit(commit: ClaimEvidenceCommit) -> bool:
    """Classify a commit as supporting while avoiding mixed-risk false positives."""

    claim_has_positive_signal = _contains_any(commit.claim, POSITIVE_KEYWORDS)
    claim_or_tag_has_negative_signal = _contains_any(
        f"{commit.claim} {commit.risk_tag}",
        NEGATIVE_KEYWORDS | RISK_TAG_KEYWORDS,
    )
    is_clean_bull_case = (
        commit.branch_name == "bull-case"
        and not _contains_any(commit.risk_tag, RISK_TAG_KEYWORDS)
    )
    return claim_has_positive_signal and (
        not claim_or_tag_has_negative_signal or is_clean_bull_case
    )


def _is_evidence_gap_commit(commit: ClaimEvidenceCommit) -> bool:
    """Return whether a commit represents missing validation or reasoning risk."""

    if _contains_any(commit.risk_tag, EVIDENCE_GAP_TAGS):
        if not _contains_any(commit.risk_tag, {"counter_evidence"}):
            return True

    if not _contains_any(commit.risk_tag, {"counter_evidence"}):
        return False

    return _contains_any(
        commit.claim,
        {
            "validation",
            "validate",
            "requires",
            "checked",
            "evaluated",
            "explicit",
            "missing",
            "counter evidence",
        },
    )


def _themes_for_commit(commit: ClaimEvidenceCommit) -> set[str]:
    """Infer simple themes from a commit claim, risk tag, and evidence."""

    text = _commit_text(commit)
    themes: set[str] = set()
    for theme, keywords in THEME_KEYWORDS.items():
        if _contains_any(text, keywords):
            themes.add(theme)
    return themes


def _commit_text(commit: ClaimEvidenceCommit) -> str:
    """Return normalized searchable text for a commit."""

    return (
        f"{commit.claim} {commit.risk_tag} {commit.temporal_status or ''} "
        f"{commit.evidence.content} {commit.evidence.metric_name or ''}"
    )


def _count_conflict_types(conflicts: list[Conflict]) -> dict[str, int]:
    """Count conflicts by their conflict_type."""

    counts: dict[str, int] = {}
    for conflict in conflicts:
        counts[conflict.conflict_type] = counts.get(conflict.conflict_type, 0) + 1
    return counts


def _clamp_unit(value: float) -> float:
    """Clamp a numeric score to the 0-1 interval."""

    return max(0.0, min(1.0, value))


def _clamp_score(value: float) -> float:
    """Clamp and round a score to the 0-100 interval."""

    return round(max(0.0, min(100.0, value)), 1)


def _percent(value: float) -> float:
    """Convert a 0-1 score to a 0-100 score."""

    return _clamp_score(100 * value)


def _position_sizing_suggestion(
    recommendation: str,
    risk_level: str,
    conviction_score: float,
) -> str:
    """Translate score posture into a conservative sizing suggestion."""

    if recommendation == "Buy" and risk_level == "low" and conviction_score >= 70:
        return "Core starter position, up to 5-8% of portfolio"
    if recommendation == "Buy":
        return "Small starter position, up to 2-4% of portfolio"
    if recommendation == "Hold":
        return "Watchlist or existing-position hold; new entry only on better setup"
    if recommendation == "Sell":
        return "Reduce or avoid adding exposure"
    return "Observation only; avoid new exposure"


def _normalize_match_text(text: str) -> str:
    """Normalize separators so risk tags and prose match the same keywords."""

    return text.lower().replace("_", " ").replace("-", " ")


def _should_use_substring_match(keyword: str) -> bool:
    """Use substring matching for Chinese and other non-word keywords."""

    return any(ord(character) > 127 for character in keyword)
