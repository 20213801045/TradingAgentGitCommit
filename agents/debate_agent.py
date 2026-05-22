"""LLM-driven debate agent — replaces keyword-matching merge with dialectical reasoning.

Instead of classifying commits by keyword and averaging scores, this agent:
1.  Gathers all evidence from the workspace branches
2.  Organises it into bull / bear / neutral briefs
3.  Asks the LLM to *debate* both sides and produce an actionable decision
4.  Falls back to a lightweight evidence-weighted vote when no LLM is available
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from llm.base import BaseLLMClient, LLMError, LLMMessage
from models.schemas import (
    ClaimEvidenceCommit,
    Conflict,
    DecisionScores,
    MergeResult,
    Workspace,
)


# ── structured output the LLM is asked to return ──────────────────────────
DEBATE_JSON_SCHEMA = """{
  "recommendation": "Buy|Sell|Hold",
  "confidence": "high|medium|low",
  "confidence_reason": "why this confidence level",
  "entry_price_low": number or null,
  "entry_price_high": number or null,
  "stop_loss": number or null,
  "target_1": number or null,
  "target_2": number or null,
  "position_size_pct": number or null,
  "time_horizon": "e.g. 3-6 months",
  "bull_case_strongest": "single strongest bull argument",
  "bear_case_strongest": "single strongest bear argument",
  "debate_verdict": "which side won the debate and why",
  "key_risk": "the #1 risk that could invalidate the bull case",
  "decision_rationale": "2-3 sentence summary of the reasoning",
  "conditions_to_revise": ["condition 1", "condition 2", "condition 3"],
  "risk_level": "low|medium|high"
}"""


# ── system prompt that forces the LLM into a debate posture ───────────────
DEBATE_SYSTEM_PROMPT = """You are an elite investment committee of three members:
- A seasoned **Bull** portfolio manager who finds reasons to invest
- A skeptical **Bear** risk manager who stress-tests every assumption
- A **Chair** who weighs both sides and makes the final call

Your job: debate the evidence below and produce ONE actionable decision.
RULES:
1. You MUST pick a side — Buy, Sell, or Hold.  No "Hold/Observe" cop-outs.
2. If the bull case is clearly stronger → Buy.  If bear is stronger → Sell.
   If the evidence is truly mixed BUT one side has better quality evidence → lean that way.
3. Provide SPECIFIC numbers for entry, stop-loss, and targets.
   Use the current price and volatility context to set realistic levels.
4. Be honest about what you DON'T know — flag missing evidence.
5. The output MUST be valid JSON matching the schema exactly.
6. Think in probabilities, not certainties.  A 60%-confident Buy is better than
   a 100%-confident "Hold" that says nothing."""


class DebateAgent:
    """LLM-powered debate agent that replaces the deterministic MergeAgent."""

    def __init__(self, llm_client: BaseLLMClient | None = None, tracker: Any = None) -> None:
        self.llm_client = llm_client
        self.tracker = tracker

    # ── public API (same signature as MergeAgent.merge) ───────────────────
    def debate(self, workspace: Workspace) -> MergeResult:
        """Run the debate and return a MergeResult."""

        all_commits = self._collect_commits(workspace)

        if self.llm_client is not None:
            try:
                return self._llm_debate(workspace, all_commits)
            except (LLMError, json.JSONDecodeError, ValueError):
                pass  # fall through to lightweight fallback

        return self._fallback_decision(workspace, all_commits)

    # ── evidence collection ───────────────────────────────────────────────
    def _collect_commits(self, workspace: Workspace) -> list[ClaimEvidenceCommit]:
        commits: list[ClaimEvidenceCommit] = []
        for branch in workspace.branches.values():
            commits.extend(branch.commits)
        return commits

    # ── LLM debate ────────────────────────────────────────────────────────
    def _llm_debate(
        self,
        workspace: Workspace,
        commits: list[ClaimEvidenceCommit],
    ) -> MergeResult:
        """Send evidence to the LLM and parse the debate verdict."""

        prompt = self._build_debate_prompt(workspace, commits)
        response = self.llm_client.complete(  # type: ignore[union-attr]
            [
                LLMMessage(role="system", content=DEBATE_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.3,  # slight creativity for debate nuance
            response_format={"type": "json_object"},
        )
        verdict = json.loads(response.content)
        return self._verdict_to_merge_result(verdict, commits)

    def _build_debate_prompt(
        self,
        workspace: Workspace,
        commits: list[ClaimEvidenceCommit],
    ) -> str:
        """Build a structured debate brief from all workspace evidence."""

        # separate evidence by posture
        bull_commits = [c for c in commits if self._is_bullish(c)]
        bear_commits = [c for c in commits if self._is_bearish(c)]
        neutral_commits = [
            c for c in commits if not self._is_bullish(c) and not self._is_bearish(c)
        ]

        # extract price if available
        current_price = self._extract_price(commits)

        parts: list[str] = []

        # header
        company = workspace.company_name or workspace.ticker
        parts.append(
            f"# INVESTMENT DEBATE BRIEF\n"
            f"**Ticker:** {workspace.ticker}\n"
            f"**Company:** {company}\n"
            f"**Research Question:** {workspace.research_question}\n"
            f"**Current Price:** {current_price or 'unknown'}\n"
            f"**Total Evidence Items:** {len(commits)} "
            f"(🟢 Bull: {len(bull_commits)} | 🔴 Bear: {len(bear_commits)} | "
            f"⚪ Neutral: {len(neutral_commits)})\n"
        )

        # bull case
        parts.append("## 🟢 BULL CASE")
        if bull_commits:
            for i, c in enumerate(bull_commits, 1):
                parts.append(self._format_evidence(i, c))
        else:
            parts.append("(No explicit bull evidence found.)")

        # bear case
        parts.append("## 🔴 BEAR CASE")
        if bear_commits:
            for i, c in enumerate(bear_commits, 1):
                parts.append(self._format_evidence(i, c))
        else:
            parts.append("(No explicit bear evidence found.)")

        # neutral context
        if neutral_commits:
            parts.append("## ⚪ NEUTRAL CONTEXT")
            for i, c in enumerate(neutral_commits, 1):
                parts.append(self._format_evidence(i, c))

        # ── Closed-loop: inject prediction history ──────────────────
        if self.tracker:
            feedback = self.tracker.feedback_string(ticker=workspace.ticker)
            if feedback:
                parts.append(feedback)

        # instructions
        parts.append(
            "## YOUR TASK\n"
            "Debate the bull and bear cases above.  Then produce a JSON verdict "
            "with this EXACT schema:\n"
            f"{DEBATE_JSON_SCHEMA}\n\n"
            "IMPORTANT:\n"
            "- Use the current price to set realistic entry/stop/target levels.\n"
            "- If price is unknown, set numeric fields to null.\n"
            "- Write in English (the report layer will translate to Chinese).\n"
            "- Be specific.  'Buy' without price levels is useless."
        )

        return "\n\n".join(parts)

    @staticmethod
    def _format_evidence(index: int, commit: ClaimEvidenceCommit) -> str:
        """Format one evidence item for the debate brief."""

        quality = f"quality={commit.evidence_quality_score:.1f}" if commit.evidence_quality_score else ""
        temporal = commit.temporal_status or "unknown"
        return (
            f"**{index}. [{commit.branch_name}]** {commit.claim}\n"
            f"   Evidence: {commit.evidence.content}\n"
            f"   Source: {commit.evidence.source} | {commit.evidence.source_type}\n"
            f"   Confidence: {commit.confidence} | Horizon: {commit.time_horizon}\n"
            f"   Quality: {quality} | Temporal: {temporal}\n"
        )

    @staticmethod
    def _extract_price(commits: list[ClaimEvidenceCommit]) -> str | None:
        """Try to extract the current stock price from evidence."""

        for c in commits:
            content = f"{c.evidence.content} {c.evidence.metric_name or ''} {c.claim}"
            for marker in ("current price", "latest price", "股价", "现价", "close"):
                if marker in content.lower():
                    # crude extraction — LLM can still reason from context
                    return "see evidence context"
        return None

    # ── classification helpers ────────────────────────────────────────────
    @staticmethod
    def _is_bullish(commit: ClaimEvidenceCommit) -> bool:
        """Simple heuristic: is this commit primarily bullish?"""

        branch = commit.branch_name
        if branch in {"bull-case"}:
            return True
        if branch in {"bear-case", "counter-evidence", "risk-review"}:
            return False

        text = f"{commit.claim} {commit.risk_tag}".lower()
        bull_markers = [
            "growth", "strong", "positive", "support", "bullish",
            "undervalued", "attractive", "momentum", "improving",
            "增长", "强劲", "支持", "看多", "低估", "改善",
        ]
        bear_markers = [
            "risk", "weak", "declining", "overvalued", "expensive",
            "volatile", "uncertainty", "downside", "bearish", "caution",
            "风险", "下跌", "高估", "波动", "不确定", "看空",
        ]
        bull_score = sum(1 for m in bull_markers if m in text)
        bear_score = sum(1 for m in bear_markers if m in text)
        return bull_score > bear_score

    @staticmethod
    def _is_bearish(commit: ClaimEvidenceCommit) -> bool:
        """Simple heuristic: is this commit primarily bearish?"""

        branch = commit.branch_name
        if branch in {"bear-case", "counter-evidence"}:
            return True
        if branch in {"bull-case"}:
            return False

        text = f"{commit.claim} {commit.risk_tag}".lower()
        bear_markers = [
            "risk", "weak", "declining", "overvalued", "expensive",
            "volatile", "uncertainty", "downside", "bearish", "caution",
            "sell", "avoid", "风险", "下跌", "高估", "波动", "不确定",
        ]
        bull_markers = [
            "growth", "strong", "positive", "support", "bullish",
            "undervalued", "attractive", "momentum",
        ]
        bear_score = sum(1 for m in bear_markers if m in text)
        bull_score = sum(1 for m in bull_markers if m in text)
        return bear_score > bull_score

    # ── verdict → MergeResult ─────────────────────────────────────────────
    def _verdict_to_merge_result(
        self,
        verdict: dict[str, Any],
        commits: list[ClaimEvidenceCommit],
    ) -> MergeResult:
        """Convert the LLM debate verdict into the standard MergeResult."""

        rec = verdict.get("recommendation", "Hold")
        conf = verdict.get("confidence", "medium")

        # build richer rationale by combining debate fields
        rationale_parts = [
            f"Bull case strongest point: {verdict.get('bull_case_strongest', 'N/A')}",
            f"Bear case strongest point: {verdict.get('bear_case_strongest', 'N/A')}",
            f"Debate verdict: {verdict.get('debate_verdict', 'N/A')}",
            f"#1 risk: {verdict.get('key_risk', 'N/A')}",
        ]
        decision_rationale = verdict.get("decision_rationale", "") or " | ".join(rationale_parts)

        # translate recommendation to entry / conviction scores
        entry_score = self._entry_score_from_verdict(rec, conf, verdict)
        conviction_score = self._conviction_from_confidence(conf)
        # compute directional conviction from bull/bear weights
        bull_claims = [c.claim for c in commits if self._is_bullish(c)]
        bear_claims = [c.claim for c in commits if self._is_bearish(c)]
        bull_weight = sum(c.evidence_quality_score or 0.5 for c in commits if self._is_bullish(c))
        bear_weight = sum(c.evidence_quality_score or 0.5 for c in commits if self._is_bearish(c))
        total_weight = bull_weight + bear_weight or 1.0
        bull_pct = bull_weight / total_weight * 100

        # build position sizing suggestion
        sizing = self._build_sizing(rec, conf, verdict)

        return MergeResult(
            final_recommendation=rec,
            confidence=conf,
            decision_scores=DecisionScores(
                directional_conviction=float(max(bull_pct, 100 - bull_pct) * 0.6),
                entry_timing=entry_score,
                risk_level=verdict.get("risk_level", "medium"),
                position_sizing_suggestion=sizing,
            ),
            main_supporting_claims=bull_claims[:5],
            main_opposing_claims=bear_claims[:5],
            key_conflicts=self._build_conflicts_from_verdict(verdict, commits),
            risk_adjustment=(
                f"Key risk factor: {verdict.get('key_risk', 'none identified')}. "
                f"Confidence: {conf}. {verdict.get('confidence_reason', '')}"
            ),
            decision_rationale=decision_rationale,
            conditions_for_revision=verdict.get("conditions_to_revise", []),
        )

    @staticmethod
    def _entry_score_from_verdict(
        rec: str,
        conf: str,
        verdict: dict[str, Any],
    ) -> float:
        """Derive an entry score (0-100) from the debate verdict."""

        base = {"Buy": 70, "Hold": 45, "Sell": 20}.get(rec, 45)
        conf_adj = {"high": 15, "medium": 0, "low": -15}.get(conf, 0)

        # boost if the LLM provided concrete entry levels
        has_entry = verdict.get("entry_price_low") is not None
        has_target = verdict.get("target_1") is not None
        specificity_bonus = 10 if (has_entry and has_target) else 0

        return max(0.0, min(100.0, base + conf_adj + specificity_bonus))

    @staticmethod
    def _conviction_from_confidence(conf: str) -> float:
        return {"high": 80.0, "medium": 55.0, "low": 30.0}.get(conf, 50.0)

    @staticmethod
    def _risk_reward_from_verdict(verdict: dict[str, Any], rec: str) -> float:
        """Estimate risk/reward score from entry, stop, target."""

        entry_high = verdict.get("entry_price_high")
        stop = verdict.get("stop_loss")
        target = verdict.get("target_1")

        if None in (entry_high, stop, target) or rec == "Hold":
            return 50.0

        try:
            reward = (float(target) - float(entry_high)) / float(entry_high)
            risk = (float(entry_high) - float(stop)) / float(entry_high)
            if risk <= 0:
                return 30.0
            rr = reward / risk
            # map rough RR to score: RR=1→50, RR=2→70, RR=3→85
            return max(10.0, min(95.0, 50 + (rr - 1) * 20))
        except (ZeroDivisionError, ValueError):
            return 50.0

    @staticmethod
    def _build_sizing(rec: str, conf: str, verdict: dict[str, Any]) -> str:
        """Build a concrete position sizing suggestion."""

        pct = verdict.get("position_size_pct")
        if pct is not None:
            pct_str = f"{float(pct):.0f}%"
        else:
            pct_str = "2-4%" if rec == "Buy" else "0%"

        if rec == "Buy" and conf == "high":
            return f"Core position, {pct_str} of portfolio"
        if rec == "Buy":
            return f"Starter position, {pct_str} of portfolio"
        if rec == "Hold":
            return "Watchlist only; no new entry without better setup"
        return "Reduce or avoid; no new exposure"

    def _build_conflicts_from_verdict(
        self,
        verdict: dict[str, Any],
        commits: list[ClaimEvidenceCommit],
    ) -> list[Conflict]:
        """Build conflict records from the debate's bull/bear tension."""

        conflicts: list[Conflict] = []

        # the fundamental conflict: bull vs bear strongest points
        bull_point = verdict.get("bull_case_strongest", "")
        bear_point = verdict.get("bear_case_strongest", "")
        if bull_point and bear_point:
            conflicts.append(
                Conflict(
                    conflict_id=str(uuid4()),
                    conflict_type="bull_vs_bear",
                    claim_a=bull_point,
                    claim_b=bear_point,
                    explanation=verdict.get("debate_verdict", ""),
                    severity=verdict.get("risk_level", "medium"),
                )
            )

        # flag stale evidence as temporal warnings
        for c in commits:
            if c.temporal_status in {"stale", "expired"}:
                conflicts.append(
                    Conflict(
                        conflict_id=str(uuid4()),
                        conflict_type="temporal_warning",
                        claim_a=c.claim,
                        claim_b=f"Evidence may be {c.temporal_status} for horizon {c.time_horizon}",
                        explanation="Stale evidence reduces confidence in time-sensitive conclusions.",
                        severity="low" if c.temporal_status == "stale" else "medium",
                    )
                )
                if len(conflicts) >= 6:
                    break

        return conflicts

    # ── fallback (no LLM) ─────────────────────────────────────────────────
    def _fallback_decision(
        self,
        workspace: Workspace,
        commits: list[ClaimEvidenceCommit],
    ) -> MergeResult:
        """Lightweight fallback when no LLM is available.

        Much simpler than the old MergeAgent — just counts bull vs bear
        and weights by evidence quality.  No keyword-matching spaghetti.
        """

        if not commits:
            return MergeResult(
                final_recommendation="Hold",
                confidence="low",
                decision_scores=DecisionScores(position_sizing_suggestion="Insufficient data"),
                main_supporting_claims=[],
                main_opposing_claims=[],
                key_conflicts=[],
                risk_adjustment="No evidence available.",
                decision_rationale="Cannot make a decision without evidence.",
                conditions_for_revision=["Gather fundamental, technical, and valuation data."],
            )

        bull_commits = [c for c in commits if self._is_bullish(c)]
        bear_commits = [c for c in commits if self._is_bearish(c)]

        # weight by evidence quality
        bull_weight = sum(c.evidence_quality_score or 0.5 for c in bull_commits)
        bear_weight = sum(c.evidence_quality_score or 0.5 for c in bear_commits)
        total_weight = bull_weight + bear_weight or 1.0

        bull_pct = bull_weight / total_weight * 100
        bear_pct = bear_weight / total_weight * 100

        if bull_pct > 65:
            rec, conf = "Buy", "medium"
        elif bear_pct > 65:
            rec, conf = "Sell", "medium"
        elif bull_pct > bear_pct + 10:
            rec, conf = "Buy", "low"
        elif bear_pct > bull_pct + 10:
            rec, conf = "Sell", "low"
        else:
            rec, conf = "Hold", "low"

        stale_count = sum(1 for c in commits if c.temporal_status in {"stale", "expired"})
        if stale_count / max(len(commits), 1) > 0.4:
            conf = "low"

        company = workspace.company_name or workspace.ticker
        return MergeResult(
            final_recommendation=rec,
            confidence=conf,
            decision_scores=DecisionScores(
                directional_conviction=round(max(bull_pct, bear_pct) * 0.6, 1),
                entry_timing=round(bull_pct * 0.7, 1),
                risk_level="high" if bear_pct > 60 else ("medium" if bear_pct > 35 else "low"),
                position_sizing_suggestion=(
                    "Starter position, 2-4% of portfolio" if rec == "Buy" else "Watchlist only"
                ),
            ),
            main_supporting_claims=[c.claim for c in bull_commits[:5]],
            main_opposing_claims=[c.claim for c in bear_commits[:5]],
            key_conflicts=[],
            risk_adjustment=f"Fallback mode (no LLM). Bull weight: {bull_pct:.0f}%, Bear: {bear_pct:.0f}%.",
            decision_rationale=(
                f"{company}: {len(bull_commits)} bullish vs {len(bear_commits)} bearish "
                f"evidence items. Weighted bull/bear ratio = {bull_pct:.0f}/{bear_pct:.0f}. "
                f"Recommendation: {rec} (confidence: {conf}). "
                "Upgrade to LLM-powered debate for higher quality decisions."
            ),
            conditions_for_revision=[
                "Enable LLM for richer debate and actionable price targets.",
                "Refresh stale evidence.",
                "Add more granular valuation and technical data.",
            ],
        )


# ── convenience function to drop into existing pipeline ────────────────────
def debate_workspace(workspace: Workspace, llm_client: BaseLLMClient | None = None) -> MergeResult:
    """One-line replacement for MergeAgent().merge(workspace)."""
    return DebateAgent(llm_client).debate(workspace)
