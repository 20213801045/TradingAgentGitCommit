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


# ── structured output the LLM is asked to return ─────────────────────────DEBATE_JSON_SCHEMA = ""{
  \"recommendation\": \"Buy|Sell|Hold\",
  \"confidence\": \"high|medium|low\",
  \"confidence_reason\": \"why this confidence level\",
  \"entry_price_low\": number or null,
  \"entry_price_high\": number or null,
  \"stop_loss\": number or null,
  \"target_1\": number or null,
  \"target_2\": number or null,
  \"position_size_pct\": number or null,
  \"time_horizon\": \"e.g. 3-6 months\",
  \"bull_case_strongest\": \"single strongest bull argument\",
  \"bear_case_strongest\": \"single strongest bear argument\",
  \"debate_verdict\": \"which side won the debate and why\",
  \"key_risk\": \"the #1 risk that could invalidate the bull case\",
  \"decision_rationale\": \"2-3 sentence summary of the reasoning\",
  \"conditions_to_revise\": [\"condition 1\", \"condition 2\", \"condition 3\"],
  \"risk_level\": \"low|medium|high\"
}"""


# ── system prompt that forces the LLM into a debate posture ──────────────
DEBATE_SYSTEM_PROMPT = ""You are an elite investment committee of three members:\n- A seasoned **Bull** portfolio manager who finds reasons to invest\n- A skeptical **Bear** risk manager who stress-tests every assumption\n- A **Chair** who weighs both sides and makes the final call\n\nYour job: debate the evidence below and produce ONE actionable decision.\nRULES:\n1. You MUST pick a side — Buy, Sell, or Hold.  No \"Hold/Observe\" cop-outs.\n2. If the bull case is clearly stronger → Buy.  If bear is stronger → Sell.\n   If the evidence is truly mixed BUT one side has better quality evidence → lean that way.\n3. Provide SPECIFIC numbers for entry, stop-loss, and targets.\n   Use the current price and volatility context to set realistic levels.\n4. Adjust position size based on confidence and conviction.\n5. Output ONLY VALID JSON — no extra text, no explanation outside the JSON."