"""LLM-first macro & sentiment agent — replaces old template macro_agent."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from llm.base import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


class MacroInsight(BaseModel):
    dimension: str = "macro"
    claim: str = Field(min_length=15)
    confidence: str = "medium"
    risk_tag: str = "macro_context"
    time_horizon: str = "6-12 months"


class MacroOutput(BaseModel):
    insights: list[MacroInsight] = Field(min_length=2, max_length=5)


SYSTEM_PROMPT = """You are a macro strategist. Analyze the macro context and produce 2-4 insights.
Focus on: rate environment impact, demand backdrop, inflation pressure, currency exposure.
For each: say whether it helps or hurts the investment case, and by how much.
Be honest if data is insufficient — flag gaps instead of guessing.
Return valid JSON: {"insights":[{"dimension":"macro","claim":"...","confidence":"low|medium|high","risk_tag":"...","time_horizon":"..."}]}"""


class MacroSentimentAgent(BaseAgent):
    """LLM-first macro & sentiment analysis."""

    name = "MacroSentimentAgent"
    role = "macro-sentiment-agent"
    branch_name = "macro-analysis"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(self, input_data: dict[str, Any], workspace: Workspace) -> list[ClaimEvidenceCommit]:
        if self.llm_client is not None:
            try:
                return self._llm_analyze(input_data, workspace)
            except (LLMError, json.JSONDecodeError, ValidationError):
                pass
        return self._fallback(input_data)

    def _llm_analyze(self, input_data: dict[str, Any], workspace: Workspace) -> list[ClaimEvidenceCommit]:
        macro = input_data.get("macro_context", {})
        timestamp = str(input_data.get("as_of_date", "2026-05-22"))
        prompt = (
            f"Ticker: {workspace.ticker} | Company: {workspace.company_name or 'unknown'}\n"
            f"Macro context:\n" + "\n".join(f"- {k}: {v}" for k, v in sorted(macro.items())) +
            "\n\nProduce 2-4 macro insights. Flag data gaps. Return JSON."
        )
        response = self.llm_client.complete(  # type: ignore[union-attr]
            [LLMMessage(role="system", content=SYSTEM_PROMPT), LLMMessage(role="user", content=prompt)],
            temperature=0.1, response_format={"type": "json_object"},
        )
        parsed = parse_json_response(response.content)
        validated = MacroOutput.model_validate(parsed)

        commits: list[ClaimEvidenceCommit] = []
        for ins in validated.insights:
            commits.append(self.create_commit(
                claim=ins.claim.strip(),
                evidence=self._make_evidence(content=ins.claim, source="Macro Analysis (LLM)", source_type="llm_macro", timestamp=timestamp, metric_name="macro", metric_value=ins.dimension),
                confidence=ins.confidence, risk_tag=ins.risk_tag, time_horizon=ins.time_horizon,
            ))
        return commits

    def _fallback(self, input_data: dict[str, Any]) -> list[ClaimEvidenceCommit]:
        timestamp = str(input_data.get("as_of_date", "2026-05-22"))
        return [self.create_commit(
            claim="Macro analysis unavailable — enable LLM for macro context assessment.",
            evidence=self._make_evidence(content="No LLM available for macro analysis.", source="System", source_type="system", timestamp=timestamp, metric_name="macro", metric_value="unavailable"),
            confidence="low", risk_tag="macro_evidence_gap", time_horizon="N/A",
        )]
