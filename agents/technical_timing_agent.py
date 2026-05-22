"""LLM-first technical & timing agent — replaces technical_agent + backtest_agent."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from llm.base import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


class TechInsight(BaseModel):
    dimension: str = Field(description="trend|momentum|volatility|support_resistance|timing")
    claim: str = Field(min_length=15)
    confidence: str = "medium"
    risk_tag: str = "technical"
    time_horizon: str = "1-3 months"


class TechnicalOutput(BaseModel):
    insights: list[TechInsight] = Field(min_length=3, max_length=6)


SYSTEM_PROMPT = """You are a technical analyst at a trading desk. Analyze the price/indicator data.
Produce 3-6 insights covering: trend (moving averages), momentum (RSI/MACD), volatility, support/resistance, and entry timing.
Be specific: "RSI at 69.5 suggests..." not "momentum seems OK".
If data is missing, note the gap.
Return valid JSON: {"insights":[{"dimension":"trend|momentum|volatility|support_resistance|timing","claim":"...","confidence":"low|medium|high","risk_tag":"...","time_horizon":"..."}]}"""


class TechnicalTimingAgent(BaseAgent):
    """LLM-first technical analysis with timing signals."""

    name = "TechnicalTimingAgent"
    role = "technical-timing-agent"
    branch_name = "technical-analysis"

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
        market = input_data.get("market_data", {})
        indicators = input_data.get("technical_indicators", {})
        timestamp = str(input_data.get("as_of_date", "2026-05-22"))

        all_data = {**market, **indicators}
        prompt = (
            f"Ticker: {workspace.ticker}\n"
            "Technical data:\n" + "\n".join(f"- {k}: {v}" for k, v in sorted(all_data.items())) +
            "\n\nProduce 3-6 technical insights. Be quantitative. Return JSON."
        )
        response = self.llm_client.complete(  # type: ignore[union-attr]
            [LLMMessage(role="system", content=SYSTEM_PROMPT), LLMMessage(role="user", content=prompt)],
            temperature=0.1, response_format={"type": "json_object"},
        )
        parsed = parse_json_response(response.content)
        validated = TechnicalOutput.model_validate(parsed)

        commits: list[ClaimEvidenceCommit] = []
        for ins in validated.insights:
            commits.append(self.create_commit(
                claim=ins.claim.strip(),
                evidence=self._make_evidence(content=ins.claim, source="Technical Analysis (LLM)", source_type="llm_technical", timestamp=timestamp, metric_name=ins.dimension, metric_value=ins.dimension),
                confidence=ins.confidence, risk_tag=ins.risk_tag, time_horizon=ins.time_horizon,
            ))
        return commits

    def _fallback(self, input_data: dict[str, Any]) -> list[ClaimEvidenceCommit]:
        timestamp = str(input_data.get("as_of_date", "2026-05-22"))
        return [self.create_commit(
            claim="Technical analysis unavailable — enable LLM for real technical insights.",
            evidence=self._make_evidence(content="No LLM available for technical analysis.", source="System", source_type="system", timestamp=timestamp, metric_name="technical", metric_value="unavailable"),
            confidence="low", risk_tag="technical_evidence_gap", time_horizon="N/A",
        )]
