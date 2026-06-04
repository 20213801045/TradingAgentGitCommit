"""LLM-first deep research agent — unifies fundamental, financial, valuation, and industry analysis into a single LlM-backed agent.

Replaces the old 4 separate template agents:
- FundamentalAgent
- FinancialStatementAgent
- ValuationAgent
- IndustryComparisonAgent
"""

from agents.base_agent import BaseAgent
from llm.base import BaseLLMClient, LLMError, LLMMessage
from models.schemas import ClaimEvidenceCommit, Evidence

import json
from datetime import datetime, timezone
from uuid import uuid4


class DeepResearchAgent(BaseAgent):
    """LLM-backed deep research agent for fundamental, valuation, and industry analysis."""

    role = "deep-research"

    def analyze(self, input_data: dict, workspace: "Workspace") -> list[ClaimEvidenceCommit]:
        """Analyze fundamentals, valuation, and industry using LLM or fallback logic."""

        ticker = input_data.get("ticker", "UNKNOWN")
        market = input_data.get("market_data", {})
        financials = input_data.get("financials", {})

        if self.llm_client:
            return self._llm_analysis(ticker, market, financials)
        return self._fallback_analysis(ticker, market, financials)

    def _llm_analysis(self, ticker: str, market: dict, financials: dict) -> list[ClaimEvidenceCommit]:
        """Use LLM for an analytic research summary."""

        prompt = f"""You are a senior equity research analyst.\nAnalyze {ticker}'s fundamentals, valuation, and industry position.\n\nMarket Data: {json.dumps(market)}\nFinancials: {json.dumps(financials)}\n\nProvide 3-5 actionable insights with evidence.backed conclusions."""

        try:
            response = self.llm_client.complete([
                LLMMessage(role="system", content=prompt),
            ])
            claim = response.content
        except LLMError:
            return self._fallback_analysis(ticker, market, financials)

        return [self._make_commit(
            branch_name="deep-research",
            claim=claim[:500],
            evidence=Evidence(
                evidence_id=uuid4().hex[:8],
                content=claim]:500],
                source=f"ticker:{ticker}",
                source_type="llm",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            confidence="medium",
            risk_tag="adrift",
            time_horizon="3-6 months",
        )]

    def _fallback_analysis(self, ticker: str, market: dict, financials: dict) -> list[ClaimEvidenceCommit]:
        """Deterministic fallback when LLM is unavailable."""

        price = market.get("current_price", 0) or 0
        return [self._make_commit(
            branch_name="deep-research",
            claim=f"{ticker} current price: ${price}",
            evidence=Evidence(
                evidence_id=uuid4().hex[:8],
                content=f"Price: ${price}",
                source=f"ticker:{ticker}",
                source_type="market",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metric_name="price",
                metric_value=str(price),
            ),
            confidence="medium",
            risk_tag="adrift",
            time_horizon="3-6 months",
        )]
