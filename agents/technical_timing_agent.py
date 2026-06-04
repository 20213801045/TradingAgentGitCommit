"""LLM-backed technical timing agent — analyzes trend, momentum, volatility, and timing signals."""

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Evidence
from llm.base import LLMError, LLMMessage

import json
from datetime import datetime, timezone
from uuid import uuid4


class TechnicalTimingAgent(BaseAgent):
    """AN LLM-backed price trend, volatility, and timing signal analyst."""

    role = "technical-timing"

    def analyze(self, input_data: dict, workspace: "Workspace") -> list[ClaimEvidenceCommit]:
        """Run an LLM-backed technical timing analysis or fallback."""

        ticker = input_data.get("ticker", "UNKNOWN")
        market = input_data.get("market_data", {})

        if self.llm_client:
            try:
                prompt = f"""Analyze the price trend, momentum, volatility, and timing for {ticker}\n                    Market Data: {json.dumps(market)}\n                    Provide 1-2 technical insights."""
                response = self.llm_client.complete([
                    LLMMessage(role="system", content=prompt),
                ])
                claim = response.content
            except LLMError:
                claim = f"Technical data unavailable for {ticker}"
        else:
            claim = f"Technical data unavailable for {ticker}"

        return [self._make_commit(
            branch_name="technical-analysis",
            claim=claim[:500],
            evidence=Evidence(
                evidence_id=uuid4().hex[:8],
                content=claim[:500],
                source=f"ticker:{ticker}",
                source_type="llm",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            confidence="medium",
            risk_tag="adrift",
            time_horizon="3-6 months",
        )]
