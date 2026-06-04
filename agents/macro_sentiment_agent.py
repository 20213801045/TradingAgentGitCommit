"""LLM-backed macro sentiment agent — analyses macro environment, rates, and market sentiment."""

from agents.base_agent import BaseAgent
from models.schemas import ClaimEvidenceCommit, Evidence
from llm.base import LLMError, LLMMessage

import json
from datetime import datetime, timezone
from uuid import uuid4


class MacroSentimentAgent(BaseAgent):
    """AN LLM-backed macro environment and sentiment analyst."""

    role = "macro-sentiment"

    def analyze(self, input_data: dict, workspace: "Workspace") -> list[ClaimEvidenceCommit]:
        """Run an LLM-backed macro sentiment analysis or fallback."""

        ticker = input_data.get("ticker", "UNKNOWN")
        market = input_data.get("market_data", {})

        if self.llm_client:
            try:
                prompt = f"""Analyze the macro sentiment and rate environment for {ticker}\n                    Market Data: {json.dumps(market)}\n                    Provide 1-2 perspectives on macro impact."""
                response = self.llm_client.complete([
                    LLMMessage(role="system", content=prompt),
                ])
                claim = response.content
            except LLMError:
                claim = f"Macro environment data unavailable for {ticker}"
        else:
            claim = f"Macro environment data unavailable for {ticker}"

        return [self._make_commit(
            branch_name="macro-analysis",
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
