"""Deep Research Agent — LLM-first, replaces 6 template agents.

Replaces: fundamental_agent + financial_statement_agent + valuation_agent +
           industry_agent + portfolio_agent + llm_analysis_agent

One LLM call → 8–12 high-quality, evidence-grounded commits covering
all fundamental and strategic dimensions of a stock.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import BaseAgent
from llm.base import BaseLLMClient, LLMError, LLMMessage
from llm.json_utils import parse_json_response
from models.schemas import ClaimEvidenceCommit, Workspace


# ── structured output schema ──────────────────────────────────────────────
class ResearchInsight(BaseModel):
    dimension: str = Field(description="growth|profitability|cash_flow|balance_sheet|valuation|industry|portfolio|risk|opportunity")
    claim: str = Field(min_length=15, description="Evidence-grounded claim, 1-3 sentences")
    confidence: str = Field(default="medium", description="low|medium|high")
    risk_tag: str = Field(default="", description="snake_case risk tag")
    time_horizon: str = Field(default="12 months")
    metric_reference: str = Field(default="", description="Key metric this claim is based on")


class DeepResearchOutput(BaseModel):
    ticker: str
    company_name: str
    insights: list[ResearchInsight] = Field(min_length=6, max_length=14)


# ── the system prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a world-class equity research analyst at a top hedge fund.
Your job: analyze the financial data below and produce 8–12 concise, evidence-grounded insights.

RULES:
1. Every claim MUST reference a specific metric from the provided data.
2. Be quantitative — say "FCF margin of 52% indicates..." not "cash flow seems OK".
3. Flag RED FLAGS honestly — if debt is high or margins are thin, say so.
4. Do NOT invent data. If a metric is missing, note the gap instead of guessing.
5. Write in the language of the input data (Chinese or English).
6. Each insight must be self-contained — someone reading it alone should understand the point.

COVER THESE DIMENSIONS (at least one insight each):
- growth: Revenue/earnings growth trajectory and quality
- profitability: Margin structure, ROE, unit economics
- cash_flow: FCF conversion, cash generation quality, sustainability
- balance_sheet: Leverage, liquidity, financial flexibility
- valuation: P/E, PEG, EV/EBITDA relative to growth and peers
- industry: Competitive position vs sector benchmarks
- risk: Key fundamental risks or data gaps
- opportunity: Bullish catalysts or underappreciated strengths

OUTPUT: Valid JSON matching this schema:
{"ticker":"...","company_name":"...","insights":[{"dimension":"...","claim":"...","confidence":"low|medium|high","risk_tag":"...","time_horizon":"...","metric_reference":"..."}]}"""


# ── the agent ─────────────────────────────────────────────────────────────
class DeepResearchAgent(BaseAgent):
    """LLM-first deep fundamental research agent — replaces 6 template agents."""

    name = "DeepResearchAgent"
    role = "deep-research-agent"
    branch_name = "deep-research"

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Run deep LLM research, fall back to lightweight extract if needed."""

        if self.llm_client is not None:
            try:
                return self._llm_analyze(input_data, workspace)
            except (LLMError, json.JSONDecodeError, ValidationError, ValueError):
                pass

        return self._fallback_analyze(input_data, workspace)

    def _llm_analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Call the LLM with a comprehensive research prompt."""

        prompt = self._build_prompt(input_data, workspace)
        response = self.llm_client.complete(  # type: ignore[union-attr]
            [
                LLMMessage(role="system", content=SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_response(response.content)
        validated = DeepResearchOutput.model_validate(parsed)

        timestamp = str(input_data.get("as_of_date", "2026-05-22"))
        source = "Deep Research (LLM) based on provider data"
        commits: list[ClaimEvidenceCommit] = []
        for insight in validated.insights:
            evidence = self._make_evidence(
                content=insight.claim,
                source=source,
                source_type="llm_deep_research",
                timestamp=timestamp,
                metric_name=insight.metric_reference or insight.dimension,
                metric_value=insight.dimension,
            )
            commits.append(
                self.create_commit(
                    claim=insight.claim.strip(),
                    evidence=evidence,
                    confidence=insight.confidence,
                    risk_tag=insight.risk_tag or self._dimension_risk_tag(insight.dimension),
                    time_horizon=insight.time_horizon or "12 months",
                )
            )
        return commits

    def _build_prompt(self, input_data: dict[str, Any], workspace: Workspace) -> str:
        """Assemble all available financial data into a research brief."""

        financial = input_data.get("financial_metrics", {})
        statements = input_data.get("financial_statements", {})
        industry = input_data.get("industry_comparison", {})
        macro = input_data.get("macro_context", {})
        market = input_data.get("market_data", {})

        lines = [
            f"# Research Brief: {workspace.ticker}",
            f"Company: {workspace.company_name or 'unknown'}",
            f"Research Question: {workspace.research_question}",
            f"Data Source: {input_data.get('data_source', 'provider')}",
            f"As-of Date: {input_data.get('as_of_date', 'latest')}",
            "",
            "## Financial Metrics",
        ]
        for k, v in sorted(financial.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Financial Statements Deep-Dive")
        for k, v in sorted(statements.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Industry & Peer Comparison")
        for k, v in sorted(industry.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Macro Context")
        for k, v in sorted(macro.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        if market:
            price_info = {k: market[k] for k in ["current_price", "market_cap", "beta", "pe_ratio"] if k in market}
            if price_info:
                lines.append("## Market Data")
                for k, v in sorted(price_info.items()):
                    lines.append(f"- {k}: {v}")

        lines.append("")
        lines.append("## Instructions")
        lines.append("Produce 8–12 insights covering growth, profitability, cash_flow, balance_sheet, valuation, industry, risk, and opportunity.")
        lines.append("Every claim MUST reference specific metrics from the data above.")
        lines.append("Return ONLY valid JSON.")

        return "\n".join(lines)

    @staticmethod
    def _dimension_risk_tag(dimension: str) -> str:
        mapping = {
            "growth": "growth_analysis",
            "profitability": "profitability_analysis",
            "cash_flow": "cash_flow_analysis",
            "balance_sheet": "balance_sheet_analysis",
            "valuation": "valuation_analysis",
            "industry": "industry_analysis",
            "risk": "fundamental_risk",
            "opportunity": "opportunity",
        }
        return mapping.get(dimension, "deep_research")

    def _fallback_analyze(
        self,
        input_data: dict[str, Any],
        workspace: Workspace,
    ) -> list[ClaimEvidenceCommit]:
        """Minimal fallback: extract what we can from raw metrics."""

        financial = input_data.get("financial_metrics", {})
        timestamp = str(input_data.get("as_of_date", "2026-05-22"))
        commits: list[ClaimEvidenceCommit] = []

        # key metrics to extract
        key_metrics = [
            ("revenue_growth_yoy", "growth", "Revenue growth YoY"),
            ("gross_margin", "profitability", "Gross margin"),
            ("net_margin", "profitability", "Net margin"),
            ("free_cash_flow_margin", "cash_flow", "FCF margin"),
            ("debt_to_equity", "balance_sheet", "Debt-to-equity"),
            ("forward_pe", "valuation", "Forward P/E"),
            ("return_on_equity", "profitability", "ROE"),
            ("earnings_growth_yoy", "growth", "Earnings growth YoY"),
        ]

        for metric_key, dimension, label in key_metrics:
            value = financial.get(metric_key, "unknown")
            if str(value).lower() == "unknown":
                continue
            evidence = self._make_evidence(
                content=f"{label}: {value}",
                source="Provider data (fallback mode)",
                source_type="financial_metric",
                timestamp=timestamp,
                metric_name=metric_key,
                metric_value=str(value),
            )
            commits.append(
                self.create_commit(
                    claim=f"{label} is {value}.",
                    evidence=evidence,
                    confidence="low",
                    risk_tag=self._dimension_risk_tag(dimension),
                    time_horizon="12 months",
                )
            )

        if not commits:
            commits.append(
                self.create_commit(
                    claim="Insufficient financial data available for analysis. Enable LLM for deeper research.",
                    evidence=self._make_evidence(
                        content="No usable financial metrics found in provider data.",
                        source="Provider data (fallback mode)",
                        source_type="system",
                        timestamp=timestamp,
                        metric_name="data_availability",
                        metric_value="none",
                    ),
                    confidence="low",
                    risk_tag="data_gap",
                    time_horizon="N/A",
                )
            )
        return commits
