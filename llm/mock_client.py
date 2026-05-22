"""Deterministic mock LLM client."""

import json

from llm.base import BaseLLMClient, LLMMessage, LLMResponse


class MockLLMClient(BaseLLMClient):
    """A local deterministic LLM replacement used by tests and demos."""

    provider = "mock"

    def __init__(self, model: str = "mock-llm") -> None:
        self.model = model

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        response_format: dict[str, object] | None = None,
    ) -> LLMResponse:
        """Return a deterministic counter-evidence question."""

        del temperature
        last_user_message = _last_user_message(messages)
        if "risk review source commits" in last_user_message.lower():
            content = json.dumps(
                {
                    "insights": [
                        {
                            "dimension": "valuation_risk_review",
                            "claim": "Valuation risk should cap upside conviction until the premium is better supported.",
                            "confidence": "medium",
                            "risk_tag": "valuation_risk",
                            "time_horizon": "6-12 months",
                        },
                        {
                            "dimension": "technical_timing_risk",
                            "claim": "Volatility creates timing risk for near-term entries.",
                            "confidence": "medium",
                            "risk_tag": "volatility_risk",
                            "time_horizon": "1-3 months",
                        },
                        {
                            "dimension": "bullish_thesis_challenge",
                            "claim": "The bullish thesis needs explicit counter-evidence before conviction improves.",
                            "confidence": "medium",
                            "risk_tag": "evidence_gap",
                            "time_horizon": "ongoing",
                        },
                        {
                            "dimension": "evidence_quality_risk",
                            "claim": "Evidence quality and freshness should be monitored before relying on time-sensitive claims.",
                            "confidence": "low",
                            "risk_tag": "temporal_uncertainty",
                            "time_horizon": "ongoing",
                        },
                        {
                            "dimension": "portfolio_positioning_risk",
                            "claim": "Portfolio positioning risk should keep sizing disciplined.",
                            "confidence": "medium",
                            "risk_tag": "portfolio_risk",
                            "time_horizon": "ongoing",
                        },
                    ]
                }
            )
        elif "technical indicators" in last_user_message.lower():
            content = json.dumps(
                {
                    "insights": [
                        {
                            "dimension": "trend_structure",
                            "claim": "The price trend remains constructive above key moving averages.",
                            "confidence": "medium",
                            "risk_tag": "price_trend_support",
                            "time_horizon": "1-3 months",
                        },
                        {
                            "dimension": "momentum",
                            "claim": "RSI supports positive short-term momentum without requiring new facts.",
                            "confidence": "medium",
                            "risk_tag": "momentum",
                            "time_horizon": "1-3 months",
                        },
                        {
                            "dimension": "volatility_risk",
                            "claim": "Volatility remains a timing risk that should shape entry sizing.",
                            "confidence": "medium",
                            "risk_tag": "volatility_risk",
                            "time_horizon": "1-3 months",
                        },
                        {
                            "dimension": "support_resistance",
                            "claim": "Defined support and resistance levels frame near-term risk-reward.",
                            "confidence": "medium",
                            "risk_tag": "support_resistance",
                            "time_horizon": "1-3 months",
                        },
                    ]
                }
            )
        elif "valuation metrics" in last_user_message.lower():
            content = json.dumps(
                {
                    "insights": [
                        {
                            "dimension": "relative_valuation",
                            "claim": "The stock trades at a valuation premium versus the sector benchmark.",
                            "confidence": "medium",
                            "risk_tag": "valuation_risk",
                            "time_horizon": "6-12 months",
                        },
                        {
                            "dimension": "growth_adjusted_valuation",
                            "claim": "Growth and free-cash-flow yield provide only partial valuation support.",
                            "confidence": "medium",
                            "risk_tag": "valuation_mixed",
                            "time_horizon": "12 months",
                        },
                    ]
                }
            )
        elif "fundamental metrics" in last_user_message.lower():
            content = json.dumps(
                {
                    "insights": [
                        {
                            "dimension": "growth_quality",
                            "claim": "Revenue growth provides a positive fundamental growth-quality signal.",
                            "confidence": "medium",
                            "risk_tag": "growth_quality",
                            "time_horizon": "12 months",
                        },
                        {
                            "dimension": "margin_quality",
                            "claim": "Profitability appears stable based on reported margin metrics.",
                            "confidence": "medium",
                            "risk_tag": "profitability",
                            "time_horizon": "12-24 months",
                        },
                        {
                            "dimension": "cash_generation",
                            "claim": "Free-cash-flow margin supports cash generation quality.",
                            "confidence": "medium",
                            "risk_tag": "cash_generation_support",
                            "time_horizon": "12-24 months",
                        },
                        {
                            "dimension": "balance_sheet",
                            "claim": "Balance sheet resilience has support from cash and leverage evidence.",
                            "confidence": "medium",
                            "risk_tag": "balance_sheet_strength",
                            "time_horizon": "12-24 months",
                        },
                        {
                            "dimension": "capital_allocation",
                            "claim": "Cash and free cash flow provide support for capital allocation capacity.",
                            "confidence": "medium",
                            "risk_tag": "capital_allocation_support",
                            "time_horizon": "12-24 months",
                        },
                        {
                            "dimension": "fundamental_valuation_risk",
                            "claim": "Valuation risk should be monitored alongside the growth profile.",
                            "confidence": "medium",
                            "risk_tag": "valuation_risk",
                            "time_horizon": "6-12 months",
                        },
                    ]
                }
            )
        elif (
            "research_plan" in last_user_message
            or "research coordinator" in last_user_message.lower()
        ):
            content = json.dumps(
                {
                    "research_plan": [
                        "Coordinate source-analysis branches before synthesis.",
                        "Verify that every material claim remains evidence-linked.",
                    ],
                    "quality_checks": [
                        "Check branch coverage before merge.",
                        "Check that stale evidence is visible in the final report.",
                    ],
                    "follow_up_priorities": [
                        "Upgrade FundamentalAgent to LLM-first analysis.",
                        "Ask specialist agents for reruns when branch coverage is incomplete.",
                    ],
                    "confidence": "medium",
                }
            )
        elif "insights" in last_user_message.lower() or "洞察" in last_user_message:
            content = json.dumps(
                {
                    "insights": [
                        {
                            "claim": "LLM 综合判断认为增长质量仍有支持，但估值溢价限制上行空间。",
                            "evidence_commit_id": "",
                            "confidence": "medium",
                            "risk_tag": "llm_mixed",
                            "time_horizon": "6-12 months",
                            "counter_evidence": [
                                "如果后续收入增长继续放缓，该判断需要下调。",
                                "如果估值倍数继续扩张，该判断需要降低置信度。",
                            ],
                        }
                    ]
                }
            )
        elif "technical" in last_user_message.lower():
            content = "更新后的技术指标是否会推翻这个信号？"
        elif "growth" in last_user_message.lower():
            content = "哪些近期证据会显示增长正在放缓？"
        elif "profitability" in last_user_message.lower() or "margin" in last_user_message.lower():
            content = "哪些证据会显示利润率正在承压？"
        else:
            content = "什么证据最能直接挑战这个观点？"

        if response_format == {"type": "json_object"} and not content.startswith("{"):
            content = json.dumps({"question": content})

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            raw=None,
        )


def _last_user_message(messages: list[LLMMessage]) -> str:
    """Return the last user message content from a chat prompt."""

    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""
