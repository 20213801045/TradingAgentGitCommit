"""Real new-evidence adapter for decision revision."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict

from data.financial_data import fetch_financial_summary
from data.market_data import fetch_price_history
from data.technical_indicators import build_technical_summary


def get_real_new_evidence(ticker: str) -> Dict[str, Any]:
    """Fetch current real data and convert it to RevisionEngine input shape."""

    normalized_ticker = ticker.upper().strip()
    if not normalized_ticker:
        raise ValueError("Ticker must be a non-empty string.")

    price_history = fetch_price_history(normalized_ticker, period="6mo")
    financial_metrics = fetch_financial_summary(normalized_ticker)
    technical_indicators = build_technical_summary(normalized_ticker, price_history)

    return {
        "ticker": normalized_ticker,
        "new_financial_metrics": {
            "revenue_growth_yoy": financial_metrics.get("revenue_growth_yoy", "unknown"),
            "net_margin": financial_metrics.get("net_margin", "unknown"),
            "forward_pe": financial_metrics.get("forward_pe", "unknown"),
        },
        "new_technical_indicators": {
            "rsi": technical_indicators.get("rsi", "unknown"),
            "price_trend": technical_indicators.get("price_trend", "unknown"),
            "volatility": technical_indicators.get("volatility", "unknown"),
        },
        "new_events": [
            {
                "title": (
                    f"Latest yfinance data snapshot for {normalized_ticker} "
                    "used as revision evidence"
                ),
                "source": "Yahoo Finance via yfinance",
                "source_type": "financial_data_provider",
                "timestamp": date.today().isoformat(),
            }
        ],
        "data_source": "yfinance",
        "as_of_date": date.today().isoformat(),
    }
