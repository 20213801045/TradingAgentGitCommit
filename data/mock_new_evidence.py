"""Mock new evidence used to demonstrate decision revision."""

from typing import Any


def get_mock_new_evidence(ticker: str = "AAPL") -> dict[str, Any]:
    """Return deterministic ticker-specific new evidence for revision demos."""

    normalized_ticker = ticker.upper().strip()
    if normalized_ticker == "AAPL":
        return _aapl_new_evidence()
    if normalized_ticker == "MU":
        return _mu_new_evidence()
    return _generic_new_evidence(normalized_ticker)


def _aapl_new_evidence() -> dict[str, Any]:
    """Return deterministic new evidence for AAPL revision demos."""

    return {
        "ticker": "AAPL",
        "new_financial_metrics": {
            "revenue_growth_yoy": "2%",
            "net_margin": "20%",
            "forward_pe": "30",
        },
        "new_technical_indicators": {
            "rsi": "42",
            "price_trend": "below 20-day moving average",
            "volatility": "high",
        },
        "new_events": [
            {
                "title": "Company guidance weakened due to slowing demand",
                "source": "Mock Financial News",
                "source_type": "reputable_news",
                "timestamp": "2026-05-15",
            }
        ],
    }


def _mu_new_evidence() -> dict[str, Any]:
    """Return deterministic new evidence for MU revision demos."""

    return {
        "ticker": "MU",
        "new_financial_metrics": {
            "revenue_growth_yoy": "42%",
            "net_margin": "32%",
            "forward_pe": "9.5",
        },
        "new_technical_indicators": {
            "rsi": "58",
            "price_trend": "mixed trend near recent resistance",
            "volatility": "high",
        },
        "new_events": [
            {
                "title": (
                    "Memory demand remains cyclical despite AI-driven growth "
                    "tailwinds"
                ),
                "source": "Mock Semiconductor News",
                "source_type": "reputable_news",
                "timestamp": "2026-05-15",
            }
        ],
    }


def _generic_new_evidence(ticker: str) -> dict[str, Any]:
    """Return neutral ticker-specific new evidence for unsupported tickers."""

    return {
        "ticker": ticker,
        "new_financial_metrics": {
            "revenue_growth_yoy": "unknown",
            "net_margin": "unknown",
            "forward_pe": "unknown",
        },
        "new_technical_indicators": {
            "rsi": "unknown",
            "price_trend": "unknown",
            "volatility": "unknown",
        },
        "new_events": [
            {
                "title": f"No ticker-specific mock revision event is available for {ticker}",
                "source": "Mock Revision Data",
                "source_type": "reputable_news",
                "timestamp": "2026-05-15",
            }
        ],
    }
