"""Mock local company data used by the prototype agents."""

from typing import Any


def get_mock_company_data(ticker: str = "AAPL") -> dict[str, Any]:
    """Return a small mock dataset for one company.

    The values are illustrative only. They are designed to exercise the data
    schemas and branch workflow without relying on external APIs.
    """

    if ticker.upper() != "AAPL":
        raise ValueError("Mock dataset currently supports only AAPL.")

    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "financial_metrics": {
            "revenue_growth_yoy": "8%",
            "gross_margin": "45%",
            "net_margin": "24%",
            "forward_pe": "28",
            "cash_position": "Strong",
            "debt_to_equity": "1.5",
            "free_cash_flow_margin": "22%",
            "return_on_equity": "145%",
        },
        "valuation_metrics": {
            "forward_pe": "28",
            "sector_forward_pe": "24",
            "earnings_growth_yoy": "6%",
            "free_cash_flow_yield": "3.2%",
            "dividend_yield": "0.5%",
        },
        "financial_statements": {
            "revenue_growth_3y": "7%",
            "gross_margin_trend": "stable",
            "free_cash_flow_margin": "22%",
            "debt_to_equity": "1.5",
            "return_on_equity": "145%",
        },
        "industry_comparison": {
            "sector": "Consumer Electronics",
            "revenue_growth_yoy": "8%",
            "sector_revenue_growth_yoy": "5%",
            "net_margin": "24%",
            "sector_net_margin": "16%",
            "forward_pe": "28",
            "sector_forward_pe": "24",
        },
        "macro_context": {
            "rate_environment": "elevated",
            "consumer_demand": "mixed",
            "usd_trend": "stable",
            "inflation_pressure": "moderate",
        },
        "technical_indicators": {
            "price_trend": "above 20-day and 50-day moving averages",
            "rsi": "62",
            "volatility": "moderate",
            "support_level": "185",
            "resistance_level": "205",
        },
        "backtest_summary": {
            "strategy": "trend_following_ma20_ma50",
            "lookback_period": "6 months",
            "win_rate": "54%",
            "max_drawdown": "12%",
            "annualized_return": "9%",
        },
        "portfolio_context": {
            "position_size": "3%",
            "max_position_size": "5%",
            "correlation_to_market": "0.75",
            "liquidity": "high",
            "portfolio_role": "core quality growth watchlist candidate",
        },
        "news": [
            {
                "title": "Apple reports stable revenue growth in latest quarter",
                "source": "Mock Financial News",
                "timestamp": "2026-05-01",
            },
        ],
    }
