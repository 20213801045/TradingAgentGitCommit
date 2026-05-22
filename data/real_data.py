"""Real company data adapter built on yfinance."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict

from data.backtest import build_backtest_summary
from data.financial_data import fetch_company_name, fetch_financial_summary
from data.market_data import fetch_price_history
from data.technical_indicators import build_technical_summary


def get_real_company_data(ticker: str) -> Dict[str, Any]:
    """Fetch real market and financial data in EVIR's agent input shape."""

    normalized_ticker = ticker.upper().strip()
    price_history = fetch_price_history(normalized_ticker)
    technical_indicators = build_technical_summary(normalized_ticker, price_history)
    backtest_summary = build_backtest_summary(price_history)
    financial_metrics = fetch_financial_summary(normalized_ticker)
    company_name = fetch_company_name(normalized_ticker)
    sector = financial_metrics.get("sector", "unknown")

    return {
        "ticker": normalized_ticker,
        "company_name": company_name,
        "financial_metrics": financial_metrics,
        "valuation_metrics": {
            "forward_pe": financial_metrics.get("forward_pe", "unknown"),
            "sector_forward_pe": _sector_benchmark(sector, "forward_pe"),
            "earnings_growth_yoy": financial_metrics.get("earnings_growth_yoy", "unknown"),
            "free_cash_flow_yield": financial_metrics.get("free_cash_flow_yield", "unknown"),
            "dividend_yield": financial_metrics.get("dividend_yield", "unknown"),
        },
        "financial_statements": {
            "revenue_growth_3y": financial_metrics.get("revenue_growth_yoy", "unknown"),
            "gross_margin_trend": "unknown",
            "free_cash_flow_margin": financial_metrics.get("free_cash_flow_margin", "unknown"),
            "debt_to_equity": financial_metrics.get("debt_to_equity", "unknown"),
            "return_on_equity": financial_metrics.get("return_on_equity", "unknown"),
        },
        "industry_comparison": {
            "sector": sector,
            "revenue_growth_yoy": financial_metrics.get("revenue_growth_yoy", "unknown"),
            "sector_revenue_growth_yoy": _sector_benchmark(sector, "revenue_growth_yoy"),
            "net_margin": financial_metrics.get("net_margin", "unknown"),
            "sector_net_margin": _sector_benchmark(sector, "net_margin"),
            "forward_pe": financial_metrics.get("forward_pe", "unknown"),
            "sector_forward_pe": _sector_benchmark(sector, "forward_pe"),
        },
        "macro_context": {
            "rate_environment": "unknown",
            "consumer_demand": "unknown",
            "usd_trend": "unknown",
            "inflation_pressure": "unknown",
        },
        "technical_indicators": technical_indicators,
        "backtest_summary": backtest_summary,
        "portfolio_context": {
            "position_size": "watchlist",
            "max_position_size": "unknown",
            "correlation_to_market": financial_metrics.get("beta", "unknown"),
            "liquidity": "unknown",
            "portfolio_role": "single-name watchlist candidate",
        },
        "news": [],
        "data_source": "yfinance",
        "as_of_date": date.today().isoformat(),
    }


def _sector_benchmark(sector: str, metric: str) -> str:
    """Return coarse sector benchmarks until a real benchmark feed is wired."""

    normalized_sector = sector.lower()
    if "technology" in normalized_sector or "consumer" in normalized_sector:
        benchmarks = {
            "revenue_growth_yoy": "5%",
            "net_margin": "16%",
            "forward_pe": "24",
        }
    else:
        benchmarks = {
            "revenue_growth_yoy": "4%",
            "net_margin": "10%",
            "forward_pe": "18",
        }
    return benchmarks.get(metric, "unknown")
