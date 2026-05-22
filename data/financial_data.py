"""Financial summary loading through yfinance."""

from __future__ import annotations

from typing import Any, Dict

from config import FINANCIAL_CACHE_TTL_DAYS
from data.cache import load_cache, load_stale_cache, save_cache


def fetch_financial_summary(ticker: str) -> Dict[str, str]:
    """Fetch and normalize fundamental metrics for agent input.

    yfinance's `info` dictionary is best-effort and may omit fields. Missing
    values are normalized to `"unknown"` so agents can create cautious commits
    rather than crashing.
    """

    info = _fetch_info(ticker)
    total_cash = info.get("totalCash")
    revenue_growth = _format_percent(info.get("revenueGrowth"))
    net_margin = _format_percent(info.get("profitMargins"))
    forward_pe = _format_number(info.get("forwardPE"))

    return {
        "revenue_growth_yoy": revenue_growth,
        "gross_margin": _format_percent(info.get("grossMargins")),
        "net_margin": net_margin,
        "forward_pe": forward_pe,
        "cash_position": _format_cash_position(total_cash),
        "debt_to_equity": _format_number(info.get("debtToEquity")),
        "free_cash_flow_margin": _free_cash_flow_margin(info),
        "return_on_equity": _format_percent(info.get("returnOnEquity")),
        "earnings_growth_yoy": _format_percent(info.get("earningsGrowth")),
        "free_cash_flow_yield": _free_cash_flow_yield(info),
        "dividend_yield": _format_percent(info.get("dividendYield")),
        "sector": str(info.get("sector") or "unknown"),
        "market_cap": _format_number(info.get("marketCap")),
        "beta": _format_number(info.get("beta")),
    }


def fetch_company_name(ticker: str) -> str:
    """Return a company name from yfinance when available."""

    info = _fetch_info(ticker)
    return str(
        info.get("longName")
        or info.get("shortName")
        or ticker.upper().strip()
    )


def _fetch_info(ticker: str) -> dict[str, Any]:
    """Fetch yfinance Ticker.info with a clear error on transport failure."""

    normalized_ticker = ticker.upper().strip()
    if not normalized_ticker:
        raise ValueError("Ticker must be a non-empty string.")

    cached_info = load_cache(
        "financials",
        normalized_ticker,
        FINANCIAL_CACHE_TTL_DAYS,
    )
    if isinstance(cached_info, dict):
        return cached_info

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ValueError(
            "yfinance is required for real financial data. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    try:
        info = yf.Ticker(normalized_ticker).info
    except Exception as exc:
        stale_info = load_stale_cache("financials", normalized_ticker)
        if isinstance(stale_info, dict):
            return stale_info
        raise ValueError(
            f"Failed to fetch financial summary for {normalized_ticker}: {exc}"
        ) from exc

    normalized_info = info or {}
    save_cache("financials", normalized_ticker, normalized_info)
    return normalized_info


def _format_percent(value: Any) -> str:
    """Format a yfinance ratio as a percentage string."""

    numeric_value = _to_float(value)
    if numeric_value is None:
        return "unknown"
    return f"{numeric_value * 100:.1f}%"


def _format_number(value: Any) -> str:
    """Format a numeric value as a compact string."""

    numeric_value = _to_float(value)
    if numeric_value is None:
        return "unknown"
    return f"{numeric_value:.1f}"


def _format_cash_position(value: Any) -> str:
    """Convert total cash to a coarse balance-sheet strength label."""

    total_cash = _to_float(value)
    if total_cash is None:
        return "Unknown"
    if total_cash >= 10_000_000_000:
        return "Strong"
    return "Limited"


def _free_cash_flow_margin(info: dict[str, Any]) -> str:
    """Return free cash flow as a percentage of revenue when available."""

    free_cash_flow = _to_float(info.get("freeCashflow"))
    revenue = _to_float(info.get("totalRevenue"))
    if free_cash_flow is None or revenue in {None, 0.0}:
        return "unknown"
    return f"{free_cash_flow / revenue * 100:.1f}%"


def _free_cash_flow_yield(info: dict[str, Any]) -> str:
    """Return free cash flow yield when free cash flow and market cap exist."""

    free_cash_flow = _to_float(info.get("freeCashflow"))
    market_cap = _to_float(info.get("marketCap"))
    if free_cash_flow is None or market_cap in {None, 0.0}:
        return "unknown"
    return f"{free_cash_flow / market_cap * 100:.1f}%"


def _to_float(value: Any) -> float | None:
    """Best-effort float conversion for yfinance values."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
