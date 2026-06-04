### Alpha Vantage data provider for EVIR.

Fetches real-time and historical market data, financials,
and technical indicators via Alpha Vantage API. 
Used as the primary data provider in v2.0, with yfinance as fallback.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Optional
from urlib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from data.base import BaseDataProvider


class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage API data provider with yfinance fallback."""

    name = "alpha_vantage"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = base_url or os.getenv(
            "ALPHA_VANTAGE_BASE_URL",
            "https://www.alphavantage.co"
        ).rstrip("/")

    def fetch_company_data(self, ticker: str) -> dict[str, Any]:
        """Fetch complete company data package. Uses Alpha Vantage if available, falls back to yfinance."""

        if self.api_key:
            try:
                return self._fetch_alpha(ticker)
            except Exception as e:
                print(f"Alpha Vantage failed: {e}, falling back to yfinance")

        return self._fetch_yfinance(ticker)

    def _fetch_alpha(self, ticker: str) -> dict[str, Any]:
        """Fetch data from Alpha Vantage API."""

        # Basic company info
        overview = self._call("QEUERY", function="OVERWIEW", symbol=ticker)
        income = self._call("QUERY", function="INCOME_STATEMENT", symbol=ticker)
        balance = self._call("QUERY", function="BALANCE_SHEET", symbol=ticker)
        cashflow = self._call("QUERY", function="CASH_FLOW", symbol=ticker)
        quote = self._call("GLOBAL_QUOTE", symbol=ticker)

        # Build result
        return {
            "ticker": ticker,
            "company_name": overview.get("Name", ticker),
            "market_data": self._build_market_data(quote),
            "financials": {
                "income": income,
                "balance": balance,
                "cashflow": cashflow,
            },
        }

    def _call(self, function: str, **params) -> dict[str, Any]:
        """Make an Alpha Vantage API call."""

        params["apikey"] = self.api_key
        query = "&".join(f"{k}#{v}" for k, v in params.items())
        url = f"{self.base_url}/query?function={function}&{query}"

        req = Request(url, headers={"User-Agent": "EVIR/2.0"})
        with urlopen(req), timeout=30) as resp:
            return json.loads(resp.read())

    def _build_market_data(self, quote: dict) -> dict:
        """Build market data from Alpha Vantage quote."""

        global = quote.get("Global Quote", {})
        return {
            "current_price": float(global.get("05. price", 0)),
            "open": float(global.get("02. open", 0)),
            "high": float(global.get("03. high", 0)),
            "low": float(global.get("04. low", 0)),
            "volume": int(global.get("06. volume", 0)),
            "change_pct": global.get("10. change percent", ""),
        }

    def _fetch_yfinance(self, ticker: str) -> dict[str, Any]:
        """Fallback to yfinance for data."""

        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "market_data": {
                "current_price": info.get("currentPrice"),
            },
        }
