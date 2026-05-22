"""Market price history loading through yfinance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from config import PRICE_CACHE_TTL_DAYS
from data.cache import load_cache, load_stale_cache, save_cache


def fetch_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch OHLCV price history for a ticker.

    Raises:
        ValueError: If yfinance is unavailable or no usable price data is found.
    """

    normalized_ticker = ticker.upper().strip()
    if not normalized_ticker:
        raise ValueError("Ticker must be a non-empty string.")

    cache_key = f"{normalized_ticker}_{period}"
    cached_history = _price_history_from_cache(
        load_cache("prices", cache_key, PRICE_CACHE_TTL_DAYS),
        normalized_ticker,
    )
    if cached_history is not None:
        return cached_history

    try:
        history = _fetch_price_history_uncached(normalized_ticker, period)
        price_history = _normalize_price_history(history, normalized_ticker)
        save_cache("prices", cache_key, _price_history_to_cache(price_history))
        return price_history
    except Exception as exc:
        stale_history = _price_history_from_cache(
            load_stale_cache("prices", cache_key),
            normalized_ticker,
        )
        if stale_history is not None:
            return stale_history
        raise ValueError(
            f"Failed to fetch price history for {normalized_ticker}: {exc}"
        ) from exc


def _fetch_price_history_uncached(
    normalized_ticker: str,
    period: str,
) -> pd.DataFrame:
    """Fetch OHLCV history without consulting local cache."""

    try:
        import yfinance as yf

        return yf.Ticker(normalized_ticker).history(
            period=period,
            auto_adjust=False,
        )
    except Exception:
        return _fetch_price_history_from_yahoo_chart(normalized_ticker, period)


def _normalize_price_history(
    history: pd.DataFrame,
    normalized_ticker: str,
) -> pd.DataFrame:
    """Validate and normalize OHLCV price history."""

    if history is None or history.empty:
        raise ValueError(f"No price history returned for {normalized_ticker}.")

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [
        column for column in required_columns if column not in history.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Price history for {normalized_ticker} is missing columns: "
            f"{', '.join(missing_columns)}"
        )

    price_history = history.loc[:, required_columns].dropna(subset=["Close"])
    if price_history.empty:
        raise ValueError(f"Price history for {normalized_ticker} has no Close data.")

    price_history.index.name = "Date"
    return price_history


def _fetch_price_history_from_yahoo_chart(
    ticker: str,
    period: str,
) -> pd.DataFrame:
    """Fetch price history from Yahoo's chart endpoint without crumb handling."""

    range_value = _period_to_chart_range(period)
    params = urlencode({"range": range_value, "interval": "1d", "events": "history"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
    request = Request(
        url=url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "KHTML, like Gecko Chrome/120.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"Yahoo chart fallback failed for {ticker}: {exc}") from exc

    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Yahoo chart fallback returned malformed data for {ticker}.") from exc

    index = [
        datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        for timestamp in timestamps
    ]
    history = pd.DataFrame(
        {
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        },
        index=pd.to_datetime(index),
    )
    return history.dropna(subset=["Close"])


def _period_to_chart_range(period: str) -> str:
    """Map yfinance-style period strings to Yahoo chart range values."""

    normalized_period = period.lower().strip()
    supported_ranges = {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }
    if normalized_period in supported_ranges:
        return normalized_period
    return "6mo"


def _price_history_to_cache(history: pd.DataFrame) -> dict[str, list[object]]:
    """Convert a price history frame into JSON-safe cache data."""

    return {
        "Date": [str(index.date()) for index in history.index],
        "Open": [float(value) for value in history["Open"]],
        "High": [float(value) for value in history["High"]],
        "Low": [float(value) for value in history["Low"]],
        "Close": [float(value) for value in history["Close"]],
        "Volume": [int(value) for value in history["Volume"]],
    }


def _price_history_from_cache(
    cached_data: object,
    normalized_ticker: str,
) -> pd.DataFrame | None:
    """Convert cached JSON data into a normalized price history frame."""

    if not isinstance(cached_data, dict):
        return None
    required_keys = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_keys.issubset(cached_data):
        return None
    try:
        history = pd.DataFrame(
            {
                "Open": cached_data["Open"],
                "High": cached_data["High"],
                "Low": cached_data["Low"],
                "Close": cached_data["Close"],
                "Volume": cached_data["Volume"],
            },
            index=pd.to_datetime(cached_data["Date"]),
        )
    except (TypeError, ValueError):
        return None
    return _normalize_price_history(history, normalized_ticker)
