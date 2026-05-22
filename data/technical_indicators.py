"""Technical indicator calculations for real market data."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import INVESTMENT_THRESHOLDS

def calculate_rsi(close_prices: pd.Series, window: int = 14) -> float:
    """Calculate the latest relative strength index value."""

    prices = _clean_series(close_prices)
    if len(prices) < 2:
        return float("nan")

    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()

    if average_gain.dropna().empty or average_loss.dropna().empty:
        average_gain = gains.expanding(min_periods=2).mean()
        average_loss = losses.expanding(min_periods=2).mean()

    latest_gain = float(average_gain.dropna().iloc[-1])
    latest_loss = float(average_loss.dropna().iloc[-1])
    if latest_loss == 0:
        return 100.0

    relative_strength = latest_gain / latest_loss
    return float(100 - (100 / (1 + relative_strength)))


def calculate_volatility(close_prices: pd.Series, window: int = 20) -> float:
    """Calculate latest annualized realized volatility from close prices."""

    prices = _clean_series(close_prices)
    returns = prices.pct_change().dropna()
    if returns.empty:
        return float("nan")

    rolling_volatility = returns.rolling(window=window, min_periods=2).std()
    latest_daily_volatility = float(rolling_volatility.dropna().iloc[-1])
    return float(latest_daily_volatility * np.sqrt(252))


def calculate_moving_average(close_prices: pd.Series, window: int) -> float:
    """Calculate the latest moving average over a window."""

    prices = _clean_series(close_prices)
    if prices.empty:
        return float("nan")
    return float(prices.tail(window).mean())


def calculate_support_level(
    price_history: pd.DataFrame,
    window: int = 60,
) -> float:
    """Calculate a simple support level from recent lows."""

    if "Low" not in price_history.columns or price_history.empty:
        return float("nan")
    return float(price_history["Low"].dropna().tail(window).min())


def calculate_resistance_level(
    price_history: pd.DataFrame,
    window: int = 60,
) -> float:
    """Calculate a simple resistance level from recent highs."""

    if "High" not in price_history.columns or price_history.empty:
        return float("nan")
    return float(price_history["High"].dropna().tail(window).max())


def build_technical_summary(
    ticker: str,
    price_history: pd.DataFrame,
) -> Dict[str, str]:
    """Build agent-ready technical indicator strings from price history."""

    del ticker
    if "Close" not in price_history.columns or price_history.empty:
        raise ValueError("Price history must include a Close column.")

    close_prices = _clean_series(price_history["Close"])
    if close_prices.empty:
        raise ValueError("Price history has no usable Close prices.")

    latest_close = float(close_prices.iloc[-1])
    ma20 = calculate_moving_average(close_prices, 20)
    ma50 = calculate_moving_average(close_prices, 50)
    rsi = calculate_rsi(close_prices)
    volatility = calculate_volatility(close_prices)
    support_level = calculate_support_level(price_history)
    resistance_level = calculate_resistance_level(price_history)

    if latest_close > ma20 and latest_close > ma50:
        price_trend = "above 20-day and 50-day moving averages"
    elif latest_close < ma20 and latest_close < ma50:
        price_trend = "below 20-day and 50-day moving averages"
    elif latest_close < ma20:
        price_trend = "below 20-day moving average"
    elif latest_close < ma50:
        price_trend = "below 50-day moving average"
    else:
        price_trend = "mixed trend"

    return {
        "price_trend": price_trend,
        "rsi": _format_float(rsi),
        "volatility": _classify_volatility(volatility),
        "support_level": _format_float(support_level),
        "resistance_level": _format_float(resistance_level),
    }


def _clean_series(values: pd.Series) -> pd.Series:
    """Return a numeric series with missing values removed."""

    return pd.to_numeric(values, errors="coerce").dropna()


def _classify_volatility(annualized_volatility: float) -> str:
    """Map annualized volatility to a compact risk label."""

    if np.isnan(annualized_volatility):
        return "unknown"
    if annualized_volatility < INVESTMENT_THRESHOLDS["low_volatility"]:
        return "low"
    if annualized_volatility <= INVESTMENT_THRESHOLDS["high_volatility"]:
        return "moderate"
    return "high"


def _format_float(value: float) -> str:
    """Format numeric indicator values for evidence content."""

    if np.isnan(value):
        return "unknown"
    return f"{value:.2f}"
