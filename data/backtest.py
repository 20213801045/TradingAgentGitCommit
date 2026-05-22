"""Simple backtest helpers for real market data."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def build_backtest_summary(
    price_history: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
) -> Dict[str, str]:
    """Build a trend-following backtest summary from OHLCV price history."""

    if "Close" not in price_history.columns or price_history.empty:
        return _unknown_summary(short_window, long_window)

    close_prices = pd.to_numeric(price_history["Close"], errors="coerce").dropna()
    if len(close_prices) < 2:
        return _unknown_summary(short_window, long_window)

    strategy_returns = _moving_average_strategy_returns(
        close_prices,
        short_window,
        long_window,
    )
    buy_hold_returns = close_prices.pct_change().dropna()

    if strategy_returns.empty or buy_hold_returns.empty:
        return _unknown_summary(short_window, long_window)

    strategy_total_return = _total_return(strategy_returns)
    buy_hold_total_return = _total_return(buy_hold_returns)

    return {
        "strategy": f"trend_following_ma{short_window}_ma{long_window}",
        "lookback_period": f"{len(close_prices)} trading days",
        "win_rate": _format_percent(_win_rate(strategy_returns)),
        "max_drawdown": _format_percent(abs(_max_drawdown(strategy_returns))),
        "annualized_return": _format_percent(_annualized_return(strategy_returns)),
        "total_return": _format_percent(strategy_total_return),
        "buy_hold_return": _format_percent(buy_hold_total_return),
        "excess_return": _format_percent(strategy_total_return - buy_hold_total_return),
    }


def _moving_average_strategy_returns(
    close_prices: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """Return daily returns for a long/cash moving-average strategy."""

    short_ma = close_prices.rolling(
        window=short_window,
        min_periods=min(short_window, len(close_prices)),
    ).mean()
    long_ma = close_prices.rolling(
        window=long_window,
        min_periods=min(long_window, len(close_prices)),
    ).mean()
    signal = (short_ma > long_ma).astype(float)
    daily_returns = close_prices.pct_change().fillna(0.0)
    strategy_returns = signal.shift(1).fillna(0.0) * daily_returns
    return strategy_returns.dropna()


def _total_return(returns: pd.Series) -> float:
    """Return cumulative return for a return series."""

    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def _annualized_return(returns: pd.Series) -> float:
    """Return annualized return for a return series."""

    if returns.empty:
        return 0.0
    total_return = _total_return(returns)
    periods = max(len(returns), 1)
    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    """Return maximum drawdown as a negative decimal."""

    if returns.empty:
        return 0.0
    equity_curve = (1.0 + returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdowns = equity_curve / running_peak - 1.0
    return float(drawdowns.min())


def _win_rate(returns: pd.Series) -> float:
    """Return share of active days with positive returns."""

    active_returns = returns[returns != 0]
    if active_returns.empty:
        return 0.0
    return float((active_returns > 0).mean())


def _format_percent(value: float) -> str:
    """Format a decimal return as a percent string."""

    if np.isnan(value) or np.isinf(value):
        return "unknown"
    return f"{value * 100:.1f}%"


def _unknown_summary(short_window: int, long_window: int) -> Dict[str, str]:
    """Return a stable unknown summary when backtest inputs are incomplete."""

    return {
        "strategy": f"trend_following_ma{short_window}_ma{long_window}",
        "lookback_period": "unknown",
        "win_rate": "unknown",
        "max_drawdown": "unknown",
        "annualized_return": "unknown",
        "total_return": "unknown",
        "buy_hold_return": "unknown",
        "excess_return": "unknown",
    }
