"""Tests for simple backtest helpers."""

import pandas as pd

from data.backtest import build_backtest_summary


def test_build_backtest_summary_from_rising_prices() -> None:
    """Rising prices should produce a complete numeric backtest summary."""

    price_history = pd.DataFrame(
        {
            "Open": range(100, 180),
            "High": range(101, 181),
            "Low": range(99, 179),
            "Close": range(100, 180),
            "Volume": [1_000_000] * 80,
        },
        index=pd.date_range("2026-01-01", periods=80, freq="D"),
    )

    summary = build_backtest_summary(price_history)

    assert summary["strategy"] == "trend_following_ma20_ma50"
    assert summary["lookback_period"] == "80 trading days"
    assert summary["win_rate"] != "unknown"
    assert summary["max_drawdown"] != "unknown"
    assert summary["annualized_return"] != "unknown"
    assert summary["buy_hold_return"] != "unknown"


def test_build_backtest_summary_tolerates_missing_close() -> None:
    """Missing Close data should return unknown fields instead of crashing."""

    summary = build_backtest_summary(pd.DataFrame({"Open": [1.0, 2.0]}))

    assert summary["win_rate"] == "unknown"
    assert summary["annualized_return"] == "unknown"
