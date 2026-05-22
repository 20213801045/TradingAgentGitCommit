"""Tests for technical indicator helpers."""

import pandas as pd

from data.technical_indicators import (
    build_technical_summary,
    calculate_moving_average,
    calculate_resistance_level,
    calculate_support_level,
    calculate_volatility,
)


def test_build_technical_summary_from_price_history() -> None:
    """Synthetic rising price history should produce agent-ready fields."""

    price_history = pd.DataFrame(
        {
            "Open": range(100, 160),
            "High": range(101, 161),
            "Low": range(99, 159),
            "Close": range(100, 160),
            "Volume": [1_000_000] * 60,
        },
        index=pd.date_range("2026-01-01", periods=60, freq="D"),
    )

    summary = build_technical_summary("AAPL", price_history)

    assert summary["price_trend"] == "above 20-day and 50-day moving averages"
    assert float(summary["rsi"]) >= 0
    assert summary["volatility"] in {"low", "moderate", "high"}
    assert summary["support_level"] == "99.00"
    assert summary["resistance_level"] == "160.00"


def test_indicator_helpers_tolerate_short_history() -> None:
    """Indicator helpers should return floats for short but valid histories."""

    price_history = pd.DataFrame(
        {
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.0, 11.0, 12.0],
        }
    )

    assert calculate_moving_average(price_history["Close"], 20) == 11.0
    assert calculate_support_level(price_history, 60) == 9.0
    assert calculate_resistance_level(price_history, 60) == 13.0
    assert calculate_volatility(price_history["Close"], 20) >= 0.0
