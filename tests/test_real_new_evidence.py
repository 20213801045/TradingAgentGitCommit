"""Tests for real revision evidence adapter shape."""

import pandas as pd

from data.real_new_evidence import get_real_new_evidence


def test_get_real_new_evidence_builds_revision_input(monkeypatch) -> None:
    """Real new evidence should match RevisionEngine's expected input shape."""

    def fake_fetch_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
        assert ticker == "MU"
        assert period == "6mo"
        return pd.DataFrame(
            {
                "Open": range(100, 160),
                "High": range(101, 161),
                "Low": range(99, 159),
                "Close": range(100, 160),
                "Volume": [1_000_000] * 60,
            },
            index=pd.date_range("2026-01-01", periods=60, freq="D"),
        )

    def fake_fetch_financial_summary(ticker: str) -> dict[str, str]:
        assert ticker == "MU"
        return {
            "revenue_growth_yoy": "42.0%",
            "gross_margin": "58.4%",
            "net_margin": "32.0%",
            "forward_pe": "9.5",
            "cash_position": "Strong",
        }

    monkeypatch.setattr(
        "data.real_new_evidence.fetch_price_history",
        fake_fetch_price_history,
    )
    monkeypatch.setattr(
        "data.real_new_evidence.fetch_financial_summary",
        fake_fetch_financial_summary,
    )

    new_evidence = get_real_new_evidence("MU")

    assert new_evidence["ticker"] == "MU"
    assert new_evidence["data_source"] == "yfinance"
    assert new_evidence["new_financial_metrics"]["revenue_growth_yoy"] == "42.0%"
    assert new_evidence["new_financial_metrics"]["forward_pe"] == "9.5"
    assert new_evidence["new_technical_indicators"]["price_trend"] == (
        "above 20-day and 50-day moving averages"
    )
    assert new_evidence["new_events"][0]["source"] == "Yahoo Finance via yfinance"
