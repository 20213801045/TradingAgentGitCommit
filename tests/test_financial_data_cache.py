"""Tests for financial data caching."""

from data.financial_data import fetch_financial_summary


def test_fetch_financial_summary_reads_fresh_cache(monkeypatch, tmp_path) -> None:
    """Fresh financial cache should avoid repeated yfinance info calls."""

    monkeypatch.setattr("data.cache.DEFAULT_CACHE_DIR", tmp_path / "cache")

    class FirstTicker:
        def __init__(self, ticker: str) -> None:
            del ticker

        @property
        def info(self) -> dict[str, object]:
            return {
                "revenueGrowth": 0.12,
                "grossMargins": 0.50,
                "profitMargins": 0.25,
                "forwardPE": 12.5,
                "totalCash": 12_000_000_000,
            }

    class ExplodingTicker:
        def __init__(self, ticker: str) -> None:
            del ticker

        @property
        def info(self) -> dict[str, object]:
            raise AssertionError("external provider should not be called")

    class FirstYFinance:
        @staticmethod
        def Ticker(ticker: str) -> FirstTicker:
            return FirstTicker(ticker)

    class ExplodingYFinance:
        @staticmethod
        def Ticker(ticker: str) -> ExplodingTicker:
            return ExplodingTicker(ticker)

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FirstYFinance())
    first_summary = fetch_financial_summary("INTC")

    monkeypatch.setitem(__import__("sys").modules, "yfinance", ExplodingYFinance())
    second_summary = fetch_financial_summary("INTC")

    assert first_summary["revenue_growth_yoy"] == "12.0%"
    assert second_summary["revenue_growth_yoy"] == "12.0%"
