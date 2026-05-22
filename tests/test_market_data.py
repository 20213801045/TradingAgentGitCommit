"""Tests for market data fallback behavior."""

import json

import pandas as pd

from data.market_data import fetch_price_history


def test_fetch_price_history_uses_yahoo_chart_fallback(monkeypatch, tmp_path) -> None:
    """Yahoo chart fallback should work when yfinance raises."""

    monkeypatch.setattr("data.cache.DEFAULT_CACHE_DIR", tmp_path / "cache")

    class FailingTicker:
        def __init__(self, ticker: str) -> None:
            del ticker

        def history(self, period: str, auto_adjust: bool) -> pd.DataFrame:
            del period
            del auto_adjust
            raise RuntimeError("Too Many Requests")

    class FakeYFinance:
        @staticmethod
        def Ticker(ticker: str) -> FailingTicker:
            return FailingTicker(ticker)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type
            del exc
            del traceback

        def read(self) -> bytes:
            return json.dumps(
                {
                    "chart": {
                        "result": [
                            {
                                "timestamp": [1_700_000_000, 1_700_086_400],
                                "indicators": {
                                    "quote": [
                                        {
                                            "open": [10.0, 11.0],
                                            "high": [11.0, 12.0],
                                            "low": [9.5, 10.5],
                                            "close": [10.5, 11.5],
                                            "volume": [1000, 1100],
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout: int):
        del request
        del timeout
        return FakeResponse()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYFinance())
    monkeypatch.setattr("data.market_data.urlopen", fake_urlopen)

    history = fetch_price_history("INTC")

    assert list(history.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(history) == 2
    assert history["Close"].iloc[-1] == 11.5


def test_fetch_price_history_reads_fresh_cache(monkeypatch, tmp_path) -> None:
    """Fresh price cache should avoid external provider calls."""

    monkeypatch.setattr("data.cache.DEFAULT_CACHE_DIR", tmp_path / "cache")

    first_history = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.5],
            "Close": [10.5],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-01-01"]),
    )

    class FirstTicker:
        def __init__(self, ticker: str) -> None:
            del ticker

        def history(self, period: str, auto_adjust: bool) -> pd.DataFrame:
            del period
            del auto_adjust
            return first_history

    class ExplodingTicker:
        def __init__(self, ticker: str) -> None:
            del ticker

        def history(self, period: str, auto_adjust: bool) -> pd.DataFrame:
            del period
            del auto_adjust
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
    assert fetch_price_history("INTC")["Close"].iloc[-1] == 10.5

    monkeypatch.setitem(__import__("sys").modules, "yfinance", ExplodingYFinance())
    assert fetch_price_history("INTC")["Close"].iloc[-1] == 10.5
