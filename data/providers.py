"""Data provider abstractions for EVIR."""

from abc import ABC, abstractmethod
from typing import Any

from data.mock_data import get_mock_company_data


class BaseDataProvider(ABC):
    """Interface for company research data providers."""

    name: str

    @abstractmethod
    def fetch_company_data(self, ticker: str) -> dict[str, Any]:
        """Fetch company data for a ticker."""


class MockDataProvider(BaseDataProvider):
    """Deterministic local data provider used by tests and demos."""

    name = "mock"

    def fetch_company_data(self, ticker: str) -> dict[str, Any]:
        """Return local mock company data."""

        return get_mock_company_data(ticker)


class RealDataProvider(BaseDataProvider):
    """yfinance-backed provider for real market and financial data."""

    name = "real"

    def fetch_company_data(self, ticker: str) -> dict[str, Any]:
        """Return yfinance-backed company data in EVIR's input shape."""

        from data.real_data import get_real_company_data

        return get_real_company_data(ticker)


def get_data_provider(provider_name: str = "mock") -> BaseDataProvider:
    """Create a data provider by name."""

    normalized_name = provider_name.lower().strip()
    if normalized_name == "mock":
        return MockDataProvider()
    if normalized_name in {"real", "yfinance"}:
        return RealDataProvider()
    raise ValueError(
        f"Unsupported data provider '{provider_name}'. "
        "Available providers: mock, real."
    )
