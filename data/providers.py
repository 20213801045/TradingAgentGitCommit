"""Data provider factory for EVIR.

Supports:
- 'alpha_vantage' (primary, real-time)
- 'real' (yfinance)
- 'mock' (deterministic, for testing)
"""

from data.alpha_vantage import AlphaVantageProvider
from data.real_data import RealDataProvider
from data.mock_data import MockDataProvider
from data.base import BaseDataProvider


def get_data_provider(provider_name: str) -> BaseDataProvider:
    """Factory to get the right data provider."""

    name = provider_name.lower().strip()
    if name == "alpha_vantage" or name == "alphavantage":
        return AlphaVantageProvider()
    if name == "real":
        return RealDataProvider()
    if name == "mock":
        return MockDataProvider()
    raise ValueError(f"Unknown data provider: {provider_name}")
