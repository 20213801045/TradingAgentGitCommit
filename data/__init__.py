"""Data sources and provider abstractions for local EVIR demos."""

from data.mock_data import get_mock_company_data
from data.mock_new_evidence import get_mock_new_evidence
from data.providers import (
    BaseDataProvider,
    MockDataProvider,
    RealDataProvider,
    get_data_provider,
)
from data.real_new_evidence import get_real_new_evidence
from data.real_data import get_real_company_data

__all__ = [
    "BaseDataProvider",
    "MockDataProvider",
    "RealDataProvider",
    "get_data_provider",
    "get_mock_company_data",
    "get_mock_new_evidence",
    "get_real_company_data",
    "get_real_new_evidence",
]
