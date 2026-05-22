"""Tests for data provider abstractions."""

from typing import Any

import pytest

from data import BaseDataProvider, MockDataProvider, RealDataProvider, get_data_provider
from main import run_pipeline


def test_get_data_provider_returns_mock_provider() -> None:
    """The mock provider should be available by name."""

    provider = get_data_provider("mock")

    assert isinstance(provider, MockDataProvider)
    assert provider.fetch_company_data("AAPL")["ticker"] == "AAPL"


def test_get_data_provider_returns_real_provider_without_fetching() -> None:
    """The yfinance provider should be selectable without network access."""

    provider = get_data_provider("real")

    assert isinstance(provider, RealDataProvider)
    assert provider.name == "real"


def test_get_data_provider_rejects_unknown_provider() -> None:
    """Unknown data providers should fail loudly."""

    with pytest.raises(ValueError, match="Unsupported data provider"):
        get_data_provider("real-api")


def test_run_pipeline_accepts_custom_data_provider(tmp_path, capsys) -> None:
    """The pipeline should accept provider objects for future integrations."""

    class CustomProvider(BaseDataProvider):
        name = "custom"

        def fetch_company_data(self, ticker: str) -> dict[str, Any]:
            assert ticker == "AAPL"
            return MockDataProvider().fetch_company_data(ticker)

    workspace, _, markdown_path, json_path = run_pipeline(
        ticker="AAPL",
        data_provider=CustomProvider(),
        llm_provider="none",
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        preview_lines=0,
    )
    capsys.readouterr()

    assert workspace.ticker == "AAPL"
    assert markdown_path.endswith("AAPL_report.md")
    assert json_path.endswith("AAPL_investment_report.json")
