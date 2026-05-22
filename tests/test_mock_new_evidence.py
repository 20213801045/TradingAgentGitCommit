"""Tests for ticker-specific mock revision evidence."""

from data import get_mock_new_evidence


def test_mock_new_evidence_is_ticker_specific() -> None:
    """Mock revision evidence should not always return AAPL."""

    aapl_evidence = get_mock_new_evidence("AAPL")
    mu_evidence = get_mock_new_evidence("MU")
    generic_evidence = get_mock_new_evidence("MSFT")

    assert aapl_evidence["ticker"] == "AAPL"
    assert mu_evidence["ticker"] == "MU"
    assert generic_evidence["ticker"] == "MSFT"
    assert "Mock Semiconductor News" == mu_evidence["new_events"][0]["source"]
    assert "AAPL" not in generic_evidence["new_events"][0]["title"]
