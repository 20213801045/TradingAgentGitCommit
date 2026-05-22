"""Tests for the EVIR paper trading simulation."""

from __future__ import annotations

import csv
from pathlib import Path

from memory.storage import load_json
from models.schemas import MergeResult
from paper_trade import _parse_tickers, main as paper_trade_main
from paper_trading import run_paper_trading


def test_paper_trading_buys_and_records_outputs(tmp_path: Path) -> None:
    """A Buy recommendation should create a simulated position and artifacts."""

    def analyzer(ticker: str) -> tuple[MergeResult, str]:
        assert ticker == "AAA"
        return _merge_result("Buy", "medium"), "reports/AAA_report.md"

    result = run_paper_trading(
        tickers=["aaa"],
        initial_cash=10_000,
        max_position_weight=0.10,
        min_trade_value=100,
        output_dir=tmp_path,
        analyzer=analyzer,
        price_fetcher=lambda ticker: 100.0,
        reset=True,
    )

    portfolio = load_json(tmp_path / "portfolio.json")
    trades = _read_csv(tmp_path / "trades.csv")
    equity_rows = _read_csv(tmp_path / "equity_curve.csv")
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")

    assert result["trade_count"] == 1
    assert portfolio["positions"]["AAA"]["shares"] == 7.5
    assert portfolio["cash"] == 9250.0
    assert trades[0]["side"] == "BUY"
    assert trades[0]["recommendation"] == "Buy"
    assert equity_rows[0]["equity"] == "10000.0"
    assert "LLM：deepseek" in summary
    assert "AAA" in summary


def test_paper_trading_hold_keeps_position_and_updates_equity(
    tmp_path: Path,
) -> None:
    """A later Hold recommendation should keep the existing paper position."""

    run_paper_trading(
        tickers=["AAA"],
        initial_cash=10_000,
        max_position_weight=0.10,
        output_dir=tmp_path,
        analyzer=lambda ticker: (_merge_result("Buy", "high"), "report.md"),
        price_fetcher=lambda ticker: 100.0,
        reset=True,
    )
    result = run_paper_trading(
        tickers=["AAA"],
        initial_cash=10_000,
        max_position_weight=0.10,
        output_dir=tmp_path,
        analyzer=lambda ticker: (_merge_result("Hold", "medium"), "report.md"),
        price_fetcher=lambda ticker: 120.0,
    )

    portfolio = load_json(tmp_path / "portfolio.json")
    equity_rows = _read_csv(tmp_path / "equity_curve.csv")

    assert result["trade_count"] == 0
    assert portfolio["positions"]["AAA"]["shares"] == 10.0
    assert result["equity"] == 10200.0
    assert len(equity_rows) == 2


def test_paper_trading_sell_exits_existing_position(tmp_path: Path) -> None:
    """A Sell recommendation should close an existing paper position."""

    run_paper_trading(
        tickers=["AAA"],
        initial_cash=10_000,
        output_dir=tmp_path,
        analyzer=lambda ticker: (_merge_result("Buy", "high"), "report.md"),
        price_fetcher=lambda ticker: 100.0,
        reset=True,
    )
    result = run_paper_trading(
        tickers=["AAA"],
        initial_cash=10_000,
        output_dir=tmp_path,
        analyzer=lambda ticker: (_merge_result("Sell", "medium"), "report.md"),
        price_fetcher=lambda ticker: 90.0,
    )

    portfolio = load_json(tmp_path / "portfolio.json")
    trades = _read_csv(tmp_path / "trades.csv")

    assert result["trade_count"] == 1
    assert portfolio["positions"] == {}
    assert trades[-1]["side"] == "SELL"
    assert result["equity"] == 9900.0


def test_paper_trade_cli_passes_arguments(monkeypatch, tmp_path: Path, capsys) -> None:
    """The CLI should pass ticker, LLM, sizing, and reset options to the engine."""

    captured: dict[str, object] = {}

    def fake_run_paper_trading(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "equity": 10000.0,
            "cash": 9000.0,
            "decision_count": 2,
            "trade_count": 1,
            "positions_count": 1,
            "portfolio_path": str(tmp_path / "portfolio.json"),
            "trades_path": str(tmp_path / "trades.csv"),
            "equity_curve_path": str(tmp_path / "equity_curve.csv"),
            "summary_path": str(tmp_path / "summary.md"),
        }

    monkeypatch.setattr("paper_trade.run_paper_trading", fake_run_paper_trading)

    paper_trade_main(
        [
            "--tickers",
            "mu, intc",
            "--llm-provider",
            "deepseek",
            "--max-position",
            "0.2",
            "--reset",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert captured["tickers"] == ["MU", "INTC"]
    assert captured["llm_provider"] == "deepseek"
    assert captured["max_position_weight"] == 0.2
    assert captured["reset"] is True
    assert "EVIR 模拟盘运行完成" in output


def test_parse_tickers_normalizes_basic_input() -> None:
    """Ticker parsing should accept comma-separated user input."""

    assert _parse_tickers("mu, intc, NVDA") == ["MU", "INTC", "NVDA"]


def _merge_result(recommendation: str, confidence: str) -> MergeResult:
    return MergeResult(
        final_recommendation=recommendation,
        confidence=confidence,
        main_supporting_claims=[],
        main_opposing_claims=[],
        key_conflicts=[],
        risk_adjustment="测试风险说明",
        decision_rationale="测试决策理由",
        conditions_for_revision=[],
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
