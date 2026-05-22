"""Paper trading engine for evaluating EVIR recommendations."""

from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PAPER_TRADING_CASH,
    DEFAULT_PAPER_TRADING_DIR,
    DEFAULT_PAPER_TRADING_MAX_POSITION,
    DEFAULT_PAPER_TRADING_MIN_TRADE_VALUE,
    DEFAULT_PAPER_TRADING_TICKERS,
    DEFAULT_REPORT_DIR,
    LLM_PROVIDER,
)
from data.market_data import fetch_price_history
from main import run_pipeline
from memory.storage import load_json, save_json
from models.schemas import MergeResult


TECH_STORAGE_TICKERS = tuple(DEFAULT_PAPER_TRADING_TICKERS)


@dataclass(frozen=True)
class PaperDecision:
    """One ticker decision used by the paper trading engine."""

    ticker: str
    recommendation: str
    confidence: str
    price: float
    report_path: str
    rationale: str


Analyzer = Callable[[str], tuple[MergeResult, str]]
PriceFetcher = Callable[[str], float]


def run_paper_trading(
    tickers: Iterable[str] = TECH_STORAGE_TICKERS,
    initial_cash: float = DEFAULT_PAPER_TRADING_CASH,
    max_position_weight: float = DEFAULT_PAPER_TRADING_MAX_POSITION,
    min_trade_value: float = DEFAULT_PAPER_TRADING_MIN_TRADE_VALUE,
    output_dir: str | Path = DEFAULT_PAPER_TRADING_DIR,
    llm_provider: str | None = None,
    data_provider: str = "real",
    analyzer: Analyzer | None = None,
    price_fetcher: PriceFetcher | None = None,
    verbose: bool = False,
    reset: bool = False,
) -> dict[str, object]:
    """Run one paper-trading rebalance using EVIR recommendations.

    By default, ``llm_provider=None`` means the existing project config decides
    which LLM to use. In the current config this calls DeepSeek, so the paper
    account uses the same LLM-assisted research pipeline as normal reports.
    """

    normalized_tickers = _normalize_tickers(tickers)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    portfolio_path = output_path / "portfolio.json"
    trades_path = output_path / "trades.csv"
    equity_path = output_path / "equity_curve.csv"
    summary_path = output_path / "summary.md"

    portfolio = _load_or_create_portfolio(portfolio_path, initial_cash, reset)
    decisions: list[PaperDecision] = []
    trades: list[dict[str, object]] = []
    prices: dict[str, float] = {}
    now = datetime.now(timezone.utc).isoformat()
    analyze = analyzer or (
        lambda ticker: _run_evir_analysis(
            ticker,
            data_provider=data_provider,
            llm_provider=llm_provider,
            verbose=verbose,
        )
    )
    get_price = price_fetcher or _latest_close_price

    for ticker in normalized_tickers:
        try:
            merge_result, report_path = analyze(ticker)
            price = get_price(ticker)
        except Exception as error:
            decisions.append(
                PaperDecision(
                    ticker=ticker,
                    recommendation="Error",
                    confidence="low",
                    price=0.0,
                    report_path="",
                    rationale=str(error),
                )
            )
            continue

        prices[ticker] = price
        decision = PaperDecision(
            ticker=ticker,
            recommendation=merge_result.final_recommendation,
            confidence=merge_result.confidence,
            price=price,
            report_path=report_path,
            rationale=merge_result.risk_adjustment,
        )
        decisions.append(decision)
        trade = _apply_decision(
            portfolio,
            decision,
            prices,
            max_position_weight,
            min_trade_value,
            now,
        )
        if trade is not None:
            trades.append(trade)

    _refresh_missing_position_prices(portfolio, prices, get_price)
    equity = _portfolio_value(portfolio, prices)
    portfolio["updated_at"] = now
    portfolio["last_equity"] = round(equity, 2)
    save_json(portfolio_path, portfolio)
    _append_csv(trades_path, trades, _trade_columns())
    _append_csv(
        equity_path,
        [
            {
                "timestamp": now,
                "equity": round(equity, 2),
                "cash": round(float(portfolio["cash"]), 2),
                "position_value": round(equity - float(portfolio["cash"]), 2),
                "positions_count": len(_positions(portfolio)),
                "tickers": ",".join(normalized_tickers),
            }
        ],
        [
            "timestamp",
            "equity",
            "cash",
            "position_value",
            "positions_count",
            "tickers",
        ],
    )
    summary = _write_summary(
        summary_path,
        portfolio,
        decisions,
        trades,
        equity,
        normalized_tickers,
        llm_provider,
        data_provider,
        max_position_weight,
        min_trade_value,
    )

    return {
        "portfolio_path": str(portfolio_path),
        "trades_path": str(trades_path),
        "equity_curve_path": str(equity_path),
        "summary_path": str(summary_path),
        "equity": round(equity, 2),
        "cash": round(float(portfolio["cash"]), 2),
        "trade_count": len(trades),
        "decision_count": len(decisions),
        "positions_count": len(_positions(portfolio)),
        "summary": summary,
    }


def _run_evir_analysis(
    ticker: str,
    data_provider: str,
    llm_provider: str | None,
    verbose: bool,
) -> tuple[MergeResult, str]:
    """Run the existing EVIR pipeline and return the merge result and report path."""

    if verbose:
        _, merge_result, markdown_path, _ = run_pipeline(
            ticker=ticker,
            data_provider=data_provider,
            llm_provider=llm_provider,
            output_dir=DEFAULT_OUTPUT_DIR,
            report_dir=DEFAULT_REPORT_DIR,
            preview_lines=0,
        )
        return merge_result, markdown_path

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _, merge_result, markdown_path, _ = run_pipeline(
            ticker=ticker,
            data_provider=data_provider,
            llm_provider=llm_provider,
            output_dir=DEFAULT_OUTPUT_DIR,
            report_dir=DEFAULT_REPORT_DIR,
            preview_lines=0,
        )
    return merge_result, markdown_path


def _latest_close_price(ticker: str) -> float:
    """Return the latest close price for a ticker."""

    history = fetch_price_history(ticker, period="6mo")
    close = history["Close"].dropna()
    if close.empty:
        raise ValueError(f"No close price available for {ticker}.")
    return float(close.iloc[-1])


def _apply_decision(
    portfolio: dict[str, object],
    decision: PaperDecision,
    prices: dict[str, float],
    max_position_weight: float,
    min_trade_value: float,
    timestamp: str,
) -> dict[str, object] | None:
    """Apply one recommendation to the paper portfolio."""

    positions = _positions(portfolio)
    current_position = positions.get(decision.ticker, {"shares": 0, "avg_cost": 0.0})
    current_shares = float(current_position.get("shares", 0))
    current_value = current_shares * decision.price
    target_weight = _target_weight(
        decision.recommendation,
        decision.confidence,
        max_position_weight,
        current_shares,
    )
    equity = _portfolio_value(portfolio, prices)
    target_value = target_weight * equity
    delta_value = target_value - current_value

    if decision.recommendation in {"Sell", "Avoid"}:
        if current_shares <= 0:
            return None
        proceeds = current_shares * decision.price
        if proceeds < min_trade_value:
            return None
        portfolio["cash"] = float(portfolio["cash"]) + proceeds
        positions.pop(decision.ticker, None)
        return _trade_row(
            timestamp,
            decision,
            "SELL",
            current_shares,
            proceeds,
            "recommendation_exit",
        )

    if decision.recommendation != "Buy" or delta_value < min_trade_value:
        return None

    available_cash = float(portfolio["cash"])
    trade_value = min(delta_value, available_cash)
    if trade_value < min_trade_value:
        return None
    shares_to_buy = trade_value / decision.price
    if shares_to_buy <= 0:
        return None

    cost = shares_to_buy * decision.price
    new_shares = current_shares + shares_to_buy
    previous_cost = current_shares * float(current_position.get("avg_cost", 0.0))
    avg_cost = (previous_cost + cost) / new_shares
    positions[decision.ticker] = {
        "shares": round(new_shares, 6),
        "avg_cost": round(avg_cost, 4),
    }
    portfolio["cash"] = available_cash - cost
    return _trade_row(
        timestamp,
        decision,
        "BUY",
        shares_to_buy,
        cost,
        "recommendation_entry",
    )


def _target_weight(
    recommendation: str,
    confidence: str,
    max_position_weight: float,
    current_shares: float,
) -> float:
    """Return target portfolio weight for a recommendation."""

    if recommendation == "Buy":
        return max_position_weight * _confidence_multiplier(confidence)
    if recommendation == "Hold" and current_shares > 0:
        return max_position_weight * 0.5 * _confidence_multiplier(confidence)
    return 0.0


def _confidence_multiplier(confidence: str) -> float:
    """Return conservative sizing multiplier from confidence."""

    normalized = confidence.lower().strip()
    if normalized == "high":
        return 1.0
    if normalized == "medium":
        return 0.75
    return 0.5


def _portfolio_value(
    portfolio: dict[str, object],
    prices: dict[str, float],
) -> float:
    """Value the portfolio using latest known prices."""

    value = float(portfolio["cash"])
    for ticker, position in _positions(portfolio).items():
        shares = float(position.get("shares", 0))
        price = prices.get(ticker, float(position.get("avg_cost", 0.0)))
        value += shares * price
    return value


def _load_or_create_portfolio(
    portfolio_path: Path,
    initial_cash: float,
    reset: bool = False,
) -> dict[str, object]:
    """Load portfolio state or initialize a new one."""

    if portfolio_path.exists() and not reset:
        data = load_json(portfolio_path)
        if isinstance(data, dict) and "cash" in data and "positions" in data:
            return data
    now = datetime.now(timezone.utc).isoformat()
    return {
        "created_at": now,
        "updated_at": now,
        "initial_cash": float(initial_cash),
        "cash": float(initial_cash),
        "positions": {},
        "last_equity": float(initial_cash),
    }


def _positions(portfolio: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return mutable positions dictionary."""

    positions = portfolio.setdefault("positions", {})
    if not isinstance(positions, dict):
        portfolio["positions"] = {}
        return portfolio["positions"]
    return positions


def _trade_row(
    timestamp: str,
    decision: PaperDecision,
    side: str,
    shares: float,
    trade_value: float,
    reason: str,
) -> dict[str, object]:
    """Build one trade CSV row."""

    return {
        "timestamp": timestamp,
        "ticker": decision.ticker,
        "side": side,
        "shares": round(shares, 6),
        "price": round(decision.price, 4),
        "trade_value": round(trade_value, 2),
        "recommendation": decision.recommendation,
        "confidence": decision.confidence,
        "reason": reason,
        "report_path": decision.report_path,
    }


def _trade_columns() -> list[str]:
    """Return stable trade CSV columns."""

    return [
        "timestamp",
        "ticker",
        "side",
        "shares",
        "price",
        "trade_value",
        "recommendation",
        "confidence",
        "reason",
        "report_path",
    ]


def _append_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Append rows to a CSV file, writing a header when needed."""

    if not rows:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=columns).writeheader()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    portfolio: dict[str, object],
    decisions: list[PaperDecision],
    trades: list[dict[str, object]],
    equity: float,
    tickers: list[str],
    llm_provider: str | None,
    data_provider: str,
    max_position_weight: float,
    min_trade_value: float,
) -> str:
    """Write a Markdown summary for the latest paper-trading run."""

    lines = [
        "# EVIR 模拟盘摘要",
        "",
        "## 运行设置",
        "",
        f"- 股票池：{', '.join(tickers)}",
        f"- 数据源：{data_provider}",
        f"- LLM：{llm_provider or LLM_PROVIDER}",
        f"- 单票最高目标仓位：{max_position_weight:.1%}",
        f"- 最小交易金额：{min_trade_value:.2f}",
        "",
        "## 组合状态",
        "",
        f"- 当前净值：{equity:.2f}",
        f"- 当前现金：{float(portfolio['cash']):.2f}",
        f"- 累计收益率：{_portfolio_return(portfolio, equity):.2%}",
        f"- 持仓数量：{len(_positions(portfolio))}",
        f"- 本次交易数：{len(trades)}",
        "",
        "## 本次决策",
        "",
        "| Ticker | 建议 | 置信度 | 价格 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for decision in decisions:
        lines.append(
            "| "
            + " | ".join(
                [
                    decision.ticker,
                    decision.recommendation,
                    decision.confidence,
                    f"{decision.price:.2f}" if decision.price else "N/A",
                    _escape_table_cell(decision.rationale[:120]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 当前持仓", "", "| Ticker | 股数 | 平均成本 |", "|---|---|---|"])
    positions = _positions(portfolio)
    if not positions:
        lines.append("| 无 | 0 | 0 |")
    for ticker, position in positions.items():
        lines.append(
            f"| {ticker} | {position.get('shares', 0)} | "
            f"{float(position.get('avg_cost', 0.0)):.4f} |"
        )

    summary = "\n".join(lines) + "\n"
    path.write_text(summary, encoding="utf-8")
    return summary


def _refresh_missing_position_prices(
    portfolio: dict[str, object],
    prices: dict[str, float],
    price_fetcher: PriceFetcher,
) -> None:
    """Fetch latest prices for existing positions not in this run's ticker list."""

    for ticker in _positions(portfolio):
        if ticker in prices:
            continue
        try:
            prices[ticker] = price_fetcher(ticker)
        except Exception:
            continue


def _portfolio_return(portfolio: dict[str, object], equity: float) -> float:
    """Return total paper-account return since initialization."""

    initial_cash = float(portfolio.get("initial_cash", equity) or equity)
    if initial_cash <= 0:
        return 0.0
    return (equity - initial_cash) / initial_cash


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    """Normalize and dedupe tickers while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        clean = ticker.upper().strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


def _escape_table_cell(value: str) -> str:
    """Escape Markdown table separators."""

    return value.replace("|", "\\|").replace("\n", " ")
