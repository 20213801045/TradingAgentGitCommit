"""Run the EVIR LLM-assisted paper trading account."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from config import (
    DEFAULT_PAPER_TRADING_CASH,
    DEFAULT_PAPER_TRADING_DIR,
    DEFAULT_PAPER_TRADING_MAX_POSITION,
    DEFAULT_PAPER_TRADING_MIN_TRADE_VALUE,
    DEFAULT_PAPER_TRADING_TICKERS,
    LLM_PROVIDER,
)
from paper_trading import run_paper_trading


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and run one paper-trading rebalance."""

    args = _parse_args(argv)
    result = run_paper_trading(
        tickers=_parse_tickers(args.tickers),
        initial_cash=args.cash,
        max_position_weight=args.max_position,
        min_trade_value=args.min_trade_value,
        output_dir=args.output_dir,
        llm_provider=args.llm_provider,
        data_provider=args.data_provider,
        verbose=args.verbose,
        reset=args.reset,
    )

    print("EVIR 模拟盘运行完成")
    print(f"当前净值：{result['equity']}")
    print(f"当前现金：{result['cash']}")
    print(f"本次决策数：{result['decision_count']}")
    print(f"本次交易数：{result['trade_count']}")
    print(f"持仓数量：{result['positions_count']}")
    print(f"组合状态：{result['portfolio_path']}")
    print(f"交易流水：{result['trades_path']}")
    print(f"净值曲线：{result['equity_curve_path']}")
    print(f"摘要报告：{result['summary_path']}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run one EVIR paper-trading simulation step.",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_PAPER_TRADING_TICKERS),
        help="Comma-separated ticker list for this run.",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=DEFAULT_PAPER_TRADING_CASH,
        help="Initial cash used when no portfolio exists, or when --reset is set.",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=DEFAULT_PAPER_TRADING_MAX_POSITION,
        help="Maximum target weight for one ticker.",
    )
    parser.add_argument(
        "--min-trade-value",
        type=float,
        default=DEFAULT_PAPER_TRADING_MIN_TRADE_VALUE,
        help="Skip simulated trades smaller than this dollar amount.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PAPER_TRADING_DIR,
        help="Directory for portfolio, trades, equity curve, and summary.",
    )
    parser.add_argument(
        "--data-provider",
        default="real",
        help="Data provider for the EVIR research pipeline. Defaults to real.",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help=(
            "LLM provider for EVIR analysis. Omit to use config default "
            f"({LLM_PROVIDER}); use none to disable."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start a fresh paper account and overwrite the saved portfolio.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full EVIR pipeline output for each ticker.",
    )
    return parser.parse_args(argv)


def _parse_tickers(raw_tickers: str) -> list[str]:
    """Parse comma-separated tickers into a normalized list."""

    return [
        ticker.strip().upper()
        for ticker in raw_tickers.split(",")
        if ticker.strip()
    ]


if __name__ == "__main__":
    main()
