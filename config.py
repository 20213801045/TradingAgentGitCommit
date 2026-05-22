"""Project-level configuration for EVIR."""

import os
from pathlib import Path


def _load_project_env(env_path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip().strip('"').strip("'")
        if normalized_key and normalized_key not in os.environ:
            os.environ[normalized_key] = normalized_value


_load_project_env()


PROJECT_NAME = "EVIR"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_REPORT_DIR = DEFAULT_OUTPUT_DIR / "reports"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
DEFAULT_TICKER = "AAPL"
ENABLE_DATA_CACHE = True
PRICE_CACHE_TTL_DAYS = 1
FINANCIAL_CACHE_TTL_DAYS = 1
ENABLE_LLM_CACHE = os.getenv("ENABLE_LLM_CACHE", "true").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
LLM_CACHE_TTL_HOURS = int(os.getenv("LLM_CACHE_TTL_HOURS", "6"))
DEFAULT_PAPER_TRADING_DIR = DEFAULT_OUTPUT_DIR / "paper_trading"
DEFAULT_PAPER_TRADING_CASH = 100_000.0
DEFAULT_PAPER_TRADING_MAX_POSITION = 0.10
DEFAULT_PAPER_TRADING_MIN_TRADE_VALUE = 100.0
DEFAULT_PAPER_TRADING_TICKERS = [
    "MU",
    "INTC",
    "NVDA",
    "AMD",
    "AVGO",
    "QCOM",
    "TSM",
    "ASML",
    "AMAT",
    "LRCX",
    "KLAC",
    "WDC",
    "STX",
    "PSTG",
    "NTAP",
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "ORCL",
]
USE_REAL_DATA = False
USE_LLM = True
LLM_PROVIDER = "deepseek"
MODEL_NAME = "deepseek-v4-pro"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_TIMEOUT_SECONDS = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))
DEFAULT_LLM_PROVIDER = LLM_PROVIDER
DEFAULT_DEEPSEEK_BASE_URL = DEEPSEEK_BASE_URL
DEFAULT_DEEPSEEK_MODEL = MODEL_NAME
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = DEEPSEEK_TIMEOUT_SECONDS

EVIDENCE_RECENCY_THRESHOLDS_DAYS = {
    "technical_stale": 7,
    "technical_expired": 14,
    "news_stale": 30,
    "news_expired": 90,
    "financial_stale": 180,
    "financial_expired": 365,
    "default_stale": 90,
}

INVESTMENT_THRESHOLDS = {
    "high_support_score": 0.75,
    "medium_support_score": 0.55,
    "medium_risk_score": 0.45,
    "high_risk_score": 0.65,
    "weak_evidence_score": 0.50,
    "weak_or_stale_ratio": 0.60,
    "low_confidence_stale_or_weak_ratio": 0.40,
    "high_confidence_score": 0.75,
    "low_valuation_pe": 15.0,
    "high_valuation_pe": 25.0,
    "revision_high_valuation_pe": 28.0,
    "weak_growth_percent": 5.0,
    "healthy_growth_percent": 8.0,
    "weak_net_margin_percent": 15.0,
    "revision_weak_net_margin_percent": 22.0,
    "bullish_rsi": 55.0,
    "weak_rsi": 50.0,
    "low_volatility": 0.20,
    "high_volatility": 0.40,
}
