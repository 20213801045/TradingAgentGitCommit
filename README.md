# EVIR: Evidence-Versioned Investment Research

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-82%20passed-brightgreen)](tests/)

> 🚀 **v2.0** — LLM-first multi-agent investment research system with dialectical debate.

EVIR is a Python multi-agent system that produces **actionable trade plans** (Buy/Sell/Hold
with entry/stop/target/position sizing) through structured, evidence-backed research.
Six LLM-powered specialist agents analyse a stock, then a debate agent simulates a
three-member investment committee (Bull / Bear / Chair) to make the final call.

⚠️ This is a research and engineering prototype. It is **not financial advice** and
does not place real trades.

---

## 🎯 What EVIR Does

1. **Collects data** — Alpha Vantage (primary) or yfinance (fallback) for price,
   financials, technical indicators, and macro data.
2. **Runs 6 specialist agents** in parallel, each producing evidence-linked claims.
3. **Audits risk** and generates counter-evidence against positive claims.
4. **Debates** — LLM simulates a Bull / Bear / Chair committee → Buy / Sell / Hold.
5. **Produces a trade plan** — concrete entry range, stop-loss, targets, position size.
6. **Revision check** — evaluates new evidence against the original decision.
7. **Closed-loop learning** — prediction tracker evaluates historical accuracy.

---

## 🔗 The Evidence Chain (What Makes This Different)

Most AI stock tools give you a Buy/Sell output and you're supposed to trust it.
EVIR doesn't — **every claim is traceable back to its source data.**

Each agent produces a structured `ClaimEvidenceCommit` that bundles:

| Field | What it tracks |
|-------|----------------|
| `evidence.source` | Where the data came from (e.g. Alpha Vantage, SEC filing, macro report) |
| `evidence.source_type` | Type label — `financial_data_provider`, `official_report`, `technical_indicator`, etc. |
| `evidence.timestamp` | When the data was observed |
| `evidence.metric_name` / `metric_value` | The specific metric and its value |
| `evidence_quality_score` | Machine-calculated quality score (0–1), based on source reliability, specificity, relevance, and recency — **not self-reported by the agent** |
| `confidence` | Agent's own confidence: `low` / `medium` / `high` |
| `agent_role` | Which agent made the claim |
| `created_at` | UTC timestamp of the claim |

All commits are stored in a ticker-specific JSON workspace under versioned branches.
The final recommendation can be **audited end-to-end**:

```
Final Decision → MergeResult → Branch → Commit → Evidence → Source Data
```

You can trace any "Buy" or "Sell" all the way back to which agent said what, based on
which data point, with what confidence — and check if the evidence score holds up.

### PredictionEvaluator — Closed-Loop Learning

The `PredictionEvaluator` tracks every trade plan prediction (entry, direction, target)
against actual market outcomes over time. It computes:

- **Directional accuracy** — was the Buy/Sell call correct?
- **Timing score** — did the entry window capture a good price?
- **Calibration** — are confidence labels (`high`/`medium`/`low`) backed by results?

This feedback feeds back into the system, self-calibrating agent behavior so that
overconfident agents get tempered and accurate agents gain more weight in future debates.

---

## 🏗️ Architecture (v2.0)

```
DATA LAYER                    AGENT LAYER                     DECISION LAYER
┌──────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐
│ Alpha Vantage │────▶│ ResearchCoordinator     │     │                     │
│  (primary)   │     │  (research plan)        │     │   DebateAgent       │
├──────────────┤     ├─────────────────────────┤     │   ┌─────────────┐   │
│  yfinance    │     │ DeepResearchAgent       │     │   │ 🟢 BULL     │   │
│  (fallback)  │     │  (fundamental/valuation)│────▶│   │ 🔴 BEAR     │──▶│ Buy/Sell/Hold
├──────────────┤     ├─────────────────────────┤     │   │ ⚪ CHAIR    │   │
│  Mock data   │     │ MacroSentimentAgent     │     │   └─────────────┘   │
│  (testing)   │     │  (macro/rate/sentiment) │     └─────────────────────┘
└─────────────┘     ├─────────────────────────┤              │
                     │ TechnicalTimingAgent    │              ▼
                     │  (trend/momentum/timing)│     ┌─────────────────────┐
                     ├─────────────────────────┤     │ TradePlanReportAgent │
                     │ RiskAgent               │     │ (entry/stop/target/  │
                     │  (risk audit)           │     │  position/risk)      │
                     ├─────────────────────────┤     └─────────────────────┘
                     │ CounterEvidenceAgent    │
                     │  (challenge bull case)  │
                     └─────────────────────────┘
```

### Agent roster

| Agent | Role | LLM | Output |
|-------|------|-----|--------|
| `ResearchCoordinatorAgent` | Research plan, coverage review, pre-merge audit | ✅ | Plan & review commits |
| `DeepResearchAgent` | Fundamental, financial, valuation, industry analysis | ✅ | Evidence commits |
| `MacroSentimentAgent` | Macro environment, rates, FX, sentiment | ✅ | Evidence commits |
| `TechnicalTimingAgent` | Price trend, momentum, volatility, timing signals | ✅ | Evidence commits |
| `RiskAgent` | Risk audit across all branches | ✅ | Risk commits |
| `CounterEvidenceAgent` | Challenge positive claims | ✅ | Counter commits |
| `DebateAgent` | Bull vs Bear vs Chair → verdict | ✅ | Buy/Sell/Hold + price levels |
| `TradePlanReportAgent` | Actionable trade plan | — | Markdown + JSON report |

> 13 legacy template-driven agents (v1.x) are preserved in `agents/legacy/` for reference.

---

## 🖥️ Web Application

EVIR includes a **Flask-based chat platform** with two modes:

| Mode | Description | Endpoint |
|------|-------------|----------|
| 💬 **AI Chat** | General conversation via DeepSeek (streaming) | `/api/chat/stream` |
| 📊 **Stock Evaluation** | Full multi-agent pipeline with live progress (SSE) | `/api/evir/analyze` |

### Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r webapp/requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — add DEEPSEEK_API_KEY (required) and ALPHA_VANTAGE_API_KEY (optional)

# 3. Launch
python webapp/app.py
# Open http://localhost:5000
```

### Features
- **Dark theme UI** with collapsible sidebar
- **SSE streaming** for both chat responses and pipeline progress
- **Live progress tracking** — see each agent stage as it completes
- **Result dashboard** — recommendation, confidence, risk level, timing score, full report
- **Responsive design** — works on desktop and mobile
- **Keyboard shortcuts** — `Ctrl+B` toggles sidebar

---

## 🔧 CLI Usage

```bash
# Basic run (uses config defaults)
python main.py

# With specific ticker and real data
python main.py --ticker AAPL --data-provider real

# Paper trading simulation
python paper_trade.py

# Custom ticker universe
python paper_trade.py --tickers MU,INTC,NVDA,AMD

# Disable LLM (fallback mode)
python main.py --llm-provider none
```

---

## 📂 Project Layout

```
EVIR/
├── main.py                  # CLI pipeline entry point
├── paper_trade.py           # Paper trading CLI
├── config.py                # Global configuration
├── requirements.txt         # Core dependencies
│
├── agents/                  # Active v2.0 agents (8)
│   ├── debate_agent.py             # LLM Bull/Bear/Chair debate
│   ├── trade_plan_report_agent.py  # Trade plan report generator
│   ├── deep_research_agent.py      # LLM fundamental/valuation
│   ├── macro_sentiment_agent.py    # LLM macro/sentiment
│   ├── technical_timing_agent.py   # LLM technical/timing
│   ├── research_coordinator_agent.py
│   ├── risk_agent.py
│   ├── counter_evidence_agent.py
│   ├── base_agent.py               # Base class
│   └── legacy/                     # v1.x template agents (13, deprecated)
│
├── data/                    # Alpha Vantage, yfinance, caching
├── llm/                     # DeepSeek client, caching, JSON utils
├── evidence/                # Evidence scoring & temporal checks
├── revision/                # Decision revision engine
├── evaluation/              # Audit & quality metrics
├── memory/                  # Workspace JSON storage
├── paper_trading/           # Simulated trading engine
├── models/                  # Pydantic schemas
│
├── webapp/                  # Flask chat platform
│   ├── app.py                      # Flask application
│   ├── backend/
│   │   ├── chat.py                 # DeepSeek chat (streaming)
│   │   └── evir_runner.py          # Pipeline wrapper with SSE
│   ├── templates/index.html        # Chat + eval UI
│   ├── static/css/style.css        # Dark theme
│   └── static/js/app.js            # Frontend logic
│
├── outputs/                 # Generated workspaces, reports, caches
└── tests/                   # Test suite (82 tests, all passing ✅)
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```bash
# Required — AI chat + agent intelligence
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-pro

# Optional — for richer fundamental/macro data
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Optional — controls
ENABLE_LLM_CACHE=true
LLM_CACHE_TTL_HOURS=6
```

---

## 📊 Generated Outputs

```
outputs/
├── workspaces/<TICKER>_workspace.json    # Full audit trail
├── reports/<TICKER>_trade_plan.md       # Trade plan (Markdown)
├── reports/<TICKER>_trade_plan.json     # Trade plan (JSON)
├── reports/<TICKER>_revision_report.md  # Revision analysis
└── paper_trading/                        # Simulated portfolio results
```

---

## 🧪 Tests

```bash
python -m pytest    # 82 tests, all passing ✅
```

---

## 📝 Current Limitations

- Real data quality depends on Alpha Vantage / yfinance availability
- Paper trading ignores transaction costs, slippage, and dividends
- Sector benchmarks are coarse placeholders
- Local JSON storage instead of a database
- Single-ticker analysis (no portfolio-level correlation yet)

---

## 🗺️ Roadmap

- [x] v2.0: LLM-first agents + dialectical debate (done)
- [x] Prediction evaluation with closed-loop learning (done)
- [x] Flask webapp with chat + evaluation UI (done)
- [ ] Portfolio-level multi-ticker correlation analysis
- [ ] Persistent database storage (SQLite/PostgreSQL)
- [ ] Real-time market data streaming
- [ ] Strategy backtesting with historical data
- [ ] Docker deployment
