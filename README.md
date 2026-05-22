# EVIR: Evidence-Versioned Investment Research

EVIR is a Python prototype for building an auditable, multi-agent investment
research system.

The project explores a simple idea: investment research should not be a single
black-box answer. It should be a structured process where different research
roles generate claims, attach evidence, challenge each other, merge conflicting
views, and leave a clear audit trail behind every recommendation.

EVIR currently combines deterministic research agents, optional LLM-assisted
analysis, local JSON workspaces, evidence scoring, revision checks, report
generation, and a paper-trading simulation. The next development goal is to
turn the rule-based agents into LLM-backed specialist agents, coordinated by a
central research coordinator.

This is a research and engineering prototype. It is not financial advice and it
does not place real trades.

## What This Project Is Trying To Build

EVIR is intended to become a multi-agent stock research workflow with these
core properties:

- **Role separation**: each agent owns a specific research perspective, such as
  fundamentals, valuation, technicals, risk, or counter-evidence.
- **Evidence-first reasoning**: every investment claim should be linked to
  explicit evidence, metrics, timestamps, and source types.
- **Versioned research memory**: research is stored in ticker-specific
  workspaces with branch-like perspectives, similar to how software teams use
  Git branches.
- **Conflict-aware synthesis**: bullish claims, bearish claims, risks, stale
  evidence, and counter-evidence should all be considered before producing a
  recommendation.
- **Auditable reports**: the final report should show why a recommendation was
  made, which claims supported it, which claims challenged it, and what would
  trigger a future revision.
- **Paper-trading evaluation**: recommendations can be converted into simulated
  portfolio actions so the research system can be monitored over time.

## Current Architecture

At the moment, EVIR is mostly deterministic and rule-based. The project already
has a multi-agent structure, but most agents generate their conclusions from
hard-coded thresholds and templates.

The main research pipeline lives in `main.py`:

```text
data provider
  -> workspace creation
  -> specialist research agents
  -> evidence scoring and temporal checks
  -> counter-evidence generation
  -> merge and recommendation
  -> report generation
  -> evaluation
  -> revision check
```

The paper-trading workflow lives in `paper_trade.py` and
`paper_trading/engine.py`:

```text
ticker universe
  -> run EVIR research for each ticker
  -> convert recommendation into simulated action
  -> update local paper portfolio
  -> write trades, equity curve, and summary
```

## Agent System

Current specialist agents include:

- `ResearchCoordinatorAgent`: records the research plan, reviews branch
  coverage, and captures follow-up priorities for the workflow.
- `FundamentalAgent`: revenue growth, profitability, valuation pressure, and
  balance-sheet health.
- `FinancialStatementAgent`: statement quality, cash generation, leverage, and
  return metrics.
- `ValuationAgent`: relative valuation, forward P/E, growth-adjusted valuation,
  and free-cash-flow yield.
- `IndustryComparisonAgent`: company metrics compared with coarse sector
  benchmarks.
- `MacroAgent`: interest-rate environment, demand, inflation, and currency
  context.
- `TechnicalAgent`: moving-average trend, RSI, volatility, support, and
  resistance.
- `BacktestAgent`: a simple MA20/MA50 trend-following backtest summary.
- `PortfolioAgent`: position sizing, liquidity, correlation, and portfolio fit.
- `BullAgent`: constructs a bullish case from prior evidence.
- `BearAgent`: constructs a cautious or bearish case from prior evidence.
- `RiskAgent`: audits valuation risk, volatility risk, evidence gaps, and stale
  evidence.
- `CounterEvidenceAgent`: generates questions that challenge important positive
  claims. It can optionally use an LLM for one additional question.
- `LLMAnalysisAgent`: optionally uses an LLM to synthesize evidence-linked
  Chinese investment insights.
- `MergeAgent`: merges all branches into a final recommendation using
  deterministic scoring and conflict rules.
- `ReportAgent`: creates the final Markdown and JSON investment report.

Today, `ResearchCoordinatorAgent`, `FundamentalAgent`, `ValuationAgent`,
`TechnicalAgent`, `RiskAgent`, `LLMAnalysisAgent`, and part of
`CounterEvidenceAgent` can use an LLM. Most other specialist research agents
are still rule-based.

## Target Architecture

The next major direction is to move from a mostly rule-based prototype to an
LLM-backed multi-agent research system.

The intended design is:

```text
ResearchCoordinatorAgent
  -> FundamentalAgent
  -> FinancialStatementAgent
  -> ValuationAgent
  -> IndustryComparisonAgent
  -> MacroAgent
  -> TechnicalAgent
  -> BacktestAgent
  -> PortfolioAgent
  -> BullAgent
  -> BearAgent
  -> RiskAgent
  -> CounterEvidenceAgent
  -> MergeAgent / SynthesisAgent
  -> ReportAgent
```

The first version of `ResearchCoordinatorAgent` already records planning and
coverage-review commits. Its planned role is to grow into the main
orchestration agent. It should not simply replace the specialist agents.
Instead, it should eventually:

- create or revise the research plan,
- decide which specialist agents need to run,
- inspect whether agent outputs are evidence-grounded,
- identify missing analysis,
- request follow-up work from specific agents,
- coordinate disagreement between bullish, bearish, and risk views,
- decide whether the final recommendation has enough support,
- maintain an auditable research state.

The current `LLMAnalysisAgent` is not the coordinator. It is better understood
as a synthesis agent: it summarizes existing evidence but does not manage the
research process.

## LLM Migration Plan

The planned migration strategy is **LLM-first, rules-as-fallback**.

Each specialist agent should eventually:

- receive structured input data and workspace context,
- call an LLM with a role-specific prompt,
- return validated JSON matching the existing Pydantic schemas,
- attach every claim to explicit evidence,
- fail safely back to the current deterministic logic when the LLM call fails,
- preserve the same audit trail and report structure.

The first agent migrated to this pattern is:

```text
FundamentalAgent
```

`FundamentalAgent` and `ValuationAgent` now have LLM-backed versions that keep
the original deterministic rules as fallbacks. `FundamentalAgent` covers growth
quality, margin quality, cash generation, balance sheet, capital allocation,
and fundamental valuation risk, with a lightweight validator that rejects LLM
claims that obviously contradict supplied metrics. The LLM writes structured
insights, while the code still binds each claim to supplied metrics such as
revenue growth, net margin, free-cash-flow margin, debt-to-equity, forward P/E,
sector P/E, free-cash-flow yield, price trend, RSI, volatility, support,
resistance, and prior risk-review source commits.

## Data Providers

EVIR supports two data modes:

- `mock`: deterministic local data, mainly for tests and demos.
- `real`: yfinance-backed market and financial data.

Real data mode fetches:

- price history,
- company name,
- financial summary,
- valuation fields,
- technical indicators,
- backtest inputs.

Price and financial data are cached under:

```text
outputs/cache/prices/
outputs/cache/financials/
```

If a real-data refresh fails but stale cache exists, EVIR can fall back to the
cached data so the workflow can continue.

## Paper Trading

EVIR includes a local paper-trading simulation. It does not place real orders.

For each ticker, the paper-trading engine:

1. runs the EVIR research pipeline,
2. reads the final recommendation,
3. fetches the latest close price,
4. converts the recommendation into a simulated trade,
5. updates a local portfolio,
6. writes a trade log and equity curve.

Default behavior:

- `Buy`: buy up to a target position size.
- `Hold`: keep existing exposure without forcing a new trade.
- `Sell` or `Avoid`: exit existing exposure.
- confidence affects sizing.
- small trades below the minimum trade value are skipped.

Paper-trading outputs are saved under:

```text
outputs/paper_trading/portfolio.json
outputs/paper_trading/trades.csv
outputs/paper_trading/equity_curve.csv
outputs/paper_trading/summary.md
```

## Project Layout

```text
evir/
  main.py                       # main research pipeline
  paper_trade.py                # paper-trading CLI
  config.py                     # thresholds and default settings
  models/                       # Pydantic schemas
  agents/                       # research agents
  data/                         # mock and real data providers
  evidence/                     # evidence scoring and temporal checks
  revision/                     # decision revision engine
  evaluation/                   # audit and quality metrics
  memory/                       # workspace and JSON storage
  paper_trading/                # paper-trading engine
  tests/                        # test suite
  outputs/                      # generated workspaces, reports, caches
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the default research pipeline:

```bash
python main.py
```

Run with mock data:

```bash
python main.py --ticker AAPL --data-provider mock --llm-provider none
```

Run with real data:

```bash
python main.py --ticker AAPL --data-provider real
```

Run the paper-trading simulation:

```bash
python paper_trade.py
```

Run a smaller ticker universe:

```bash
python paper_trade.py --tickers MU,INTC,NVDA,AMD
```

Disable LLM calls:

```bash
python main.py --llm-provider none
python paper_trade.py --llm-provider none
```

## LLM Configuration

DeepSeek is currently the default LLM provider.

Create a local `.env` file:

```bash
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_MODEL=deepseek-v4-pro
```

Then run:

```bash
python main.py --ticker AAPL --llm-provider deepseek
```

If no valid LLM client is available, EVIR falls back to deterministic no-LLM
mode for the parts of the system that support fallback.

## Generated Outputs

Research runs generate:

```text
outputs/workspaces/<TICKER>_workspace.json
outputs/reports/<TICKER>_report.md
outputs/reports/<TICKER>_investment_report.json
outputs/reports/<TICKER>_revision_report.md
outputs/evaluation/<TICKER>_evaluation.json
```

The workspace JSON is the most important artifact for debugging. It contains
all branches, commits, claims, evidence, risk tags, confidence labels, evidence
scores, and temporal status values.

## Tests

Run the test suite:

```bash
python -m pytest
```

## Current Limitations

- Most agents are still rule-based and template-driven.
- LLM usage is limited to synthesis and partial counter-evidence generation.
- Merge logic is deterministic and keyword-based.
- Real data quality depends on yfinance and Yahoo Finance availability.
- Sector benchmarks are coarse placeholders.
- Paper trading ignores transaction costs, slippage, taxes, dividends, and
  execution constraints.
- Local JSON storage is used instead of a database.

## Near-Term Roadmap

- Add a shared LLM-backed agent base layer.
- Keep improving `FundamentalAgent` with richer data sources and stricter claim
  validation.
- Add an entry-point scoring layer for current-buy-point assessment.
- Expand risk review into entry-score risk penalties.
- Expand `ResearchCoordinatorAgent` from audit logging into active orchestration.
- Improve merge logic with LLM-assisted conflict review.
- Improve report generation so it reflects both deterministic evidence and LLM
  reasoning more naturally.
- Add stronger portfolio and benchmark evaluation for paper trading.
