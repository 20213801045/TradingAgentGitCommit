"""Prototype research agents for EVIR."""

from .backtest_agent import BacktestAgent
from .bear_agent import BearAgent
from .bull_agent import BullAgent
from .counter_evidence_agent import CounterEvidenceAgent
from .debate_agent import DebateAgent
from .deep_research_agent import DeepResearchAgent
from .financial_statement_agent import FinancialStatementAgent
from .fundamental_agent import FundamentalAgent
from .industry_agent import IndustryComparisonAgent
from .llm_analysis_agent import LLMAnalysisAgent
from .macro_agent import MacroAgent
from .macro_sentiment_agent import MacroSentimentAgent
from .merge_agent import MergeAgent
from .portfolio_agent import PortfolioAgent
from .report_agent import ReportAgent
from .research_coordinator_agent import ResearchCoordinatorAgent
from .risk_agent import RiskAgent
from .technical_agent import TechnicalAgent
from .technical_timing_agent import TechnicalTimingAgent
from .trade_plan_report_agent import TradePlanReportAgent
from .valuation_agent import ValuationAgent

__all__ = [
    "BacktestAgent",
    "BearAgent",
    "BullAgent",
    "CounterEvidenceAgent",
    "DebateAgent",
    "DeepResearchAgent",
    "FinancialStatementAgent",
    "FundamentalAgent",
    "IndustryComparisonAgent",
    "LLMAnalysisAgent",
    "MacroAgent",
    "MacroSentimentAgent",
    "MergeAgent",
    "PortfolioAgent",
    "ReportAgent",
    "ResearchCoordinatorAgent",
    "RiskAgent",
    "TechnicalAgent",
    "TechnicalTimingAgent",
    "TradePlanReportAgent",
    "ValuationAgent",
]
