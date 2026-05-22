"""Prototype research agents for EVIR."""

from agents.bear_agent import BearAgent
from agents.backtest_agent import BacktestAgent
from agents.bull_agent import BullAgent
from agents.counter_evidence_agent import CounterEvidenceAgent
from agents.financial_statement_agent import FinancialStatementAgent
from agents.fundamental_agent import FundamentalAgent
from agents.industry_agent import IndustryComparisonAgent
from agents.llm_analysis_agent import LLMAnalysisAgent
from agents.macro_agent import MacroAgent
from agents.merge_agent import MergeAgent
from agents.portfolio_agent import PortfolioAgent
from agents.research_coordinator_agent import ResearchCoordinatorAgent
from agents.report_agent import ReportAgent
from agents.risk_agent import RiskAgent
from agents.technical_agent import TechnicalAgent
from agents.valuation_agent import ValuationAgent

__all__ = [
    "BacktestAgent",
    "BearAgent",
    "BullAgent",
    "CounterEvidenceAgent",
    "FinancialStatementAgent",
    "FundamentalAgent",
    "IndustryComparisonAgent",
    "LLMAnalysisAgent",
    "MacroAgent",
    "MergeAgent",
    "PortfolioAgent",
    "ResearchCoordinatorAgent",
    "ReportAgent",
    "RiskAgent",
    "TechnicalAgent",
    "ValuationAgent",
]
