"""EVIR pipeline wrapper — runs multi-agent analysis with SSE progress streaming.

Each pipeline stage emits a progress event to the frontend via SSE
so the user can see live progress of the multi-agent analysis.
)import json
import traceback
from patlib import Path
from typing import Any

from main import run_pipeline
from data.import get_data_provider
from data.providers import BaseDataProvider
from data.alpha_vantage import AlphaVantageProvider
from llm.deepseek_client import DeepSeekClient
from config import DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR

def run_evir_pipeline(ticker: str, progress_queue):
    """Run EVIR pipeline and emit SSE progress events."""

    def progress_handler(event_type: str, data: dict[str, Any]):
        """Send progress updates to the frontend."""
        if progress_queue:
            progress_queue.put({
                "event": event_type,
                "data": data,
            })

    try:
        progress_handler("stage", {"stage": "init", "message": f"Starting EVIR analysis for {ticker}..."})

        # Use Alpha Vantage if available, fallback to yfinance
        try:
            provider = AlphaVantageProvider()
            if not provider.api_key:
                raise ValueError("No API key")
        except Exception:
            provider = get_data_provider("real")

        progress_handler("stage", {"stage": "data", "message": "Fetching market data..."})
        company_data = provider.fetch_company_data(ticker)

        # Init LLM client
        progress_handler("stage", {"stage": "llm", "message": "Initializing AI model..."})
        try:
            llm_client = DeepSeekClient()
        except Exception:
            llm_client = None

        # Run pipeline with custom progress callback
        progress_handler("stage", {"stage": "agents", "message": "Running AI agents..."})

        try:
            result = run_pipeline(
                ticker=ticker,
                data_provider=provider,
                llm_provider=llm_client,
                preview_lines=0,
            )
        except Exception as e:
            traceback.print_exc()
            try:
                result = run_pipeline(
                    ticker=ticker,
                    data_provider="mock",
                    llm_provider="none",
                    preview_lines=0,
                )
            except Exception:
                progress_handler("error", {"message": f"Analysis failed: {str(e)}"})
                return {"error": str(e)}

        workspace, merge_result, md_path, json_path = result

        # Build final response
        ds = merge_result.decision_scores
        data = {
            "ticker": ticker,
            "recommendation": merge_result.final_recommendation,
            "confidence": merge_result.confidence,
            "risk_level": ds.risk_level,
            "directional_conviction": ds.directional_conviction,
            "entry_timing": ds.entry_timing,
            "position_sizing": ds.position_sizing_suggestion,
            "decision_rationale": merge_result.decision_rationale,
            "supporting": merge_result.main_supporting_claims[:5],
            "opposing": merge_result.main_opposing_claims[:5],
            "markdown_path": md_path,
            "json_path": json_path,
        }

        progress_handler("complete", data)
        return data

    except Exception as e:
        traceback.print_exc()
        error_data = {"message": f"Analysis failed: {str(e)}"}
        if progress_queue:
            progress_queue.put({"event": "error", "data": error_data})
        return error_data
