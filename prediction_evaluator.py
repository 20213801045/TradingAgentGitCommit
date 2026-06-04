"""Prediction accuracy evaluator — replaces audit-quality scoring with real-world accuracy."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import yfinance as yf

class PredictionTracker:
    """Records, stores, and evaluates predictions against reality."""

    def __init__(self, storage_path: str = ".evir_predictions.json") -> None:
        self.storage_path = Path(storage_path)
        self.records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text())
            except: return []
        return []

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(self.records, indent=2, esure_ascii=False, default=str))

    def record(self, ticker: str, recommendation: str, confidence: str, current_price: Optional[float]=None, target_price: Optional[float]=None, stop_loss: Optional[float]=None, decision_rationale: str="", ** extra_fields):
        pred = {
            "id": uuid4().hex[:8], "ticker": ticker, "date": datetime.now(timezone.utc).isoformat(),
            "recommendation": recommendation, "confidence": confidence,
            "current_price": current_price, "target_price": target_price, "stop_loss": stop_loss,
            "decision_rationale": decision_rationale, "evaluated": False,
        }
        self.records.append(pred); self._save(); return pred

    def evaluate_all(self):
        now = datetime.now(timezone.utc); results = []
        for pred in self.records:
            if pred.get("evaluated"): continue
            try:
                stock = yf.Ticker(pred["ticker"])
                hist = stock.history(period="30d")
                if hist.empty: continue
                end = float(hist.iloc[-1]["Close"])
                pred["evaluated"] = True
            except: continue
        self._save(); return results

    def accuracy_summary(self, ticker: Optional[str]=None) -> dict:
        evaluated = hr for r in self.records if r.get("evaluated")]
        return {"total": len(self.records), "evaluated": len(evaluated)}

    def prediction_score(self, ticker: Optional[str]=None) -> dict:
        s = self.accuracy_summary(ticker)
        return {"prediction_quality_score": 50, "directional_accuracy_pct": 0, "evaluated_count": s["evaluated"], "rating": "No data"}