"""Prediction accuracy evaluator — replaces audit-quality scoring with real-world accuracy.

Tracks every prediction, evaluates them against actual market data,
and provides a feedback loop for continuous improvement.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import yfinance as yf


class PredictionTracker:
    """Records, stores, and evaluates predictions against reality.

    Replaces the old Evaluator which only scored "report completeness".
    Now we measure what investors actually care about: was the call right?
    """

    def __init__(self, storage_path: str = ".evir_predictions.json") -> None:
        self.storage_path = Path(storage_path)
        self.records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False, default=str),
        )

    # ── record a prediction ──────────────────────────────────────────
    def record(
        self,
        ticker: str,
        recommendation: str,
        confidence: str,
        current_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        decision_rationale: str = "",
        time_horizon_days: int = 180,
    ) -> dict:
        """Save a prediction after each pipeline run."""
        pred = {
            "id": str(uuid4())[:8],
            "ticker": ticker,
            "date": datetime.now(timezone.utc).isoformat(),
            "recommendation": recommendation,
            "confidence": confidence,
            "current_price": current_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "time_horizon_days": time_horizon_days,
            "decision_rationale": decision_rationale[:500],
            "evaluated": False,
            "evaluation_date": None,
            "actual_price": None,
            "actual_return_pct": None,
            "directional_correct": None,
            "target_hit": None,
            "max_drawdown_pct": None,
        }
        self.records.append(pred)
        self._save()
        return pred

    # ── evaluate against reality ─────────────────────────────────────
    def evaluate_all(self) -> list[dict]:
        """Check all unevaluated predictions against current market prices."""
        now = datetime.now(timezone.utc)
        evaluated: list[dict] = []

        for pred in self.records:
            if pred.get("evaluated"):
                continue

            pred_date = datetime.fromisoformat(pred["date"].replace("Z", "+00:00"))
            days_passed = (now - pred_date).days
            min_wait = min(30, int(pred.get("time_horizon_days", 180) * 0.3))

            if days_passed < min_wait:
                continue

            try:
                ticker = pred["ticker"]
                stock = yf.Ticker(ticker)
                hist = stock.history(period=f"{max(days_passed, 5)}d")

                if hist.empty or len(hist) < 2:
                    continue

                start_price = float(hist.iloc[0]["Close"])
                end_price = float(hist.iloc[-1]["Close"])
                actual_return = ((end_price - start_price) / start_price) * 100

                # directional accuracy
                rec = pred["recommendation"]
                if rec == "Buy":
                    directional = actual_return > 2
                elif rec == "Sell":
                    directional = actual_return < -2
                else:
                    directional = -5 <= actual_return <= 5

                # target hit
                target = pred.get("target_price")
                target_hit: Optional[bool] = None
                if target and target > 0:
                    high = float(hist["High"].max())
                    target_hit = high >= target

                # max drawdown
                peak = hist["Close"].cummax()
                drawdown = float(((hist["Close"] - peak) / peak * 100).min())

                pred["evaluated"] = True
                pred["evaluation_date"] = now.isoformat()
                pred["actual_price"] = round(end_price, 2)
                pred["actual_return_pct"] = round(actual_return, 2)
                pred["directional_correct"] = directional
                pred["target_hit"] = target_hit
                pred["max_drawdown_pct"] = round(drawdown, 2)

                evaluated.append(pred)

            except Exception:
                continue

        if evaluated:
            self._save()

        return evaluated

    # ── accuracy metrics ─────────────────────────────────────────────
    def accuracy_summary(self, ticker: Optional[str] = None) -> dict:
        """Generate prediction accuracy metrics across all history."""
        records = self.records
        if ticker:
            records = [r for r in records if r["ticker"] == ticker]

        evaluated = [r for r in records if r.get("evaluated")]
        total = len(evaluated)

        if total == 0:
            return {
                "total_predictions": len(records),
                "evaluated": 0,
                "summary": "No evaluated predictions yet.",
            }

        correct = sum(1 for r in evaluated if r.get("directional_correct"))
        accuracy = correct / total * 100 if total > 0 else 0

        # calibration by confidence
        by_confidence: dict[str, dict[str, int]] = {}
        for r in evaluated:
            conf = r.get("confidence", "medium")
            by_confidence.setdefault(conf, {"total": 0, "correct": 0})
            by_confidence[conf]["total"] += 1
            if r.get("directional_correct"):
                by_confidence[conf]["correct"] += 1

        # by recommendation type
        by_rec: dict[str, dict[str, int]] = {}
        for r in evaluated:
            rec = r.get("recommendation", "Hold")
            by_rec.setdefault(rec, {"total": 0, "correct": 0})
            by_rec[rec]["total"] += 1
            if r.get("directional_correct"):
                by_rec[rec]["correct"] += 1

        avg_return = sum(r.get("actual_return_pct", 0) or 0 for r in evaluated) / total

        return {
            "total_predictions": len(records),
            "evaluated": total,
            "directional_accuracy_pct": round(accuracy, 1),
            "avg_actual_return_pct": round(avg_return, 2),
            "calibration": {
                conf: {
                    "accuracy_pct": round(stats["correct"] / stats["total"] * 100, 1)
                    if stats["total"] else 0,
                    "count": stats["total"],
                }
                for conf, stats in by_confidence.items()
            },
            "by_recommendation": by_rec,
            "recent": [
                {
                    "ticker": r["ticker"],
                    "date": r["date"][:10],
                    "recommendation": r["recommendation"],
                    "confidence": r["confidence"],
                    "actual_return_pct": r.get("actual_return_pct"),
                    "correct": r.get("directional_correct"),
                }
                for r in evaluated[-5:]
            ],
        }

    # ── closed-loop: feedback for the LLM ────────────────────────────
    def feedback_string(self, ticker: Optional[str] = None) -> str:
        """Generate a compact feedback string for injection into LLM debate prompts.

        This is the core of the closed-loop: the LLM sees its own track record
        and learns from past mistakes.  If it was wrong on MU last time,
        it will see that before making a new call.
        """
        summary = self.accuracy_summary(ticker)

        if summary["evaluated"] == 0:
            return ""

        lines = [
            "## 📈 YOUR PREDICTION TRACK RECORD (for calibration)",
            f"- Total predictions evaluated: {summary['evaluated']}",
            f"- Directional accuracy: {summary['directional_accuracy_pct']}%",
        ]

        if summary.get("calibration"):
            lines.append("- Calibration by confidence level:")
            for conf, stats in summary["calibration"].items():
                lines.append(
                    f"  • {conf} confidence: {stats['accuracy_pct']}% accurate "
                    f"({stats['count']} predictions)",
                )

        if summary.get("recent"):
            lines.append("- Recent prediction outcomes:")
            for r in summary["recent"]:
                icon = "✓" if r.get("correct") else "✗"
                ret = r.get("actual_return_pct", "?")
                if ret is not None:
                    ret = f"{ret:+.1f}%"
                lines.append(
                    f"  {icon} {r['ticker']} [{r['date']}]: {r['recommendation']} "
                    f"({r['confidence']}) → actual: {ret}",
                )

        lines.append(
            "\nUse this track record to CALIBRATE your confidence. "
            "If your high-confidence calls are often wrong, temper your certainty. "
            "If a specific ticker has surprised you before, acknowledge that.",
        )

        return "\n".join(lines)

    # ── prediction quality score ─────────────────────────────────────
    def prediction_score(self, ticker: Optional[str] = None) -> dict:
        """Generate the new evaluation score that replaces old audit metrics.

        Old metrics: evidence_coverage, temporal_validity, audit_completeness...
        New metric: was the prediction actually right?
        """
        summary = self.accuracy_summary(ticker)

        evaluated = summary.get("evaluated", 0)
        if evaluated == 0:
            return {
                "prediction_quality_score": 0.0,
                "rating": "Unknown — no evaluated predictions yet",
                "directional_accuracy_pct": 0.0,
                "evaluated_count": 0,
            }

        accuracy = summary.get("directional_accuracy_pct", 0)

        # calibration bonus: high-confidence predictions should be more accurate
        calibration_bonus = 0.0
        if summary.get("calibration"):
            conf_map = summary["calibration"]
            if "high" in conf_map:
                calibration_bonus = min(30, conf_map["high"].get("accuracy_pct", 0) * 0.3)

        score = min(100.0, accuracy * 0.7 + calibration_bonus)

        if score >= 70:
            rating = "⭐ Excellent — highly reliable"
        elif score >= 50:
            rating = "👍 Good — above random"
        elif score >= 30:
            rating = "⚠️ Fair — near coin-flip"
        else:
            rating = "❌ Poor — unreliable"

        return {
            "prediction_quality_score": round(score, 1),
            "rating": rating,
            "directional_accuracy_pct": accuracy,
            "evaluated_count": evaluated,
        }
