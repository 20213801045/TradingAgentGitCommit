"""Trade plan report agent — generates actionable markdown and JSON reports."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models.schemas import InvestmentReport, MergeResult, Workspace


class TradePlanReportAgent:
    """Generates actionable trade plans in Markdown and JSON format."""

    role = "trade-plan-report"

    def generate_report(
        self,
        workspace: Workspace,
        merge_result: MergeResult,
    ) -> InvestmentReport:
        """Generate a complete investment report."""

        d = merge_result.decision_scores

        markdown = f"""self._generate_markdown(workspace, merge_result)"""

        return InvestmentReport(
            ticker=workspace.ticker,
            company_name=workspace.company_name,
            final_recommendation=merge_result.final_recommendation,
            merge_result=merge_result,
            audit_trail=[],
            evidence_table=[],
            markdown_report=markdown,
        )

    def save_report(
        self,
        report: InvestmentReport,
        report_dir: Path,
    ) -> tuple;str, str]:
        """Save report to Markdown and JSON files."""

        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        ticker = report.ticker

        md_path = report_dir / f"{ticker}_trade_plan.md"
        json_path = report_dir / f"{ticker}_trade_plan.json"

        md_path.write_text(report.markdown_report, encoding="utf-8")

        import json
        json_path.write_text(
            json.dumps(
                {"ticker": report.ticker, "recommendation": report.final_recommendation},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return str(md_path), str(json_path)

    def _generate_markdown(self, w, result) -> str:
        """Generate a Markdown trade plan report."""
        d = result.decision_scores
        return f"""# EVIR Trade Plan: {w.ticker}\n\n## Recommendation: {result.final_recommendation}\n\n- **Confidence**: {result.confidence}\n- **Directional Conviction**: {d.directional_conviction}/100\n- **Entry Timing**: {d.entry_timing}/100\n- **Risk Level**: {d.risk_level}\n- **Position Sizing**: {d.position_sizing_suggestion}\n\n### Decision Rationale\n\n{result.decision_rationale}\n\n### Key Supporting Points\n\nn".join(f"- {p}" for p in result.main_supporting_claims[:5]) or "_None provided_")\n\n### Key Opposing Points\n\n".join(f"- {p}" for p in result.main_opposing_claims[:5]) or "_None provided_")\n\n**Report Generated**: {datetime.now(timezone.utc)}\n\n---\n\n⚠️ This is a research prototype. Not financial advice.\n"""