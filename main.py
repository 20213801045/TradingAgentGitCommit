"""Run a local EVIR prototype workflow."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Sequence

from agents import (
    CounterEvidenceAgent,
    DeepResearchAgent,
    MacroSentimentAgent,
    ResearchCoordinatorAgent,
    RiskAgent,
    TechnicalTimingAgent,
)
from agents.debate_agent import DebateAgent
from agents.trade_plan_report_agent import TradePlanReportAgent
from config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_TICKER,
    ENABLE_LLM_CACHE,
    LLM_PROVIDER,
    LLM_CACHE_TTL_HOURS,
    MODEL_NAME,
    USE_LLM,
    USE_REAL_DATA,
)
from data import (
    BaseDataProvider,
    get_data_provider,
    get_mock_new_evidence,
    get_real_new_evidence,
)
from evidence import process_workspace_evidence
from evaluation import Evaluator, save_evaluation_result
from prediction_evaluator import PredictionTracker
from llm import BaseLLMClient, CachedLLMClient, LLMError, get_llm_client
from models.schemas import (
    ClaimEvidenceCommit,
    EvaluationResult,
    MergeResult,
    RevisionResult,
    Workspace,
)
from memory.workspace import add_commit, create_branch, create_workspace, save_workspace
from revision import RevisionEngine, save_revision_report


def main(argv: Sequence[str] | None = None) -> None:
    """Run the deterministic EVIR agent pipeline from CLI arguments."""

    args = _parse_args(argv)
    run_pipeline(
        ticker=args.ticker,
        data_provider=args.data_provider,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        use_llm_cache=not args.no_llm_cache,
        llm_cache_ttl_hours=args.llm_cache_ttl_hours,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        preview_lines=args.preview_lines,
    )


def run_pipeline(
    ticker: str = DEFAULT_TICKER,
    data_provider: str | BaseDataProvider | None = None,
    llm_provider: str | BaseLLMClient | None = None,
    llm_model: str | None = MODEL_NAME,
    use_llm_cache: bool = ENABLE_LLM_CACHE,
    llm_cache_ttl_hours: int = LLM_CACHE_TTL_HOURS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    preview_lines: int = 12,
    new_evidence_provider: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[Workspace, MergeResult, str, str]:
    """Run the deterministic EVIR pipeline and return generated artifacts."""

    pipeline_start = perf_counter()
    _log_stage("Pipeline", f"start ticker={ticker} data_provider={data_provider or 'config'}")
    selected_data_provider = data_provider
    if selected_data_provider is None:
        selected_data_provider = "real" if USE_REAL_DATA else "mock"
    provider = (
        get_data_provider(selected_data_provider)
        if isinstance(selected_data_provider, str)
        else selected_data_provider
    )
    company_data = _run_stage(
        f"Fetch company data ({provider.name})",
        lambda: provider.fetch_company_data(ticker),
    )
    llm_client = _run_stage(
        f"Resolve LLM client ({llm_provider or 'config'})",
        lambda: _resolve_llm_client(
            llm_provider,
            llm_model,
            use_llm_cache=use_llm_cache,
            llm_cache_ttl_hours=llm_cache_ttl_hours,
        ),
    )
    workspace = create_workspace(
        ticker=company_data["ticker"],
        company_name=company_data["company_name"],
        research_question=(
            f"{company_data['ticker']} 是否适合纳入长期投资观察名单？"
        ),
    )

    branch_descriptions = {
        "research-coordination": "主协调 agent 的研究计划、覆盖审查和后续优先级。",
        "deep-research": "LLM驱动的深度基本面、财报、估值和行业分析（一站式）。",
        "macro-analysis": "LLM驱动的宏观环境、利率和汇率评估。",
        "technical-analysis": "LLM驱动的技术面趋势、动量、波动率和择时信号。",
        "risk-review": "关键风险、脆弱点和修正条件。",
        "counter-evidence": "针对重要正面观点的反证检查。",
    }

    _run_stage(
        "Create workspace branches",
        lambda: _create_branches(workspace, branch_descriptions),
    )

    # Stage 0: coordinator records the research plan before specialist work.
    _run_agent_stage(
        "ResearchCoordinatorAgent initial_plan",
        ResearchCoordinatorAgent(llm_client=llm_client),
        {"phase": "initial_plan", "company_data": company_data},
        workspace,
    )

    # Stage 1: LLM-driven analysis (3 agents replace the old 8 + bull/bear).
    _run_agent_group_stage(
        "Analysis agents",
        [
            ("DeepResearchAgent", DeepResearchAgent(llm_client=llm_client), company_data),
            ("MacroSentimentAgent", MacroSentimentAgent(llm_client=llm_client), company_data),
            ("TechnicalTimingAgent", TechnicalTimingAgent(llm_client=llm_client), company_data),
        ],
        workspace,
    )

    # Stage 2: risk review audits all prior claim-evidence commits.
    _run_agent_stage("RiskAgent", RiskAgent(llm_client=llm_client), {}, workspace)

    # Stage 4: score evidence before selecting positive claims for challenge.
    _run_stage("Process evidence before counter-evidence", lambda: process_workspace_evidence(workspace))

    counter_evidence_agent = CounterEvidenceAgent(llm_client=llm_client)
    _run_stage(
        "CounterEvidenceAgent",
        lambda: _add_commits(workspace, counter_evidence_agent.analyze(workspace)),
        result_formatter=lambda count: f"{count} commits",
    )

    # Stage 5: process the newly generated counter-evidence commits.
    _run_stage("Process evidence after counter-evidence", lambda: process_workspace_evidence(workspace))
    _run_agent_stage(
        "ResearchCoordinatorAgent pre_merge_review",
        ResearchCoordinatorAgent(llm_client=llm_client),
        {"phase": "pre_merge_review", "company_data": company_data},
        workspace,
    )
    workspace_path = _run_stage(
        "Save workspace",
        lambda: save_workspace(workspace, output_dir),
    )

    print(f"Saved workspace to: {workspace_path}")
    print()
    _print_commits_by_branch(workspace)

    # ── Prediction tracker (closed-loop learning) ───────────────
    tracker = PredictionTracker(
        storage_path=Path(output_dir) / ".evir_predictions.json"
    )
    # Evaluate old predictions before making new ones
    tracker.evaluate_all()

    # ── NEW: LLM-powered debate agent (replaces keyword-matching merge) ──
    debate_agent = DebateAgent(llm_client=llm_client, tracker=tracker)
    merge_result = _run_stage(
        "DebateAgent (LLM debate)",
        lambda: debate_agent.debate(workspace),
        result_formatter=lambda result: f"recommendation={result.final_recommendation} confidence={result.confidence}",
    )
    _print_merge_result(merge_result, debate_agent)

    # ── NEW: Trade-plan report (actionable format) ──
    trade_plan_agent = TradePlanReportAgent()
    report = _run_stage(
        "TradePlanReportAgent generate",
        lambda: trade_plan_agent.generate_report(workspace, merge_result),
    )
    markdown_report_path, json_report_path = _run_stage(
        "TradePlanReportAgent save",
        lambda: trade_plan_agent.save_report(report, report_dir),
        result_formatter=lambda paths: f"trade_plan_md={paths[0]} json={paths[1]}",
    )
    print()
    print(f"Saved Trade Plan to: {markdown_report_path}")
    if preview_lines > 0:
        print()
        _print_report_preview(report.markdown_report, preview_lines)

    # ── Record prediction & evaluate accuracy (replaces old audit scoring) ──
    market_data = company_data.get("market_data", {})
    current_price = market_data.get("current_price") if isinstance(market_data, dict) else company_data.get("current_price")
    _run_stage(
        "PredictionTracker record",
        lambda: tracker.record(
            ticker=workspace.ticker,
            recommendation=merge_result.final_recommendation,
            confidence=merge_result.confidence,
            current_price=float(current_price) if current_price else None,
            decision_rationale=merge_result.decision_rationale,
        ),
        result_formatter=lambda _: "saved",
    )
    accuracy = tracker.accuracy_summary()
    score = tracker.prediction_score()
    print()
    print(f"{'='*60}")
    print(f"📊 PREDICTION QUALITY: {score['prediction_quality_score']}/100 — {score['rating']}")
    print(f"   Directional accuracy: {score['directional_accuracy_pct']}% ({score['evaluated_count']} evaluated)")
    print(f"{'='*60}")

    revision_evidence_provider = new_evidence_provider or _revision_evidence_provider(
        provider,
    )
    new_evidence_data = _run_stage(
        "Fetch revision evidence",
        lambda: revision_evidence_provider(workspace.ticker),
    )
    revision_engine = RevisionEngine()
    revision_result = _run_stage(
        "RevisionEngine",
        lambda: revision_engine.revise(workspace, merge_result, new_evidence_data),
        result_formatter=lambda result: f"revised_recommendation={result.revised_recommendation}",
    )
    revision_report_path = _run_stage(
        "Save revision report",
        lambda: save_revision_report(revision_result, workspace.ticker, report_dir),
    )
    print()
    _print_revision_result(revision_result)
    print(f"Saved revision report to: {revision_report_path}")
    _log_stage("Pipeline", f"done in {perf_counter() - pipeline_start:.2f}s")

    return workspace, merge_result, markdown_report_path, json_report_path


def _revision_evidence_provider(
    provider: BaseDataProvider,
) -> Callable[[str], dict[str, Any]]:
    """Choose revision evidence source to match the primary data provider."""

    if provider.name == "real":
        return get_real_new_evidence
    return get_mock_new_evidence


def _create_branches(
    workspace: Workspace,
    branch_descriptions: dict[str, str],
) -> int:
    """Create all configured workspace branches and return the count."""

    for branch_name, description in branch_descriptions.items():
        create_branch(workspace, branch_name, description)
    return len(branch_descriptions)


def _run_stage(
    label: str,
    action: Callable[[], Any],
    result_formatter: Callable[[Any], str] | None = None,
) -> Any:
    """Run one pipeline stage with start/done timing logs."""

    _log_stage(label, "start")
    started_at = perf_counter()
    try:
        result = action()
    except Exception:
        _log_stage(label, f"failed after {perf_counter() - started_at:.2f}s")
        raise

    elapsed = perf_counter() - started_at
    suffix = ""
    if result_formatter is not None:
        suffix = f" ({result_formatter(result)})"
    _log_stage(label, f"done in {elapsed:.2f}s{suffix}")
    return result


def _run_agent_stage(
    label: str,
    agent: object,
    input_data: dict[str, object],
    workspace: Workspace,
) -> int:
    """Run one agent with timing logs and return appended commit count."""

    return _run_stage(
        label,
        lambda: _run_agent(agent, input_data, workspace),
        result_formatter=lambda count: f"{count} commits",
    )


SourceAgentTask = tuple[str, object, dict[str, object]]


def _run_agent_group_stage(
    label: str,
    tasks: Sequence[SourceAgentTask],
    workspace: Workspace,
) -> int:
    """Run independent source agents concurrently, then append commits in order."""

    def run_task(task: SourceAgentTask) -> list[ClaimEvidenceCommit]:
        task_label, agent, input_data = task
        _log_stage(task_label, "start")
        started_at = perf_counter()
        try:
            commits = agent.analyze(input_data, workspace)
        except Exception:
            _log_stage(task_label, f"failed after {perf_counter() - started_at:.2f}s")
            raise
        _log_stage(
            task_label,
            f"done in {perf_counter() - started_at:.2f}s ({len(commits)} commits)",
        )
        return commits

    def run_group() -> int:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            ordered_commit_batches = [future.result() for future in futures]

        total_commits = 0
        for commits in ordered_commit_batches:
            total_commits += _add_commits(workspace, commits)
        return total_commits

    return _run_stage(
        label,
        run_group,
        result_formatter=lambda count: f"{count} commits",
    )


def _add_commits(
    workspace: Workspace,
    commits: list[ClaimEvidenceCommit],
) -> int:
    """Append commits to their branches and return the count."""

    for commit in commits:
        add_commit(workspace, commit.branch_name, commit)
    return len(commits)


def _log_stage(label: str, message: str) -> None:
    """Print a compact pipeline progress log line."""

    print(f"[stage] {label}: {message}", flush=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for the local pipeline."""

    parser = argparse.ArgumentParser(description="Run the EVIR research pipeline.")
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker to research. The mock provider currently supports AAPL.",
    )
    parser.add_argument(
        "--data-provider",
        default=None,
        help=(
            "Data provider to use. Available providers: mock, real. "
            "Defaults to config.USE_REAL_DATA."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help=(
            "LLM provider for optional reasoning. Available providers: mock, "
            "deepseek, none. Defaults to config.USE_LLM / config.LLM_PROVIDER."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=MODEL_NAME,
        help="Optional model name for the selected LLM provider.",
    )
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        help="Disable local LLM response cache for this run.",
    )
    parser.add_argument(
        "--llm-cache-ttl-hours",
        type=int,
        default=LLM_CACHE_TTL_HOURS,
        help="Maximum age for cached LLM responses. Defaults to config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save workspace JSON files.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory used to save Markdown and JSON reports.",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=12,
        help="Number of Markdown report preview lines to print. Use 0 to disable.",
    )
    return parser.parse_args(argv)


def _resolve_llm_client(
    llm_provider: str | BaseLLMClient | None,
    llm_model: str | None,
    use_llm_cache: bool = ENABLE_LLM_CACHE,
    llm_cache_ttl_hours: int = LLM_CACHE_TTL_HOURS,
) -> BaseLLMClient | None:
    """Resolve an optional LLM client for pipeline agents."""

    if llm_provider is None:
        if not USE_LLM:
            return None
        llm_provider = LLM_PROVIDER
    if isinstance(llm_provider, BaseLLMClient):
        return llm_provider
    if llm_provider.lower().strip() in {"none", "off", "disabled"}:
        return None
    try:
        client = get_llm_client(llm_provider, model=llm_model)
    except LLMError as error:
        print(
            f"Warning: LLM provider '{llm_provider}' unavailable ({error}). "
            "Falling back to deterministic no-LLM mode."
        )
        return None
    if not use_llm_cache or llm_cache_ttl_hours <= 0:
        return client
    if llm_provider.lower().strip() == "mock":
        return client
    return CachedLLMClient(client, ttl_hours=llm_cache_ttl_hours)


def _run_agent(
    agent: object,
    input_data: dict[str, object],
    workspace: Workspace,
) -> int:
    """Run an agent and append each structured commit to its branch."""

    commits = agent.analyze(input_data, workspace)
    for commit in commits:
        add_commit(workspace, commit.branch_name, commit)
    return len(commits)


def _print_commits_by_branch(workspace: Workspace) -> None:
    """Print all commits grouped by branch for quick local inspection."""

    for branch_name, branch in workspace.branches.items():
        print(f"[{branch_name}]")
        if not branch.commits:
            print("  (no commits)")
            continue

        for commit in branch.commits:
            _print_commit(commit)
        print()


def _print_commit(commit: ClaimEvidenceCommit) -> None:
    """Print a compact commit summary."""

    print(f"  - {commit.commit_id}")
    print(f"    claim: {commit.claim}")
    print(f"    evidence: {commit.evidence.source} / {commit.evidence.metric_name}")
    print(f"    source_type: {commit.evidence.source_type}")
    print(f"    confidence: {commit.confidence}")
    print(f"    risk_tag: {commit.risk_tag}")
    print(f"    time_horizon: {commit.time_horizon}")
    print(f"    evidence_quality_score: {commit.evidence_quality_score}")
    print(f"    temporal_status: {commit.temporal_status}")


def _print_merge_result(merge_result: MergeResult, _agent: object = None) -> None:
    """Print the merge/decision result in a readable audit format."""

    ds = merge_result.decision_scores
    print("[decision-result]")
    print(f"  final_recommendation: {merge_result.final_recommendation}")
    print(f"  confidence: {merge_result.confidence}")
    print(f"  directional_conviction: {ds.directional_conviction}")
    print(f"  entry_timing: {ds.entry_timing}")
    print(f"  risk_level: {ds.risk_level}")
    print(f"  position_sizing_suggestion: {ds.position_sizing_suggestion}")
    print()

    print("  supporting_claims:")
    for claim in merge_result.main_supporting_claims:
        print(f"    - {claim}")

    print("  opposing_claims:")
    for claim in merge_result.main_opposing_claims:
        print(f"    - {claim}")

    print("  key_conflicts:")
    if not merge_result.key_conflicts:
        print("    - none")
    for conflict in merge_result.key_conflicts:
        print(f"    - {conflict.conflict_type} ({conflict.severity})")
        print(f"      claim_a: {conflict.claim_a}")
        print(f"      claim_b: {conflict.claim_b}")
        print(f"      explanation: {conflict.explanation}")

    print(f"  risk_adjustment: {merge_result.risk_adjustment}")
    print(f"  decision_rationale: {merge_result.decision_rationale}")
    print("  conditions_for_revision:")
    for condition in merge_result.conditions_for_revision:
        print(f"    - {condition}")


def _print_report_preview(markdown_report: str, max_lines: int = 12) -> None:
    """Print a short preview of the generated Markdown report."""

    print("[report-preview]")
    for line in markdown_report.splitlines()[:max_lines]:
        print(line)


def _print_evaluation_result(evaluation_result: EvaluationResult) -> None:
    """Print evaluation scores in a compact audit format."""

    print("[evaluation]")
    print(
        "  evidence_coverage_score: "
        f"{evaluation_result.evidence_coverage_score:.2f}"
    )
    print(
        "  temporal_validity_score: "
        f"{evaluation_result.temporal_validity_score:.2f}"
    )
    print(
        "  conflict_coverage_score: "
        f"{evaluation_result.conflict_coverage_score:.2f}"
    )
    print(
        "  decision_traceability_score: "
        f"{evaluation_result.decision_traceability_score:.2f}"
    )
    print(
        "  audit_completeness_score: "
        f"{evaluation_result.audit_completeness_score:.2f}"
    )
    print(f"  overall_score: {evaluation_result.overall_score:.2f}")


def _print_revision_result(revision_result: RevisionResult) -> None:
    """Print the decision revision result in a readable audit format."""

    print("[revision-result]")
    print(f"  previous_recommendation: {revision_result.previous_recommendation}")
    print(f"  revised_recommendation: {revision_result.revised_recommendation}")

    print("  key_changes:")
    for change in revision_result.key_changes:
        print(f"    - {change}")

    print("  revision_records:")
    for record in revision_result.revision_records:
        print(f"    - {record.revision_status} / {record.impact_on_decision}")
        print(f"      branch: {record.original_branch}")
        print(f"      claim: {record.original_claim}")
        print(f"      explanation: {record.explanation}")

    print(f"  revision_rationale: {revision_result.revision_rationale}")
    print("  updated_conditions_for_revision:")
    for condition in revision_result.updated_conditions_for_revision:
        print(f"    - {condition}")


if __name__ == "__main__":
    main()
