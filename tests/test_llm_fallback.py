"""Tests for LLM fallback behavior."""

from pathlib import Path

from main import run_pipeline


def test_deepseek_missing_key_falls_back_without_crashing(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """DeepSeek mode without an API key should warn and continue deterministically."""

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    workspace, merge_result, markdown_path, json_path = run_pipeline(
        ticker="AAPL",
        data_provider="mock",
        llm_provider="deepseek",
        output_dir=tmp_path / "outputs",
        report_dir=tmp_path / "reports",
        preview_lines=0,
    )
    captured = capsys.readouterr()

    assert workspace.ticker == "AAPL"
    assert merge_result.final_recommendation in {"Buy", "Hold", "Sell", "Avoid"}
    assert "Falling back to deterministic no-LLM mode" in captured.out
    assert Path(markdown_path).exists()
    assert Path(json_path).exists()
