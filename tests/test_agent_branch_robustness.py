"""Tests for agent behavior when optional branches are absent."""

from agents import BearAgent, BullAgent
from memory.workspace import create_workspace


def test_bull_and_bear_agents_tolerate_missing_source_branches() -> None:
    """Thesis agents should return no commits instead of crashing."""

    workspace = create_workspace(
        ticker="AAPL",
        company_name="Apple Inc.",
        research_question="Test missing branches.",
    )

    assert BullAgent().analyze({}, workspace) == []
    assert BearAgent().analyze({}, workspace) == []
