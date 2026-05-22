"""Workspace operations for branch-based investment research."""

from datetime import datetime, timezone
from pathlib import Path

from models.schemas import Branch, ClaimEvidenceCommit, Workspace
from memory.storage import load_json, save_json


def _utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _workspace_path(ticker: str, base_dir: str | Path) -> Path:
    """Build the canonical workspace JSON path for a ticker."""

    return Path(base_dir) / "workspaces" / f"{ticker.upper()}_workspace.json"


def create_workspace(
    ticker: str,
    company_name: str | None,
    research_question: str,
) -> Workspace:
    """Create an empty investment research workspace."""

    return Workspace(
        ticker=ticker.upper(),
        company_name=company_name,
        research_question=research_question,
        created_at=_utc_now(),
        branches={},
    )


def create_branch(workspace: Workspace, branch_name: str, description: str) -> Workspace:
    """Create or replace a branch in a workspace."""

    workspace.branches[branch_name] = Branch(
        branch_name=branch_name,
        description=description,
        commits=[],
    )
    return workspace


def add_commit(
    workspace: Workspace,
    branch_name: str,
    commit: ClaimEvidenceCommit,
) -> Workspace:
    """Append a claim-evidence commit to an existing branch."""

    if branch_name not in workspace.branches:
        raise KeyError(f"Branch '{branch_name}' does not exist in workspace.")

    if commit.branch_name != branch_name:
        raise ValueError(
            f"Commit branch '{commit.branch_name}' does not match '{branch_name}'."
        )

    workspace.branches[branch_name].commits.append(commit)
    return workspace


def save_workspace(workspace: Workspace, base_dir: str | Path) -> Path:
    """Persist a workspace to local JSON storage."""

    path = _workspace_path(workspace.ticker, base_dir)
    return save_json(path, workspace)


def load_workspace(ticker: str, base_dir: str | Path) -> Workspace:
    """Load a workspace from local JSON storage."""

    path = _workspace_path(ticker, base_dir)
    return Workspace.model_validate(load_json(path))

