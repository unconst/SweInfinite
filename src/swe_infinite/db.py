"""
Database layer for the SWE-rebench pipeline.

Manages a local SQLite database (pipeline.db) that stores:
  - Raw GitHub Archive events (PullRequestEvent, IssuesEvent)
  - Linked candidate tasks (issue + PR pairs passing filters)
  - Final task instances in SWE-rebench schema
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .paths import DEFAULT_DB_PATH

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Raw events from GitHub Archive
CREATE TABLE IF NOT EXISTS events (
    id            TEXT PRIMARY KEY,
    type          TEXT    NOT NULL,    -- PullRequestEvent | IssuesEvent
    repo_name     TEXT    NOT NULL,
    repo_language TEXT,
    action        TEXT,                -- opened, closed, reopened, ...
    number        INTEGER NOT NULL,    -- issue or PR number
    title         TEXT,
    body          TEXT,
    merged        INTEGER DEFAULT 0,   -- 1 if PR was merged
    base_sha      TEXT,
    head_sha      TEXT,
    merge_commit_sha TEXT,
    default_branch TEXT,
    created_at    TEXT,
    ingested_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_repo   ON events (repo_name);
CREATE INDEX IF NOT EXISTS idx_events_type   ON events (type);
CREATE INDEX IF NOT EXISTS idx_events_number ON events (repo_name, number);

-- Linked candidates (issue <-> PR) that pass preliminary filters
CREATE TABLE IF NOT EXISTS candidates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name     TEXT    NOT NULL,
    issue_number  INTEGER NOT NULL,
    pr_number     INTEGER NOT NULL,
    issue_title   TEXT,
    issue_body    TEXT,
    pr_title      TEXT,
    pr_body       TEXT,
    status        TEXT    DEFAULT 'pending',   -- pending | extracted | failed
    created_at    TEXT    NOT NULL,
    UNIQUE(repo_name, issue_number, pr_number)
);

CREATE INDEX IF NOT EXISTS idx_candidates_repo   ON candidates (repo_name);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates (status);

-- Final task instances in SWE-rebench format
CREATE TABLE IF NOT EXISTS tasks (
    instance_id       TEXT PRIMARY KEY,
    repo_name         TEXT NOT NULL,
    pr_number         INTEGER,
    base_commit       TEXT,
    version           TEXT,
    problem_statement TEXT,
    patch             TEXT,
    test_patch        TEXT,
    install_config    TEXT,    -- JSON string
    license_name      TEXT,
    meta              TEXT,    -- JSON string
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_repo ON tasks (repo_name);

-- Track which GH Archive hours have been ingested
CREATE TABLE IF NOT EXISTS ingested_hours (
    hour_key TEXT PRIMARY KEY,   -- e.g. "2025-06-01-14"
    events_stored INTEGER DEFAULT 0,
    ingested_at   TEXT NOT NULL
);

-- Repository quality cache (avoid redundant GitHub API calls)
CREATE TABLE IF NOT EXISTS repo_quality (
    repo_name          TEXT PRIMARY KEY,
    stars              INTEGER DEFAULT 0,
    forks              INTEGER DEFAULT 0,
    contributors       INTEGER DEFAULT 0,
    has_ci             INTEGER DEFAULT 0,
    has_test_framework INTEGER DEFAULT 0,
    is_archived        INTEGER DEFAULT 0,
    is_fork            INTEGER DEFAULT 0,
    passes             INTEGER DEFAULT 0,
    reason             TEXT DEFAULT '',
    checked_at         TEXT NOT NULL
);

-- Validation results for task candidates
CREATE TABLE IF NOT EXISTS validation_results (
    instance_id    TEXT PRIMARY KEY,
    status         TEXT NOT NULL,       -- validated | install_failed | tests_pass_without_fix | ...
    fail_to_pass   TEXT DEFAULT '[]',   -- JSON array of test nodeids
    pass_to_pass   TEXT DEFAULT '[]',   -- JSON array of test nodeids
    requirements   TEXT DEFAULT '',     -- pip freeze output
    error          TEXT DEFAULT '',
    validated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_validation_status ON validation_results (status);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a new SQLite connection with WAL mode and foreign keys."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create tables and indexes if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Convenience CRUD
# ---------------------------------------------------------------------------


def upsert_event(conn: sqlite3.Connection, event: dict) -> None:
    """Insert or replace a single event row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO events
            (id, type, repo_name, repo_language, action, number,
             title, body, merged, base_sha, head_sha, merge_commit_sha,
             default_branch, created_at, ingested_at)
        VALUES
            (:id, :type, :repo_name, :repo_language, :action, :number,
             :title, :body, :merged, :base_sha, :head_sha, :merge_commit_sha,
             :default_branch, :created_at, :ingested_at)
        """,
        event,
    )


def upsert_events_batch(conn: sqlite3.Connection, events: list[dict]) -> int:
    """Insert or replace a batch of events. Returns count inserted."""
    if not events:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO events
            (id, type, repo_name, repo_language, action, number,
             title, body, merged, base_sha, head_sha, merge_commit_sha,
             default_branch, created_at, ingested_at)
        VALUES
            (:id, :type, :repo_name, :repo_language, :action, :number,
             :title, :body, :merged, :base_sha, :head_sha, :merge_commit_sha,
             :default_branch, :created_at, :ingested_at)
        """,
        events,
    )
    conn.commit()
    return len(events)


def insert_candidate(conn: sqlite3.Connection, candidate: dict) -> bool:
    """Insert a candidate row. Returns True if inserted, False on duplicate."""
    try:
        conn.execute(
            """
            INSERT INTO candidates
                (repo_name, issue_number, pr_number, issue_title, issue_body,
                 pr_title, pr_body, status, created_at)
            VALUES
                (:repo_name, :issue_number, :pr_number, :issue_title, :issue_body,
                 :pr_title, :pr_body, :status, :created_at)
            """,
            candidate,
        )
        return True
    except sqlite3.IntegrityError:
        return False


def update_candidate_status(
    conn: sqlite3.Connection, candidate_id: int, status: str
) -> None:
    """Update the status of a candidate."""
    conn.execute(
        "UPDATE candidates SET status = ? WHERE id = ?",
        (status, candidate_id),
    )
    conn.commit()


def insert_task(conn: sqlite3.Connection, task: dict) -> None:
    """Insert or replace a final task instance."""
    conn.execute(
        """
        INSERT OR REPLACE INTO tasks
            (instance_id, repo_name, pr_number, base_commit, version,
             problem_statement, patch, test_patch, install_config,
             license_name, meta, created_at)
        VALUES
            (:instance_id, :repo_name, :pr_number, :base_commit, :version,
             :problem_statement, :patch, :test_patch, :install_config,
             :license_name, :meta, :created_at)
        """,
        task,
    )
    conn.commit()


def mark_hour_ingested(
    conn: sqlite3.Connection, hour_key: str, events_stored: int
) -> None:
    """Record that a GH Archive hour has been ingested."""
    from datetime import datetime, timezone

    conn.execute(
        """
        INSERT OR REPLACE INTO ingested_hours (hour_key, events_stored, ingested_at)
        VALUES (?, ?, ?)
        """,
        (hour_key, events_stored, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def is_hour_ingested(conn: sqlite3.Connection, hour_key: str) -> bool:
    """Check if a GH Archive hour has already been ingested."""
    row = conn.execute(
        "SELECT 1 FROM ingested_hours WHERE hour_key = ?", (hour_key,)
    ).fetchone()
    return row is not None


def insert_validation_result(conn: sqlite3.Connection, result: dict) -> None:
    """Insert or replace a validation result."""
    conn.execute(
        """
        INSERT OR REPLACE INTO validation_results
            (instance_id, status, fail_to_pass, pass_to_pass,
             requirements, error, validated_at)
        VALUES
            (:instance_id, :status, :fail_to_pass, :pass_to_pass,
             :requirements, :error, :validated_at)
        """,
        result,
    )
    conn.commit()


def get_validation_result(
    conn: sqlite3.Connection, instance_id: str,
) -> dict | None:
    """Get a cached validation result for an instance."""
    row = conn.execute(
        "SELECT * FROM validation_results WHERE instance_id = ?",
        (instance_id,),
    ).fetchone()
    return dict(row) if row else None


def get_stats(conn: sqlite3.Connection) -> dict:
    """Return pipeline statistics."""
    stats = {}
    for table in ("events", "candidates", "tasks"):
        row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
        stats[table] = row["cnt"]

    # Breakdown by event type
    rows = conn.execute(
        "SELECT type, COUNT(*) AS cnt FROM events GROUP BY type"
    ).fetchall()
    stats["events_by_type"] = {r["type"]: r["cnt"] for r in rows}

    # Candidate status breakdown
    rows = conn.execute(
        "SELECT status, COUNT(*) AS cnt FROM candidates GROUP BY status"
    ).fetchall()
    stats["candidates_by_status"] = {r["status"]: r["cnt"] for r in rows}

    # Unique repos in events
    row = conn.execute(
        "SELECT COUNT(DISTINCT repo_name) AS cnt FROM events"
    ).fetchone()
    stats["unique_repos"] = row["cnt"]

    # Ingested hours
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM ingested_hours"
    ).fetchone()
    stats["hours_ingested"] = row["cnt"]

    # Enriched PRs (have title filled in)
    row = conn.execute(
        """SELECT COUNT(*) AS cnt FROM events
           WHERE type = 'PullRequestEvent' AND merged = 1 AND title != ''"""
    ).fetchone()
    stats["prs_enriched"] = row["cnt"]

    # Total merged PRs
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM events WHERE type = 'PullRequestEvent' AND merged = 1"
    ).fetchone()
    stats["prs_merged_total"] = row["cnt"]

    # Validation stats
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM validation_results GROUP BY status"
        ).fetchall()
        stats["validations_by_status"] = {r["status"]: r["cnt"] for r in rows}
    except sqlite3.OperationalError:
        stats["validations_by_status"] = {}

    # Repo quality stats
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM repo_quality WHERE passes = 1"
        ).fetchone()
        stats["repos_passing_quality"] = row["cnt"]
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM repo_quality"
        ).fetchone()
        stats["repos_quality_checked"] = row["cnt"]
    except sqlite3.OperationalError:
        stats["repos_passing_quality"] = 0
        stats["repos_quality_checked"] = 0

    return stats
