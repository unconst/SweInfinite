"""
Database layer for the SWE-rebench pipeline.

Manages a local SQLite database (pipeline.db) that stores:
  - Raw GitHub Archive events (PullRequestEvent, IssuesEvent)
  - PR annotations with extraction metadata
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
    ingested_at   TEXT    NOT NULL,
    processed     INTEGER DEFAULT 0    -- 1 once fully processed (task created or skipped)
);

CREATE INDEX IF NOT EXISTS idx_events_repo   ON events (repo_name);
CREATE INDEX IF NOT EXISTS idx_events_type   ON events (type);
CREATE INDEX IF NOT EXISTS idx_events_number ON events (repo_name, number);

-- Annotation table: stores metadata for every enriched PR (nothing is thrown away)
CREATE TABLE IF NOT EXISTS pr_annotations (
    event_id          TEXT PRIMARY KEY,
    repo_name         TEXT NOT NULL,
    pr_number         INTEGER NOT NULL,
    language          TEXT,                  -- detected repo language (lowercased)
    issue_num         INTEGER,              -- linked issue number (NULL = none found)
    issue_title       TEXT,
    issue_body        TEXT,
    issue_body_len    INTEGER DEFAULT 0,    -- length of issue body (for quick filtering)
    issue_quality_ok  INTEGER DEFAULT 0,    -- 1 if passed heuristic issue pre-filter
    issue_quality_reason TEXT,              -- why it failed heuristic pre-filter (NULL = passed)
    pr_title          TEXT,
    pr_body           TEXT,
    base_sha          TEXT,
    head_sha          TEXT,
    merge_commit_sha  TEXT,
    -- Extraction results (filled after clone/diff, NULL until then)
    has_test_changes  INTEGER,              -- 1 if diff includes test files
    has_solution_changes INTEGER,           -- 1 if diff includes non-test files
    solution_files    INTEGER,              -- number of files in solution patch
    solution_lines    INTEGER,              -- changed lines in solution patch
    license_name      TEXT,                 -- detected license SPDX
    license_ok        INTEGER,              -- 1 if permissive/unknown
    -- Repo quality (filled from repo_quality table or API)
    repo_stars        INTEGER,
    -- Status tracking
    status            TEXT DEFAULT 'annotated',  -- annotated | extracting | extracted | task_created | failed
    skip_reason       TEXT,                 -- summary of why this PR can't become a task (NULL = viable)
    annotated_at      TEXT NOT NULL,
    extracted_at      TEXT,
    UNIQUE(repo_name, pr_number)
);

CREATE INDEX IF NOT EXISTS idx_annotations_status   ON pr_annotations (status);
CREATE INDEX IF NOT EXISTS idx_annotations_language ON pr_annotations (language);
CREATE INDEX IF NOT EXISTS idx_annotations_repo     ON pr_annotations (repo_name);

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
    # Migration: add 'processed' column if missing (for existing DBs)
    try:
        conn.execute("SELECT processed FROM events LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE events ADD COLUMN processed INTEGER DEFAULT 0")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# PR Annotations CRUD
# ---------------------------------------------------------------------------


def upsert_annotation(conn: sqlite3.Connection, ann: dict) -> None:
    """Insert or update an annotation row for an enriched PR."""
    conn.execute(
        """
        INSERT INTO pr_annotations
            (event_id, repo_name, pr_number, language, issue_num,
             issue_title, issue_body, issue_body_len, issue_quality_ok,
             issue_quality_reason, pr_title, pr_body,
             base_sha, head_sha, merge_commit_sha,
             status, skip_reason, annotated_at)
        VALUES
            (:event_id, :repo_name, :pr_number, :language, :issue_num,
             :issue_title, :issue_body, :issue_body_len, :issue_quality_ok,
             :issue_quality_reason, :pr_title, :pr_body,
             :base_sha, :head_sha, :merge_commit_sha,
             :status, :skip_reason, :annotated_at)
        ON CONFLICT(repo_name, pr_number) DO UPDATE SET
            language=excluded.language,
            issue_num=excluded.issue_num,
            issue_title=excluded.issue_title,
            issue_body=excluded.issue_body,
            issue_body_len=excluded.issue_body_len,
            issue_quality_ok=excluded.issue_quality_ok,
            issue_quality_reason=excluded.issue_quality_reason,
            pr_title=excluded.pr_title,
            pr_body=excluded.pr_body,
            base_sha=excluded.base_sha,
            head_sha=excluded.head_sha,
            merge_commit_sha=excluded.merge_commit_sha,
            status=excluded.status,
            skip_reason=excluded.skip_reason,
            annotated_at=excluded.annotated_at
        """,
        ann,
    )
    conn.commit()


def update_annotation_extraction(conn: sqlite3.Connection, event_id: str, data: dict) -> None:
    """Update an annotation row with extraction results (post-clone/diff)."""
    conn.execute(
        """
        UPDATE pr_annotations SET
            has_test_changes = :has_test_changes,
            has_solution_changes = :has_solution_changes,
            solution_files = :solution_files,
            solution_lines = :solution_lines,
            license_name = :license_name,
            license_ok = :license_ok,
            repo_stars = :repo_stars,
            status = :status,
            skip_reason = :skip_reason,
            extracted_at = :extracted_at
        WHERE event_id = :event_id
        """,
        {**data, "event_id": event_id},
    )
    conn.commit()


def get_extractable_annotations(
    conn: sqlite3.Connection,
    languages: set[str] | None = None,
    min_issue_length: int = 10,
    limit: int = 1,
    allow_no_issue: bool = False,
) -> list[dict]:
    """Return annotated PRs that look viable for extraction.

    Picks candidates that:
    - Have a linked issue with sufficient body length (or, when
      allow_no_issue is True, have a PR body usable as problem source)
    - Are in a supported language
    - Haven't been extracted yet
    - Passed the heuristic issue quality check
    """
    if allow_no_issue:
        # Accept PRs with a linked issue OR those with a usable PR body
        sql = """
            SELECT * FROM pr_annotations
            WHERE status = 'annotated'
              AND issue_quality_ok = 1
              AND (
                  (issue_num IS NOT NULL AND issue_body_len > ?)
                  OR (issue_num IS NULL AND pr_body IS NOT NULL AND length(pr_body) > 20)
              )
        """
    else:
        sql = """
            SELECT * FROM pr_annotations
            WHERE status = 'annotated'
              AND issue_num IS NOT NULL
              AND issue_body_len > ?
              AND issue_quality_ok = 1
        """
    params: list = [min_issue_length]

    if languages:
        placeholders = ",".join("?" for _ in languages)
        sql += f" AND language IN ({placeholders})"
        params.extend(sorted(languages))

    sql += " ORDER BY issue_body_len DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_annotation_stats(conn: sqlite3.Connection) -> dict:
    """Return a breakdown of pr_annotations for the --status command."""
    stats: dict = {}
    try:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM pr_annotations").fetchone()
        stats["total"] = row["cnt"]

        # By language
        rows = conn.execute(
            "SELECT language, COUNT(*) AS cnt FROM pr_annotations GROUP BY language ORDER BY cnt DESC"
        ).fetchall()
        stats["by_language"] = {(r["language"] or "unknown"): r["cnt"] for r in rows}

        # By status
        rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM pr_annotations GROUP BY status ORDER BY cnt DESC"
        ).fetchall()
        stats["by_status"] = {r["status"]: r["cnt"] for r in rows}

        # Issue link rate
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM pr_annotations WHERE issue_num IS NOT NULL"
        ).fetchone()
        stats["with_issue"] = row["cnt"]

        # Extraction-ready (have issue OR usable PR body, right language, quality OK)
        row = conn.execute(
            """SELECT COUNT(*) AS cnt FROM pr_annotations
               WHERE issue_quality_ok = 1
                 AND (
                     (issue_num IS NOT NULL AND issue_body_len > 10)
                     OR (issue_num IS NULL AND pr_body IS NOT NULL AND length(pr_body) > 20)
                 )"""
        ).fetchone()
        stats["extraction_ready"] = row["cnt"]

    except sqlite3.OperationalError:
        stats = {"total": 0}
    return stats


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
    for table in ("events", "tasks"):
        row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
        stats[table] = row["cnt"]

    # Breakdown by event type
    rows = conn.execute(
        "SELECT type, COUNT(*) AS cnt FROM events GROUP BY type"
    ).fetchall()
    stats["events_by_type"] = {r["type"]: r["cnt"] for r in rows}

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
