"""
Candidate filtering for the SWE-rebench pipeline.

Links closed issues to merged PRs using GitHub keyword references
(fixes #N, closes #N, resolves #N) found in PR title/body, then applies
the paper's filtering criteria:

  - Python repo (>75% Python from language field)
  - Issue is closed/resolved
  - PR is merged into default branch
  - PR not linked to multiple issues
  - Issue description > 10 characters
  - PR modifies 1-15 files (checked later at clone stage)

Produces candidate rows in the `candidates` table.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .db import DEFAULT_DB_PATH, get_connection, init_db, insert_candidate

log = logging.getLogger("swe-infinite.filters")

# ---------------------------------------------------------------------------
# Issue-reference regex
# ---------------------------------------------------------------------------

# Matches patterns like "fixes #123", "closes #45", "resolves #7"
# Also handles full URL form: fixes https://github.com/owner/repo/issues/123
_KEYWORD_ISSUE_RE = re.compile(
    r"(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)"
    r"\s+"
    r"(?:"
    r"https?://github\.com/[\w\-\.]+/[\w\-\.]+/issues/(\d+)"
    r"|"
    r"#(\d+)"
    r")",
    re.IGNORECASE,
)


def _extract_linked_issues(text: str) -> list[int]:
    """Extract issue numbers linked via GitHub keywords in text."""
    if not text:
        return []
    results = []
    for match in _KEYWORD_ISSUE_RE.finditer(text):
        # Group 1 = URL form, Group 2 = #N form
        num_str = match.group(1) or match.group(2)
        if num_str:
            results.append(int(num_str))
    return results


# ---------------------------------------------------------------------------
# Main filter pipeline
# ---------------------------------------------------------------------------

# Languages considered "Python" repos (case-insensitive)
_PYTHON_LANGUAGES = {"python"}


def run_filters(
    db_path: Path = DEFAULT_DB_PATH,
    min_issue_length: int = 10,
) -> dict:
    """Link issues to PRs and insert passing candidates into the database.

    Filtering criteria (from the paper, Section 2.1):
      1. Python repository
      2. Issue is closed
      3. PR is merged into the default branch
      4. PR is not linked to multiple issues
      5. Issue description > 10 characters
      6. (File count 1-15 checked later at patch extraction stage)

    Returns stats dict.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    stats = {
        "prs_scanned": 0,
        "non_python_skipped": 0,
        "no_issue_link": 0,
        "multi_issue_skipped": 0,
        "issue_not_found": 0,
        "issue_too_short": 0,
        "candidates_created": 0,
        "duplicates_skipped": 0,
    }

    try:
        # Get all merged PRs that have been enriched (have title/body from GitHub API)
        prs = conn.execute(
            """
            SELECT id, repo_name, repo_language, number, title, body,
                   base_sha, head_sha, merge_commit_sha, default_branch, created_at
            FROM events
            WHERE type = 'PullRequestEvent'
              AND merged = 1
              AND title != ''
            ORDER BY repo_name, number
            """
        ).fetchall()

        log.info("Scanning %d enriched merged PRs for issue links...", len(prs))
        stats["prs_scanned"] = len(prs)

        # Build issue lookup: (repo_name, issue_number) -> issue row
        issues = conn.execute(
            """
            SELECT repo_name, number, title, body
            FROM events
            WHERE type = 'IssuesEvent'
            """
        ).fetchall()

        issue_map: dict[tuple[str, int], sqlite3.Row] = {}
        for issue in issues:
            key = (issue["repo_name"], issue["number"])
            issue_map[key] = issue

        log.info("Built issue lookup: %d closed issues across repos", len(issue_map))

        for pr in prs:
            repo = pr["repo_name"]
            lang = (pr["repo_language"] or "").lower()

            # Filter: Python repo (or unknown language — verified at clone stage)
            if lang and lang not in _PYTHON_LANGUAGES:
                stats["non_python_skipped"] += 1
                continue

            # Parse linked issues from PR title + body
            linked = set()
            linked.update(_extract_linked_issues(pr["title"] or ""))
            linked.update(_extract_linked_issues(pr["body"] or ""))

            if not linked:
                stats["no_issue_link"] += 1
                continue

            # Filter: PR not linked to multiple issues
            if len(linked) > 1:
                stats["multi_issue_skipped"] += 1
                continue

            issue_num = linked.pop()

            # Look up the issue in our events
            issue_row = issue_map.get((repo, issue_num))
            if issue_row is None:
                stats["issue_not_found"] += 1
                continue

            # Filter: Issue description > 10 characters
            issue_body = issue_row["body"] or ""
            if len(issue_body.strip()) <= min_issue_length:
                stats["issue_too_short"] += 1
                continue

            # Passed all preliminary filters — create candidate
            candidate = {
                "repo_name": repo,
                "issue_number": issue_num,
                "pr_number": pr["number"],
                "issue_title": issue_row["title"] or "",
                "issue_body": issue_body,
                "pr_title": pr["title"] or "",
                "pr_body": (pr["body"] or "")[:50000],
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            if insert_candidate(conn, candidate):
                stats["candidates_created"] += 1
            else:
                stats["duplicates_skipped"] += 1

        conn.commit()

    finally:
        conn.close()

    log.info(
        "Filtering complete: %d PRs scanned, %d candidates created",
        stats["prs_scanned"],
        stats["candidates_created"],
    )
    log.info("Filter stats: %s", stats)
    return stats


def get_pending_candidates(
    db_path: Path = DEFAULT_DB_PATH,
    limit: int | None = None,
) -> list[dict]:
    """Retrieve pending candidates for patch extraction."""
    conn = get_connection(db_path)
    try:
        query = """
            SELECT c.id, c.repo_name, c.issue_number, c.pr_number,
                   c.issue_title, c.issue_body, c.pr_title, c.pr_body,
                   e.base_sha, e.head_sha, e.merge_commit_sha, e.default_branch
            FROM candidates c
            JOIN events e ON e.repo_name = c.repo_name
                         AND e.number = c.pr_number
                         AND e.type = 'PullRequestEvent'
            WHERE c.status = 'pending'
            ORDER BY c.repo_name, c.pr_number
        """
        if limit:
            query += f" LIMIT {limit}"

        rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
