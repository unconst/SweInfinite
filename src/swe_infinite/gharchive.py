"""
GitHub Archive ingester for the SWE-rebench pipeline.

Downloads hourly .json.gz NDJSON files from https://data.gharchive.org,
stream-parses them, and stores relevant events into the local SQLite database.

GH Archive event format notes:
  - PullRequestEvent with action="merged" indicates a merged PR
  - PR objects are minimal (no title/body), only base/head SHAs and PR number
  - IssuesEvent with action="closed" has full issue data (title, body, etc.)
  - Repo language is NOT available in events; enriched via GitHub API later

After ingestion, a lightweight GitHub API enrichment pass fetches PR title/body
and repo language for merged PRs, enabling issue linking in the filter stage.
"""

from __future__ import annotations

import gzip
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import orjson

from .db import (
    DEFAULT_DB_PATH,
    get_connection,
    init_db,
    is_hour_ingested,
    mark_hour_ingested,
    upsert_events_batch,
)

log = logging.getLogger("swe-infinite.gharchive")

GH_ARCHIVE_URL = "https://data.gharchive.org/{date}-{hour}.json.gz"

WANTED_TYPES = {"PullRequestEvent", "IssuesEvent"}
BATCH_SIZE = 5000


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def _extract_pr_event(raw: dict) -> dict | None:
    """Extract fields from a PullRequestEvent.

    GH Archive uses action="merged" for merged PRs (not "closed" + merged=true).
    The PR object is minimal: only base/head SHAs, number, and API URL.
    Title, body, and language are populated later via GitHub API enrichment.
    """
    payload = raw.get("payload", {})
    action = payload.get("action")

    # We want merged PRs
    if action != "merged":
        return None

    pr = payload.get("pull_request", {})
    repo = raw.get("repo", {})
    repo_name = repo.get("name", "")

    base = pr.get("base", {})
    head = pr.get("head", {})

    return {
        "id": str(raw.get("id", "")),
        "type": "PullRequestEvent",
        "repo_name": repo_name,
        "repo_language": None,  # Not in GH Archive; enriched later
        "action": "merged",
        "number": payload.get("number") or pr.get("number"),
        "title": "",     # Not in GH Archive; enriched later
        "body": "",      # Not in GH Archive; enriched later
        "merged": 1,
        "base_sha": base.get("sha"),
        "head_sha": head.get("sha"),
        "merge_commit_sha": None,  # Not available in stripped event
        "default_branch": base.get("ref"),  # base ref is typically the default branch
        "created_at": raw.get("created_at", ""),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def _extract_issue_event(raw: dict) -> dict | None:
    """Extract fields from an IssuesEvent.

    GH Archive provides full issue data (title, body, labels, etc.) for issues.
    """
    payload = raw.get("payload", {})
    action = payload.get("action")
    issue = payload.get("issue", {})

    if action != "closed":
        return None

    # Skip pull requests disguised as issues
    if issue.get("pull_request"):
        return None

    repo = raw.get("repo", {})
    repo_name = repo.get("name", "")

    return {
        "id": str(raw.get("id", "")),
        "type": "IssuesEvent",
        "repo_name": repo_name,
        "repo_language": None,
        "action": "closed",
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "body": (issue.get("body") or "")[:50000],
        "merged": 0,
        "base_sha": None,
        "head_sha": None,
        "merge_commit_sha": None,
        "default_branch": None,
        "created_at": issue.get("created_at", ""),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


_EXTRACTORS = {
    "PullRequestEvent": _extract_pr_event,
    "IssuesEvent": _extract_issue_event,
}


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------


def _build_url(dt: datetime) -> str:
    return GH_ARCHIVE_URL.format(date=dt.strftime("%Y-%m-%d"), hour=dt.hour)


def hour_key(dt: datetime) -> str:
    # Use f-string instead of %-H which is not portable across platforms
    return f"{dt.strftime('%Y-%m-%d')}-{dt.hour}"


def download_and_parse_hour(
    dt: datetime,
    db_path: Path = DEFAULT_DB_PATH,
    client: httpx.Client | None = None,
    skip_if_ingested: bool = True,
) -> dict:
    """Download one hourly archive, parse it, store events to DB."""
    url = _build_url(dt)
    label = hour_key(dt)

    if skip_if_ingested:
        conn = get_connection(db_path)
        try:
            if is_hour_ingested(conn, label):
                log.debug("Hour %s already ingested. Skipping.", label)
                return {"downloaded_bytes": 0, "total_lines": 0, "stored_events": 0, "skipped": 0, "already_ingested": True}
        finally:
            conn.close()

    log.info("Downloading %s ...", url)

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=120, follow_redirects=True)

    try:
        resp = client.get(url)
        if resp.status_code == 404:
            log.warning("Archive not found for %s (404). Skipping.", label)
            return {"downloaded_bytes": 0, "total_lines": 0, "stored_events": 0, "skipped": 0}
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.error("HTTP error downloading %s: %s", label, exc)
        return {"downloaded_bytes": 0, "total_lines": 0, "stored_events": 0, "skipped": 0}
    finally:
        if own_client:
            client.close()

    raw_bytes = resp.content
    downloaded_bytes = len(raw_bytes)
    log.info("Downloaded %s (%.1f MB). Parsing...", label, downloaded_bytes / 1e6)

    conn = get_connection(db_path)
    batch: list[dict] = []
    total_lines = 0
    stored = 0
    skipped = 0

    try:
        decompressed = gzip.decompress(raw_bytes)
        for line in decompressed.split(b"\n"):
            if not line.strip():
                continue
            total_lines += 1

            try:
                raw = orjson.loads(line)
            except orjson.JSONDecodeError:
                skipped += 1
                continue

            event_type = raw.get("type")
            if event_type not in WANTED_TYPES:
                skipped += 1
                continue

            extractor = _EXTRACTORS[event_type]
            event = extractor(raw)
            if event is None:
                skipped += 1
                continue

            batch.append(event)

            if len(batch) >= BATCH_SIZE:
                stored += upsert_events_batch(conn, batch)
                batch.clear()

        if batch:
            stored += upsert_events_batch(conn, batch)
            batch.clear()

        # Mark hour as ingested
        mark_hour_ingested(conn, label, stored)

    finally:
        conn.close()

    log.info(
        "Parsed %s: %d lines, %d events stored (%d skipped)",
        label, total_lines, stored, skipped,
    )
    return {
        "downloaded_bytes": downloaded_bytes,
        "total_lines": total_lines,
        "stored_events": stored,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# GitHub API enrichment
# ---------------------------------------------------------------------------

_GITHUB_API = "https://api.github.com"


def _get_github_headers() -> dict:
    """Build headers for GitHub API, using token if available."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def enrich_merged_prs(db_path: Path = DEFAULT_DB_PATH, limit: int = 50) -> dict:
    """Fetch PR title/body and repo language from GitHub API for un-enriched merged PRs.

    Prioritizes PRs from repos that also have closed issues, since those are
    the only repos that can produce valid candidates.

    Works with or without a GITHUB_TOKEN:
      - Unauthenticated: 60 requests/hour
      - Authenticated: 5,000 requests/hour

    Returns stats dict.
    """
    conn = get_connection(db_path)
    try:
        # Prioritize PRs from repos that also have closed issues (high-value repos)
        rows = conn.execute(
            """
            SELECT pr.id, pr.repo_name, pr.number
            FROM events pr
            INNER JOIN (
                SELECT DISTINCT repo_name
                FROM events
                WHERE type = 'IssuesEvent'
            ) iss ON pr.repo_name = iss.repo_name
            WHERE pr.type = 'PullRequestEvent'
              AND pr.merged = 1
              AND (pr.title IS NULL OR pr.title = '')
            ORDER BY pr.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        log.info("No PRs need enrichment.")
        return {"enriched": 0, "failed": 0, "rate_limited": False}

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    rate_type = "authenticated (5000/hr)" if token else "unauthenticated (60/hr)"
    log.info("Enriching %d merged PRs via GitHub API [%s]...", len(rows), rate_type)

    headers = _get_github_headers()
    client = httpx.Client(timeout=30, follow_redirects=True, headers=headers)

    enriched = 0
    failed = 0
    rate_limited = False

    conn = get_connection(db_path)
    try:
        for row in rows:
            repo_name = row["repo_name"]
            pr_number = row["number"]

            # Fetch PR details
            pr_url = f"{_GITHUB_API}/repos/{repo_name}/pulls/{pr_number}"
            try:
                resp = client.get(pr_url)
            except httpx.HTTPError as exc:
                log.warning("HTTP error fetching %s: %s", pr_url, exc)
                failed += 1
                continue

            if resp.status_code in (403, 429):
                remaining = resp.headers.get("x-ratelimit-remaining", "?")
                reset_ts = resp.headers.get("x-ratelimit-reset")
                if reset_ts:
                    reset_dt = datetime.fromtimestamp(int(reset_ts), tz=timezone.utc)
                    wait_secs = max(0, (reset_dt - datetime.now(timezone.utc)).total_seconds()) + 5
                    if wait_secs <= 900:  # Wait up to 15 min
                        log.info("Rate limited. Waiting %.0fs for reset...", wait_secs)
                        time.sleep(wait_secs)
                        continue  # Retry this PR
                log.warning("Rate limited (remaining: %s). Stopping enrichment.", remaining)
                rate_limited = True
                break

            if resp.status_code != 200:
                log.debug("Got %d for %s PR#%d", resp.status_code, repo_name, pr_number)
                failed += 1
                continue

            pr_data = resp.json()

            title = pr_data.get("title", "")
            body = (pr_data.get("body") or "")[:50000]
            merge_commit_sha = pr_data.get("merge_commit_sha")
            language = pr_data.get("base", {}).get("repo", {}).get("language")

            conn.execute(
                """
                UPDATE events
                SET title = ?, body = ?, merge_commit_sha = ?, repo_language = ?
                WHERE id = ?
                """,
                (title, body, merge_commit_sha, language, row["id"]),
            )
            enriched += 1

            # Be polite: small delay between requests
            delay = 0.5 if token else 1.5
            time.sleep(delay)

        conn.commit()
    finally:
        client.close()
        conn.close()

    log.info("Enrichment: %d done, %d failed, rate_limited=%s", enriched, failed, rate_limited)
    return {"enriched": enriched, "failed": failed, "rate_limited": rate_limited}


# ---------------------------------------------------------------------------
# Backfill + incremental modes
# ---------------------------------------------------------------------------


def ingest_range(
    start: datetime,
    end: datetime,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    """Download and ingest all hourly archives in [start, end)."""
    init_db(db_path)

    current = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)

    totals = {"hours": 0, "hours_new": 0, "stored_events": 0, "total_lines": 0, "skipped": 0}

    client = httpx.Client(timeout=120, follow_redirects=True)
    try:
        while current < end:
            stats = download_and_parse_hour(current, db_path, client)
            totals["hours"] += 1
            if not stats.get("already_ingested"):
                totals["hours_new"] += 1
            totals["stored_events"] += stats["stored_events"]
            totals["total_lines"] += stats["total_lines"]
            totals["skipped"] += stats["skipped"]
            current += timedelta(hours=1)
            time.sleep(0.3)
    finally:
        client.close()

    log.info(
        "Backfill: %d hours checked (%d new), %d events stored",
        totals["hours"], totals["hours_new"], totals["stored_events"],
    )
    return totals


def ingest_latest(db_path: Path = DEFAULT_DB_PATH) -> dict:
    """Download the most recently completed hour from GH Archive."""
    now = datetime.now(timezone.utc)
    latest = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    init_db(db_path)
    return download_and_parse_hour(latest, db_path)
