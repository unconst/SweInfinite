#!/usr/bin/env python3
"""
SWE Infinite — Validated, quality-scored SWE benchmark task factory.

5-phase pipeline for each task:
  Phase 1: Collection — pick merged PR, enrich via GitHub API, link to issue
  Phase 2: Pre-filtering — language, license, repo quality, issue quality, patch complexity
  Phase 3: Validation — install deps, run tests before/after fix, populate FAIL_TO_PASS
  Phase 4: Quality — LLM scoring (issue clarity, test quality, difficulty), decontamination
  Phase 5: Storage — save validated task to dataset/ with all fields populated

If the event database runs low, downloads the next GH Archive hour automatically.

Usage:
    uv run python swe_infinite.py                       # run continuously (full pipeline)
    uv run python swe_infinite.py --once                # produce one task then exit
    uv run python swe_infinite.py --status              # print stats
    uv run python swe_infinite.py --skip-validation     # skip execution validation
    uv run python swe_infinite.py --skip-quality        # skip LLM quality scoring
    uv run python swe_infinite.py --docker              # use Docker for validation
    uv run python swe_infinite.py --min-stars 10        # require 10+ GitHub stars
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import re
import shutil
import signal
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .db import (
    DEFAULT_DB_PATH,
    get_connection,
    get_stats,
    init_db,
    insert_validation_result,
    is_hour_ingested,
    mark_hour_ingested,
    update_candidate_status,
)
from .decontamination import check_contamination
from .gharchive import download_and_parse_hour
from .language_support import get_handler
from .quality_scorer import assess_quality, heuristic_issue_prefilter
from .recipe_generator import generate_recipe
from .repo_ops import extract_candidate
from .repo_quality import check_repo_quality
from .task_store import build_task_instance, make_instance_id, store_task
from .validator import validate_task
from .versioning import find_version_for_commit
from .paths import DATASET_DIR, REPOS_DIR, LOG_FILE

# Supported languages (extensible — see language_support.py for handlers)
SUPPORTED_LANGUAGES = {"python", "typescript", "javascript", "java", "go"}

log = logging.getLogger("swe-infinite")


def _setup_logging() -> None:
    """Configure logging with both stderr and rotating file output."""
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console handler (stderr — captured by systemd/launchd journal)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # Rotating file handler: 50 MB per file, keep 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)


_shutdown = False


def _handle_signal(signum: int, frame: object) -> None:
    global _shutdown
    log.info("Shutting down after current task...")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# GitHub API helper (single PR enrichment)
# ---------------------------------------------------------------------------

_GITHUB_API = "https://api.github.com"
_gh_client: httpx.Client | None = None


def _get_gh_client() -> httpx.Client:
    global _gh_client
    if _gh_client is None:
        headers = {"Accept": "application/vnd.github.v3+json"}
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        _gh_client = httpx.Client(timeout=30, follow_redirects=True, headers=headers)
    return _gh_client


def _enrich_pr(conn: sqlite3.Connection, event_id: str, repo_name: str, pr_number: int) -> dict | None:
    """Fetch one PR's details from GitHub API. Updates DB. Returns enriched data or None."""
    client = _get_gh_client()
    url = f"{_GITHUB_API}/repos/{repo_name}/pulls/{pr_number}"

    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        log.warning("  API error for %s PR#%d: %s", repo_name, pr_number, exc)
        # Mark so we skip this PR and move on to the next one
        conn.execute("UPDATE events SET title = '[error]' WHERE id = ?", (event_id,))
        conn.commit()
        return None

    if resp.status_code in (403, 429):
        # Rate limited — wait for reset
        remaining = resp.headers.get("x-ratelimit-remaining")
        reset_ts = resp.headers.get("x-ratelimit-reset")
        if reset_ts:
            wait = max(0, int(reset_ts) - int(time.time())) + 5
            if wait <= 3600:
                log.info("  Rate limited. Waiting %ds for reset...", wait)
                time.sleep(wait)
                return _enrich_pr(conn, event_id, repo_name, pr_number)
            else:
                log.warning("  Rate limit reset too far away (%ds). Sleeping 5 min...", wait)
                time.sleep(300)
                return None
        else:
            # 403 without rate limit header — likely access denied, not rate limit
            log.warning("  Got 403 without rate-limit headers (access denied?). Marking as error.")
            conn.execute("UPDATE events SET title = '[error]' WHERE id = ?", (event_id,))
            conn.commit()
            return None

    if resp.status_code == 404:
        log.debug("  PR not found: %s PR#%d (deleted/private)", repo_name, pr_number)
        conn.execute("UPDATE events SET title = '[deleted]' WHERE id = ?", (event_id,))
        conn.commit()
        return None

    if resp.status_code != 200:
        log.debug("  Got %d for %s PR#%d", resp.status_code, repo_name, pr_number)
        # Mark as errored so we don't retry this PR endlessly
        conn.execute("UPDATE events SET title = '[error]' WHERE id = ?", (event_id,))
        conn.commit()
        return None

    data = resp.json()
    title = data.get("title", "")
    body = (data.get("body") or "")[:50000]
    merge_commit_sha = data.get("merge_commit_sha")
    language = data.get("base", {}).get("repo", {}).get("language")

    conn.execute(
        "UPDATE events SET title=?, body=?, merge_commit_sha=?, repo_language=? WHERE id=?",
        (title, body, merge_commit_sha, language, event_id),
    )
    conn.commit()

    return {
        "title": title,
        "body": body,
        "merge_commit_sha": merge_commit_sha,
        "language": language,
    }


# ---------------------------------------------------------------------------
# Issue linking
# ---------------------------------------------------------------------------

_ISSUE_RE = re.compile(
    r"(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+"
    r"(?:https?://github\.com/[\w\-\.]+/[\w\-\.]+/issues/(\d+)|#(\d+))",
    re.IGNORECASE,
)


def _find_linked_issue(title: str, body: str) -> int | None:
    """Find a single linked issue number from PR title+body. Returns None if 0 or >1."""
    issues: set[int] = set()
    for text in (title, body):
        for m in _ISSUE_RE.finditer(text or ""):
            num = m.group(1) or m.group(2)
            if num:
                issues.add(int(num))
    if len(issues) == 1:
        return issues.pop()
    return None


# ---------------------------------------------------------------------------
# Core loop: pick one PR, process it fully
# ---------------------------------------------------------------------------


def _phase_collect(conn: sqlite3.Connection) -> tuple[dict, dict] | str:
    """Phase 1: Pick an un-processed merged PR and enrich it via GitHub API.

    Returns (row, enriched) on success, or a status string ("empty"/"skip") to abort.
    """
    # Pick the next un-enriched merged PR that's in a repo with closed issues
    row = conn.execute(
        """
        SELECT pr.id, pr.repo_name, pr.number, pr.base_sha, pr.head_sha,
               pr.merge_commit_sha, pr.default_branch, pr.title, pr.repo_language
        FROM events pr
        WHERE pr.type = 'PullRequestEvent'
          AND pr.merged = 1
          AND (pr.title IS NULL OR pr.title = '')
          AND EXISTS (
              SELECT 1 FROM events iss
              WHERE iss.type = 'IssuesEvent'
                AND iss.repo_name = pr.repo_name
          )
        ORDER BY pr.created_at DESC
        LIMIT 1
        """,
    ).fetchone()

    if row is None:
        # Try any un-enriched PR (even without matching issue in DB yet)
        row = conn.execute(
            """
            SELECT id, repo_name, number, base_sha, head_sha,
                   merge_commit_sha, default_branch, title, repo_language
            FROM events
            WHERE type = 'PullRequestEvent'
              AND merged = 1
              AND (title IS NULL OR title = '')
            ORDER BY created_at DESC
            LIMIT 1
            """,
        ).fetchone()

    if row is None:
        return "empty"

    repo_name = row["repo_name"]
    pr_number = row["number"]
    event_id = row["id"]

    log.info("Processing: %s PR#%d", repo_name, pr_number)

    enriched = _enrich_pr(conn, event_id, repo_name, pr_number)
    if enriched is None:
        return "skip"

    return row, enriched


def _phase_prefilter(
    conn: sqlite3.Connection, row: dict, enriched: dict, db_path: Path, min_stars: int,
) -> tuple[dict, str, Path] | str:
    """Phase 2: Language, issue quality, patch extraction, and repo quality checks.

    Returns (extracted, lang, repo_dir) on success, or a status string to abort.
    """
    repo_name = row["repo_name"]
    pr_number = row["number"]

    # --- Check language ---
    lang = (enriched.get("language") or "").lower()
    lang_map = {
        "javascript": "javascript", "typescript": "typescript", "python": "python",
        "java": "java", "go": "go", "golang": "go",
    }
    lang = lang_map.get(lang, lang)
    if lang not in SUPPORTED_LANGUAGES:
        log.info("  Skip: language is %s (supported: %s)",
                 lang or "unknown", ", ".join(sorted(SUPPORTED_LANGUAGES)))
        return "skip"

    # --- Find linked issue ---
    issue_num = _find_linked_issue(enriched["title"], enriched["body"])
    if issue_num is None:
        log.info("  Skip: no single issue link found")
        return "skip"

    # --- Look up the issue ---
    issue_row = conn.execute(
        "SELECT title, body FROM events WHERE repo_name=? AND number=? AND type='IssuesEvent'",
        (repo_name, issue_num),
    ).fetchone()

    if issue_row is None:
        log.info("  Skip: issue #%d not in our DB", issue_num)
        return "skip"

    issue_title = issue_row["title"] or ""
    issue_body = issue_row["body"] or ""

    if len(issue_body.strip()) <= 10:
        log.info("  Skip: issue #%d body too short (%d chars)", issue_num, len(issue_body.strip()))
        return "skip"

    # --- Issue quality pre-filter (heuristic, no LLM) ---
    rejection = heuristic_issue_prefilter(issue_title, issue_body)
    if rejection:
        log.info("  Skip: issue quality pre-filter — %s", rejection)
        return "skip"

    log.info("  Matched: issue #%d — extracting patches...", issue_num)

    # --- Clone repo, compute diff, split patches ---
    candidate = {
        "repo_name": repo_name,
        "pr_number": pr_number,
        "issue_number": issue_num,
        "issue_title": issue_title,
        "issue_body": issue_body,
        "pr_title": enriched["title"],
        "base_sha": row["base_sha"],
        "head_sha": row["head_sha"],
        "merge_commit_sha": enriched.get("merge_commit_sha") or row["merge_commit_sha"],
        "id": None,
    }

    try:
        extracted = extract_candidate(candidate)
    except Exception:
        log.exception("  Error extracting %s PR#%d", repo_name, pr_number)
        return "error"

    if extracted is None:
        log.info("  Skip: extraction failed (license/patch/file count/complexity)")
        return "skip"

    repo_dir = Path(extracted["repo_dir"])

    # --- Repo quality gate ---
    try:
        quality = check_repo_quality(
            repo_name, repo_dir=repo_dir, min_stars=min_stars, db_path=db_path,
        )
        if not quality.passes:
            log.info("  Skip: repo quality — %s", quality.reason)
            return "skip"
    except Exception:
        log.debug("  Repo quality check failed, continuing anyway")

    return extracted, lang, repo_dir


def _phase_validate(
    conn: sqlite3.Connection, task: dict, use_docker: bool, skip_validation: bool,
) -> object | None:
    """Phase 3: Run execution validation to populate FAIL_TO_PASS / PASS_TO_PASS.

    Returns a ValidationResult on success (or None if skipped/failed).
    Raises _SkipTask if validation disqualifies the task.
    """
    if skip_validation:
        return None

    log.info("  Validating task (running tests)...")
    try:
        result = validate_task(task, use_docker=use_docker, test_timeout=120)

        # Store validation result in DB
        insert_validation_result(conn, {
            "instance_id": task["instance_id"],
            "status": result.status,
            "fail_to_pass": json.dumps(result.fail_to_pass),
            "pass_to_pass": json.dumps(result.pass_to_pass),
            "requirements": result.requirements,
            "error": result.error,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        })

        if not result.is_valid:
            log.info("  Skip: validation failed — %s (%s)",
                     result.status, (result.error or "")[:100])
            raise _SkipTask()

        log.info("  Validation passed: FAIL_TO_PASS=%d, PASS_TO_PASS=%d",
                 len(result.fail_to_pass), len(result.pass_to_pass))
        return result

    except _SkipTask:
        raise
    except Exception:
        log.exception("  Validation error (continuing without validation)")
        return None


def _phase_quality(
    task: dict, repo_name: str, pr_number: int, skip_quality: bool,
) -> object | None:
    """Phase 4: LLM quality scoring and decontamination check.

    Returns QualityScores on success (or None if skipped/failed).
    Raises _SkipTask if the task is rejected or contaminated.
    """
    quality_scores = None
    if not skip_quality:
        try:
            quality_scores = assess_quality(task)

            if quality_scores.rejection_reason and not quality_scores.passes_threshold:
                log.info("  Skip: quality assessment — %s", quality_scores.rejection_reason)
                raise _SkipTask()

            log.info("  Quality scores: issue=%s, test=%s, difficulty=%s",
                     quality_scores.issue_text_score,
                     quality_scores.test_score,
                     quality_scores.difficulty_score)
        except _SkipTask:
            raise
        except Exception:
            log.debug("  Quality scoring failed, continuing without scores")

    # --- Decontamination check ---
    try:
        contamination = check_contamination(
            task["instance_id"], repo_name, pr_number, task["base_commit"],
        )
        if contamination:
            log.info("  Skip: contaminated — %s", contamination)
            raise _SkipTask()
    except _SkipTask:
        raise
    except Exception:
        log.debug("  Decontamination check failed, continuing")

    return quality_scores


def _phase_store(
    extracted: dict, version: str, install_config: dict,
    validation_result: object | None, quality_scores: object | None,
    db_path: Path,
) -> str:
    """Phase 5: Rebuild the final task with all enrichment data and save it."""
    task = build_task_instance(
        extracted,
        version=version,
        install_config=install_config,
        environment_setup_commit=extracted["base_commit"],
        validation_result=validation_result,
        quality_scores=quality_scores,
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = store_task(task, DATASET_DIR, db_path)

    log.info("  TASK SAVED: %s", task["instance_id"])
    log.info("  File: %s", path)
    log.info("  FAIL_TO_PASS: %d tests, PASS_TO_PASS: %d tests",
             len(task["FAIL_TO_PASS"]), len(task["PASS_TO_PASS"]))

    return "task"


class _SkipTask(Exception):
    """Raised by phase functions to signal that the current task should be skipped."""


def process_one(
    db_path: Path,
    skip_validation: bool = False,
    skip_quality: bool = False,
    use_docker: bool = False,
    min_stars: int = 5,
) -> str:
    """Pick one un-processed merged PR and try to turn it into a dataset task.

    Orchestrates the 5-phase pipeline:
      1. Collection   — enrich PR, link issue
      2. Pre-filter   — language, repo quality, issue quality, patch extraction
      3. Validation   — FAIL_TO_PASS / PASS_TO_PASS via test execution
      4. Quality      — LLM scoring, decontamination
      5. Storage      — save final task JSON

    Returns: "task", "skip", "empty", or "error".
    """
    conn = get_connection(db_path)

    try:
        # Phase 1: Collection
        result = _phase_collect(conn)
        if isinstance(result, str):
            return result
        row, enriched = result

        # Phase 2: Pre-filtering
        result = _phase_prefilter(conn, row, enriched, db_path, min_stars)
        if isinstance(result, str):
            return result
        extracted, lang, repo_dir = result

        # Phase 3: Version, recipe, validation
        version = find_version_for_commit(repo_dir, extracted["base_commit"]) or "unknown"

        install_config = None
        try:
            install_config = generate_recipe(repo_dir, row["repo_name"], max_retries=2)
        except Exception:
            log.debug("  Recipe generation failed, using default")

        if install_config is None:
            handler = get_handler(lang)
            if handler:
                install_config = handler.default_install_recipe(repo_dir)
            else:
                install_config = {"install": "pip install -e .", "pre_install": [], "reqs_path": []}

        log.info("  Install recipe: python=%s, install=%s",
                 install_config.get("python", "?"),
                 install_config.get("install", "?")[:60])

        task = build_task_instance(
            extracted, version=version, install_config=install_config,
            environment_setup_commit=extracted["base_commit"],
        )

        try:
            validation_result = _phase_validate(conn, task, use_docker, skip_validation)
        except _SkipTask:
            return "skip"

        # Phase 4: Quality assessment
        try:
            quality_scores = _phase_quality(
                task, row["repo_name"], row["number"], skip_quality,
            )
        except _SkipTask:
            return "skip"

        # Phase 5: Store
        return _phase_store(
            extracted, version, install_config,
            validation_result, quality_scores, db_path,
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Ensure we have data to work with
# ---------------------------------------------------------------------------


def ensure_events(db_path: Path, min_hours: int = 3) -> None:
    """Download recent GH Archive hours if the DB doesn't have enough data."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM ingested_hours").fetchone()
        have = row["cnt"]
    finally:
        conn.close()

    if have >= min_hours:
        return

    log.info("Database has %d hours. Downloading recent GH Archive data...", have)
    now = datetime.now(timezone.utc)

    # Download the last several hours
    for offset in range(min_hours + 2, 1, -1):
        if _shutdown:
            break
        dt = (now - timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0)
        download_and_parse_hour(dt, db_path)


def download_next_hour(db_path: Path) -> bool:
    """Download the next un-ingested hour from GH Archive. Returns True if a new hour was downloaded."""
    now = datetime.now(timezone.utc)
    latest = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)

    conn = get_connection(db_path)
    try:
        # Find the most recently ingested hour
        row = conn.execute(
            "SELECT hour_key FROM ingested_hours ORDER BY hour_key DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if row:
        # Parse "YYYY-MM-DD-H" back to datetime
        parts = row["hour_key"].rsplit("-", 1)
        last_dt = datetime.strptime(parts[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        last_dt = last_dt.replace(hour=int(parts[1]))
        next_dt = last_dt + timedelta(hours=1)
    else:
        # Start from 6 hours ago
        next_dt = (now - timedelta(hours=8)).replace(minute=0, second=0, microsecond=0)

    if next_dt > latest:
        log.info("All available hours ingested. Waiting for new data...")
        return False

    result = download_and_parse_hour(next_dt, db_path)
    return result.get("stored_events", 0) > 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def print_stats(db_path: Path) -> None:
    try:
        conn = get_connection(db_path)
        stats = get_stats(conn)
        conn.close()
    except Exception:
        log.warning("Database not initialized yet.")
        return

    dataset_count = len(list(DATASET_DIR.glob("*.json"))) if DATASET_DIR.exists() else 0
    prs_total = stats.get("prs_merged_total", 0)
    prs_enriched = stats.get("prs_enriched", 0)

    print()
    print("=" * 56)
    print("  SWE Infinite — Pipeline Statistics")
    print("=" * 56)
    print(f"  GH Archive hours ingested:  {stats.get('hours_ingested', 0):>8,}")
    print(f"  Events in database:         {stats.get('events', 0):>8,}")
    if stats.get("events_by_type"):
        for t, c in stats["events_by_type"].items():
            print(f"    {t + ':':30s}{c:>8,}")
    print(f"  Merged PRs (enriched/total):{prs_enriched:>5,} / {prs_total:,}")
    print(f"  Unique repos seen:          {stats.get('unique_repos', 0):>8,}")
    print(f"  Tasks in database:          {stats.get('tasks', 0):>8,}")
    print(f"  Dataset files on disk:      {dataset_count:>8,}")
    print(f"  Dataset directory:          {DATASET_DIR}")
    if stats.get("validations_by_status"):
        print(f"  Validation results:")
        for s, c in stats["validations_by_status"].items():
            print(f"    {s + ':':30s}{c:>8,}")
    if stats.get("repos_quality_checked"):
        print(f"  Repos quality-checked:      {stats['repos_quality_checked']:>8,}")
        print(f"  Repos passing quality:      {stats['repos_passing_quality']:>8,}")
    print("=" * 56)
    print()


# ---------------------------------------------------------------------------
# Disk cleanup — prevent unbounded repo accumulation
# ---------------------------------------------------------------------------


def cleanup_old_repos(max_age_days: int = 7, max_total_gb: float = 50.0) -> None:
    """Remove old cached repo clones to prevent disk exhaustion.

    Deletes repos older than *max_age_days* (by last-access time).
    If total size still exceeds *max_total_gb*, deletes oldest repos
    until under the limit.
    """
    if not REPOS_DIR.exists():
        return

    repo_dirs = [d for d in REPOS_DIR.iterdir() if d.is_dir()]
    if not repo_dirs:
        return

    now = time.time()
    max_age_secs = max_age_days * 86400

    # Phase 1: delete repos older than max_age_days
    deleted_age = 0
    for d in repo_dirs:
        try:
            last_access = d.stat().st_atime
        except OSError:
            continue
        if now - last_access > max_age_secs:
            log.info("Cleanup: removing old repo %s (%.1f days idle)", d.name, (now - last_access) / 86400)
            try:
                shutil.rmtree(d)
                deleted_age += 1
            except OSError:
                log.warning("Cleanup: failed to remove %s", d.name)

    # Phase 2: if still over size limit, remove oldest first
    remaining = [d for d in REPOS_DIR.iterdir() if d.is_dir()]
    if not remaining:
        log.info("Cleanup: removed %d old repos. Repos dir now empty.", deleted_age)
        return

    # Quick size estimate using du-like walk
    def _dir_size_bytes(path: Path) -> int:
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except OSError:
            pass
        return total

    total_bytes = sum(_dir_size_bytes(d) for d in remaining)
    max_bytes = max_total_gb * 1024**3
    deleted_size = 0

    if total_bytes > max_bytes:
        # Sort by last access time (oldest first)
        remaining.sort(key=lambda d: d.stat().st_atime)
        for d in remaining:
            if total_bytes <= max_bytes:
                break
            size = _dir_size_bytes(d)
            log.info("Cleanup: removing repo %s (%.1f MB) to reclaim disk", d.name, size / 1e6)
            try:
                shutil.rmtree(d)
                total_bytes -= size
                deleted_size += 1
            except OSError:
                log.warning("Cleanup: failed to remove %s", d.name)

    total_deleted = deleted_age + deleted_size
    if total_deleted > 0:
        log.info("Cleanup: removed %d repos total.", total_deleted)


# ---------------------------------------------------------------------------
# Heartbeat — periodic liveness / stats logging
# ---------------------------------------------------------------------------


def _log_heartbeat(
    start_time: float, tasks: int, skips: int, errors: int, db_path: Path,
) -> None:
    """Log a heartbeat line with pipeline stats."""
    uptime_secs = time.monotonic() - start_time
    hours = int(uptime_secs // 3600)
    minutes = int((uptime_secs % 3600) // 60)

    # Count cached repos
    repo_count = 0
    if REPOS_DIR.exists():
        repo_count = sum(1 for d in REPOS_DIR.iterdir() if d.is_dir())

    log.info(
        "[HEARTBEAT] uptime=%dh%02dm, tasks=%d, skips=%d, errors=%d, repos_cached=%d",
        hours, minutes, tasks, skips, errors, repo_count,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE Infinite — Linear SWE task collection pipeline",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--once", action="store_true", help="Produce one task then exit")
    parser.add_argument("--status", action="store_true", help="Print stats and exit")
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip execution validation (FAIL_TO_PASS / PASS_TO_PASS will be empty)",
    )
    parser.add_argument(
        "--skip-quality", action="store_true",
        help="Skip LLM quality scoring",
    )
    parser.add_argument(
        "--docker", action="store_true",
        help="Use Docker for isolated validation environments",
    )
    parser.add_argument(
        "--min-stars", type=int, default=5,
        help="Minimum GitHub stars for repo quality gate (default: 5)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Number of parallel workers for task processing (default: 1)",
    )
    args = parser.parse_args()

    _setup_logging()
    init_db(args.db)

    if args.status:
        print_stats(args.db)
        return

    global _shutdown  # noqa: PLW0603
    log.info("Starting SWE Infinite pipeline (pid=%d, workers=%d)", os.getpid(), args.parallel)

    # Seed the DB with some data
    ensure_events(args.db)

    tasks_produced = 0
    skips = 0
    errors = 0
    consecutive_empty = 0
    consecutive_errors = 0
    start_time = time.monotonic()
    last_heartbeat = time.monotonic()
    last_cleanup = time.monotonic()

    def _process_one_safe() -> str:
        """Wrapper for process_one with exception handling."""
        return process_one(
            args.db,
            skip_validation=args.skip_validation,
            skip_quality=args.skip_quality,
            use_docker=args.docker,
            min_stars=args.min_stars,
        )

    # --- Parallel mode ---
    if args.parallel > 1:
        log.info("Running in parallel mode with %d workers", args.parallel)
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            while not _shutdown:
                # Submit a batch of tasks
                futures = []
                for _ in range(args.parallel):
                    if _shutdown:
                        break
                    futures.append(executor.submit(_process_one_safe))

                for future in as_completed(futures):
                    if _shutdown:
                        break
                    try:
                        status = future.result(timeout=600)
                    except Exception:
                        log.exception("Unexpected error in parallel worker")
                        errors += 1
                        continue

                    if status == "task":
                        tasks_produced += 1
                        log.info("--- Tasks produced so far: %d ---", tasks_produced)
                        if args.once:
                            _shutdown = True
                    elif status == "error":
                        errors += 1
                    elif status == "empty":
                        consecutive_empty += 1

                # Handle empty state
                if consecutive_empty >= args.parallel * 2:
                    try:
                        got_new = download_next_hour(args.db)
                    except Exception:
                        got_new = False
                    if not got_new:
                        log.info("No new data. Waiting 5 minutes...")
                        for _ in range(300):
                            if _shutdown:
                                break
                            time.sleep(1)
                    consecutive_empty = 0

                time.sleep(0.5)
    else:
        # --- Sequential mode (original) ---
        while not _shutdown:
            # --- Periodic heartbeat (every 30 min) ---
            now_mono = time.monotonic()
            if now_mono - last_heartbeat >= 1800:
                _log_heartbeat(start_time, tasks_produced, skips, errors, args.db)
                last_heartbeat = now_mono

            # --- Periodic disk cleanup (every hour) ---
            if now_mono - last_cleanup >= 3600:
                try:
                    cleanup_old_repos()
                except Exception:
                    log.exception("Error during repo cleanup")
                last_cleanup = now_mono

            # --- Process one PR (crash-proof) ---
            try:
                status = _process_one_safe()
            except Exception:
                log.exception("Unexpected error in process_one")
                consecutive_errors += 1
                errors += 1
                backoff = min(2 ** consecutive_errors, 300)
                log.info("Backing off %ds after %d consecutive errors", backoff, consecutive_errors)
                for _ in range(int(backoff)):
                    if _shutdown:
                        break
                    time.sleep(1)
                continue
            else:
                consecutive_errors = 0

            if status == "task":
                tasks_produced += 1
                skips = 0
                consecutive_empty = 0
                log.info("--- Tasks produced so far: %d ---", tasks_produced)

                if args.once:
                    break

            elif status == "skip" or status == "error":
                skips += 1
                if status == "error":
                    errors += 1
                consecutive_empty = 0

            elif status == "empty":
                consecutive_empty += 1
                # Try downloading more data
                try:
                    got_new = download_next_hour(args.db)
                except Exception:
                    log.exception("Error downloading next GH Archive hour")
                    got_new = False
                if not got_new:
                    if consecutive_empty >= 3:
                        log.info("No new data available. Waiting 5 minutes...")
                        for _ in range(300):
                            if _shutdown:
                                break
                            time.sleep(1)
                        consecutive_empty = 0

            # Brief pause between iterations
            time.sleep(0.1)

    # Final heartbeat on shutdown
    _log_heartbeat(start_time, tasks_produced, skips, errors, args.db)
    print_stats(args.db)
    log.info("Done. %d tasks produced, %d PRs skipped, %d errors.", tasks_produced, skips, errors)


if __name__ == "__main__":
    main()
