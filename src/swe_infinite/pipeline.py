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
    uv run python swe_infinite.py --allow-non-permissive  # include non-permissive licenses
    uv run python swe_infinite.py --languages python,go   # only these languages
    uv run python swe_infinite.py --max-patch-files 30    # allow up to 30 files in patch
    uv run python swe_infinite.py --skip-decontamination  # skip overlap check
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .db import (
    DEFAULT_DB_PATH,
    get_annotation_stats,
    get_connection,
    get_extractable_annotations,
    get_stats,
    init_db,
    insert_validation_result,
    is_hour_ingested,
    mark_hour_ingested,
    update_annotation_extraction,
    upsert_annotation,
)
from .decontamination import check_contamination
from .gharchive import download_and_parse_hour
from .language_support import get_handler
from .quality_scorer import assess_quality, heuristic_issue_prefilter
from .recipe_generator import generate_recipe
from .repo_ops import extract_candidate
from .repo_quality import check_repo_quality
from .task_store import build_task_instance, make_instance_id, store_task, update_task_eval_result
from .validator import validate_task
from .versioning import find_version_for_commit
from .paths import DATASET_DIR, REPOS_DIR, LOG_FILE

# Supported languages (extensible — see language_support.py for handlers)
SUPPORTED_LANGUAGES = {"python", "typescript", "javascript", "java", "go"}


# ---------------------------------------------------------------------------
# Filter configuration — every filter knob, settable from CLI
# ---------------------------------------------------------------------------


@dataclass
class FilterConfig:
    """All tuneable filter settings for the pipeline.

    Each field corresponds to a CLI flag so operators can relax or tighten
    any filter without touching code.
    """

    # --- repo filters ---
    min_stars: int = 5
    min_contributors: int = 0
    require_ci: bool = False
    require_tests: bool = False
    allow_archived: bool = True
    allow_non_permissive: bool = True
    languages: set[str] = field(default_factory=lambda: set(SUPPORTED_LANGUAGES))

    # --- issue / PR filters ---
    min_issue_length: int = 10
    allow_no_issue: bool = True

    # --- patch filters ---
    max_patch_files: int = 15
    min_patch_lines: int = 3
    max_patch_lines: int = 1000
    skip_patch_checks: bool = False

    # --- quality filters ---
    min_quality_score: int = 2
    skip_quality: bool = True
    skip_decontamination: bool = True

    # --- test generation ---
    generate_tests: bool = True

    # --- validation ---
    skip_validation: bool = True
    use_docker: bool = False

    # --- evaluation (agent solving) ---
    run_eval: bool = False
    eval_model: str = "opus-4.6"
    eval_timeout: int = 300

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


def _fetch_issue_from_api(repo_name: str, issue_num: int) -> tuple[str | None, str | None]:
    """Fetch an issue's title and body from the GitHub API.

    Returns (title, body) on success, or (None, None) on failure.
    """
    client = _get_gh_client()
    url = f"{_GITHUB_API}/repos/{repo_name}/issues/{issue_num}"
    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        log.debug("  API error fetching issue %s#%d: %s", repo_name, issue_num, exc)
        return None, None

    if resp.status_code != 200:
        log.debug("  Got %d fetching issue %s#%d", resp.status_code, repo_name, issue_num)
        return None, None

    data = resp.json()
    # Make sure it's actually an issue, not a pull request
    if data.get("pull_request"):
        log.debug("  #%d is a pull request, not an issue", issue_num)
        return None, None

    title = data.get("title", "")
    body = (data.get("body") or "")[:50000]
    return title, body


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

# Strong signal: "fixes #N", "closes #N", "resolves #N", etc.
_ISSUE_RE_STRONG = re.compile(
    r"(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+"
    r"(?:https?://github\.com/[\w\-\.]+/[\w\-\.]+/issues/(\d+)|#(\d+))",
    re.IGNORECASE,
)

# Medium signal: "addresses #N", "refs #N", "related to #N", "for #N", "re: #N"
_ISSUE_RE_MEDIUM = re.compile(
    r"(?:address(?:es|ed)?|refs?|related\s+to|for|re:?|see)\s+"
    r"(?:https?://github\.com/[\w\-\.]+/[\w\-\.]+/issues/(\d+)|#(\d+))",
    re.IGNORECASE,
)

# Weak signal: full GitHub issue URL (explicit /issues/ path, no keyword needed)
_ISSUE_RE_URL = re.compile(
    r"https?://github\.com/[\w\-\.]+/[\w\-\.]+/issues/(\d+)",
    re.IGNORECASE,
)


def _find_linked_issue(title: str, body: str) -> int | None:
    """Find a linked issue number from PR title+body. Returns None if 0 found.

    When multiple issues are linked at the same priority level, returns the
    lowest-numbered one (typically the primary issue).

    Checks in priority order: strong keywords (fix/close/resolve), medium keywords
    (addresses/refs/related), then bare GitHub issue URLs.
    """
    for regex in (_ISSUE_RE_STRONG, _ISSUE_RE_MEDIUM, _ISSUE_RE_URL):
        issues: set[int] = set()
        for text in (title, body):
            for m in regex.finditer(text or ""):
                # URL-only regex has 1 group; keyword regexes have 2
                if regex.groups == 1:
                    num = m.group(1)
                else:
                    num = m.group(1) or m.group(2)
                if num:
                    issues.add(int(num))
        if issues:
            return min(issues)  # pick lowest-numbered (primary) issue
    return None


# ---------------------------------------------------------------------------
# Phase A: Annotate — enrich one PR, detect metadata, store everything
# ---------------------------------------------------------------------------


def annotate_one(conn: sqlite3.Connection, *, allow_no_issue: bool = False) -> str:
    """Pick one un-enriched merged PR, enrich it, and store all metadata.

    This is FAST — only API calls, no git cloning.
    Every enriched PR gets a row in pr_annotations regardless of viability.

    Args:
        allow_no_issue: When True, PRs without a linked issue can still be
            marked as quality_ok if their PR body passes heuristic checks.
            A problem statement will be generated from the PR diff later.

    Returns: "annotated", "skip" (API error), or "empty" (no PRs left).
    """
    # Pick the next un-enriched merged PR
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

    log.info("Annotating: %s PR#%d", repo_name, pr_number)

    # --- Enrich via GitHub API ---
    enriched = _enrich_pr(conn, event_id, repo_name, pr_number)
    if enriched is None:
        return "skip"  # API error — already marked in events table

    # --- Detect language ---
    lang = (enriched.get("language") or "").lower()
    lang_map = {
        "javascript": "javascript", "typescript": "typescript", "python": "python",
        "java": "java", "go": "go", "golang": "go",
    }
    lang = lang_map.get(lang, lang)

    # --- Find linked issue ---
    issue_num = _find_linked_issue(enriched["title"], enriched["body"])

    issue_title = None
    issue_body = ""
    skip_reason = None

    if issue_num is not None:
        # Look up issue (DB first, then GitHub API)
        issue_row = conn.execute(
            "SELECT title, body FROM events WHERE repo_name=? AND number=? AND type='IssuesEvent'",
            (repo_name, issue_num),
        ).fetchone()

        if issue_row is not None:
            issue_title = issue_row["title"] or ""
            issue_body = issue_row["body"] or ""
        else:
            log.info("  Issue #%d not in DB, fetching from GitHub API...", issue_num)
            issue_title, issue_body = _fetch_issue_from_api(repo_name, issue_num)
            if issue_title is None:
                issue_title = None
                issue_body = ""
                skip_reason = f"could not fetch issue #{issue_num} from API"

    # --- Heuristic issue quality check ---
    issue_quality_ok = 0
    issue_quality_reason = None
    if issue_num is not None and issue_title is not None:
        rejection = heuristic_issue_prefilter(issue_title, issue_body)
        if rejection:
            issue_quality_reason = rejection
        else:
            issue_quality_ok = 1
    elif allow_no_issue and issue_num is None:
        # No linked issue — check if PR body itself is usable as a problem source.
        # The actual problem statement will be generated via LLM during extraction.
        pr_body_text = enriched.get("body") or ""
        pr_title_text = enriched.get("title") or ""
        if len(pr_body_text.strip()) > 20 or len(pr_title_text.strip()) > 10:
            issue_quality_ok = 1
            issue_quality_reason = "no-issue: PR body used as problem source"
        else:
            issue_quality_reason = "no-issue: PR body too short for problem generation"

    # --- Store annotation (NOTHING is filtered out) ---
    ann = {
        "event_id": event_id,
        "repo_name": repo_name,
        "pr_number": pr_number,
        "language": lang or None,
        "issue_num": issue_num,
        "issue_title": issue_title,
        "issue_body": issue_body[:50000] if issue_body else None,
        "issue_body_len": len((issue_body or "").strip()),
        "issue_quality_ok": issue_quality_ok,
        "issue_quality_reason": issue_quality_reason,
        "pr_title": enriched["title"],
        "pr_body": enriched["body"][:10000] if enriched["body"] else None,
        "base_sha": row["base_sha"],
        "head_sha": row["head_sha"],
        "merge_commit_sha": enriched.get("merge_commit_sha") or row["merge_commit_sha"],
        "status": "annotated",
        "skip_reason": skip_reason,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
    upsert_annotation(conn, ann)

    # Mark event as processed (so we don't re-enrich)
    conn.execute("UPDATE events SET processed = 1 WHERE id = ?", (event_id,))
    conn.commit()

    log.info("  Annotated: lang=%s, issue=%s, quality_ok=%d",
             lang or "?", f"#{issue_num}" if issue_num else "none", issue_quality_ok)
    return "annotated"


# ---------------------------------------------------------------------------
# Phase B: Extract — pick best annotated candidate, clone, diff, validate
# ---------------------------------------------------------------------------


def extract_one(
    db_path: Path,
    filters: FilterConfig,
) -> str:
    """Pick the best annotated PR, clone repo, compute diff, validate, store.

    This is EXPENSIVE — involves git cloning, diff computation, and validation.
    Only runs on PRs that passed annotation checks.

    After processing (success or failure), the cloned repo is removed to save
    disk space.

    Returns: "task", "skip", "empty", or "error".
    """
    conn = get_connection(db_path)
    repo_dir: Path | None = None  # track for cleanup

    try:
        # --- Pick best candidate from annotations ---
        candidates = get_extractable_annotations(
            conn,
            languages=filters.languages,
            min_issue_length=filters.min_issue_length,
            limit=1,
            allow_no_issue=filters.allow_no_issue,
        )

        if not candidates:
            return "empty"

        ann = candidates[0]
        repo_name = ann["repo_name"]
        pr_number = ann["pr_number"]
        event_id = ann["event_id"]

        log.info("Extracting: %s PR#%d (issue #%s)", repo_name, pr_number, ann["issue_num"])

        # Mark as extracting
        conn.execute(
            "UPDATE pr_annotations SET status = 'extracting' WHERE event_id = ?",
            (event_id,),
        )
        conn.commit()

        # --- Clone repo, compute diff, split patches ---
        # Pre-compute expected repo dir for cleanup even if extraction fails
        repo_dir = REPOS_DIR / repo_name.replace("/", "__")

        candidate = {
            "repo_name": repo_name,
            "pr_number": pr_number,
            "issue_number": ann["issue_num"],
            "issue_title": ann["issue_title"] or "",
            "issue_body": ann["issue_body"] or "",
            "pr_title": ann["pr_title"] or "",
            "base_sha": ann["base_sha"],
            "head_sha": ann["head_sha"],
            "merge_commit_sha": ann["merge_commit_sha"],
            "id": None,
        }

        try:
            extracted = extract_candidate(
                candidate,
                allow_non_permissive=filters.allow_non_permissive,
                max_patch_files=filters.max_patch_files,
                min_patch_lines=filters.min_patch_lines,
                max_patch_lines=filters.max_patch_lines,
                generate_tests=filters.generate_tests,
                skip_patch_checks=filters.skip_patch_checks,
            )
        except Exception as exc:
            log.exception("  Error extracting %s PR#%d", repo_name, pr_number)
            update_annotation_extraction(conn, event_id, {
                "has_test_changes": None,
                "has_solution_changes": None,
                "solution_files": None,
                "solution_lines": None,
                "license_name": None,
                "license_ok": None,
                "repo_stars": None,
                "status": "failed",
                "skip_reason": f"extraction error: {exc!s:.100}",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            })
            return "error"

        if extracted is None:
            # Store what we know even though extraction failed
            update_annotation_extraction(conn, event_id, {
                "has_test_changes": 0,
                "has_solution_changes": 0,
                "solution_files": 0,
                "solution_lines": 0,
                "license_name": None,
                "license_ok": None,
                "repo_stars": None,
                "status": "failed",
                "skip_reason": "extraction returned None (license/patch/files/complexity)",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            })
            return "skip"

        repo_dir = Path(extracted["repo_dir"])

        # --- Update annotation with extraction results ---
        from .repo_ops import count_changed_lines
        added, removed = count_changed_lines(extracted.get("solution_patch", ""))
        update_annotation_extraction(conn, event_id, {
            "has_test_changes": 1 if extracted.get("test_patch", "").strip() else 0,
            "has_solution_changes": 1 if extracted.get("solution_patch", "").strip() else 0,
            "solution_files": extracted.get("num_modified_files", 0),
            "solution_lines": added + removed,
            "license_name": extracted.get("license_name"),
            "license_ok": 1,  # if we got here, license was OK
            "repo_stars": None,  # filled by repo quality check
            "status": "extracted",
            "skip_reason": None,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        })

        # --- Repo quality gate ---
        try:
            quality = check_repo_quality(
                repo_name,
                repo_dir=repo_dir,
                min_stars=filters.min_stars,
                min_contributors=filters.min_contributors,
                require_ci=filters.require_ci,
                require_tests=filters.require_tests,
                allow_archived=filters.allow_archived,
                db_path=db_path,
            )
            # Store stars even if quality check fails
            conn.execute(
                "UPDATE pr_annotations SET repo_stars = ? WHERE event_id = ?",
                (quality.stars if hasattr(quality, 'stars') else None, event_id),
            )
            conn.commit()

            if not quality.passes:
                log.info("  Skip: repo quality — %s", quality.reason)
                conn.execute(
                    "UPDATE pr_annotations SET status = 'failed', skip_reason = ? WHERE event_id = ?",
                    (f"repo quality: {quality.reason}", event_id),
                )
                conn.commit()
                return "skip"
        except Exception:
            log.debug("  Repo quality check failed, continuing anyway")

        lang = ann["language"] or "python"

        # --- Version, recipe, validation ---
        version = find_version_for_commit(repo_dir, extracted["base_commit"]) or "unknown"

        install_config = None
        try:
            install_config = generate_recipe(repo_dir, repo_name, max_retries=2)
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

        # --- Generate problem statement if no issue body ---
        issue_body = (extracted.get("issue_body") or "").strip()
        if filters.allow_no_issue and len(issue_body) < 20:
            from .quality_scorer import generate_problem_statement
            generated = generate_problem_statement(
                pr_title=extracted.get("pr_title") or ann.get("pr_title") or "",
                pr_body=ann.get("pr_body") or "",
                solution_patch=extracted.get("solution_patch") or "",
            )
            if generated:
                extracted["issue_title"] = generated.split("\n")[0][:200]
                extracted["issue_body"] = generated
                log.info("  Generated problem statement (%d chars) from PR diff", len(generated))

        task = build_task_instance(
            extracted, version=version, install_config=install_config,
            environment_setup_commit=extracted["base_commit"],
        )

        # --- Validation ---
        try:
            validation_result = _phase_validate(
                conn, task, filters.use_docker, filters.skip_validation,
            )
        except _SkipTask:
            conn.execute(
                "UPDATE pr_annotations SET status = 'failed', skip_reason = 'validation_failed' WHERE event_id = ?",
                (event_id,),
            )
            conn.commit()
            return "skip"

        # --- Quality assessment ---
        try:
            quality_scores = _phase_quality(task, repo_name, pr_number, filters)
        except _SkipTask:
            conn.execute(
                "UPDATE pr_annotations SET status = 'failed', skip_reason = 'quality_failed' WHERE event_id = ?",
                (event_id,),
            )
            conn.commit()
            return "skip"

        # --- Store ---
        result, task = _phase_store(
            extracted, version, install_config,
            validation_result, quality_scores, db_path,
        )

        # --- Evaluate (optional) ---
        _phase_eval(task, filters)

        # Mark annotation as task_created
        conn.execute(
            "UPDATE pr_annotations SET status = 'task_created' WHERE event_id = ?",
            (event_id,),
        )
        conn.commit()

        return result

    finally:
        # Clean up cloned repo to save disk space
        if repo_dir is not None and repo_dir.exists():
            log.info("Cleaning up repo: %s", repo_dir.name)
            try:
                shutil.rmtree(repo_dir)
            except OSError:
                log.warning("Failed to remove repo dir: %s", repo_dir)
        conn.close()


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
    task: dict, repo_name: str, pr_number: int, filters: FilterConfig,
) -> object | None:
    """Phase 4: LLM quality scoring and decontamination check.

    Returns QualityScores on success (or None if skipped/failed).
    Raises _SkipTask if the task is rejected or contaminated.
    """
    quality_scores = None
    if not filters.skip_quality:
        try:
            quality_scores = assess_quality(
                task, min_quality_score=filters.min_quality_score,
            )

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
    if not filters.skip_decontamination:
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
) -> tuple[str, dict]:
    """Phase 5: Rebuild the final task with all enrichment data and save it.

    Returns ("task", task_dict) so downstream phases can use the stored task.
    """
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

    # Upload to Hippius decentralised storage (fire-and-forget)
    try:
        from swe_infinite.hippius import upload_task as hippius_upload
        if hippius_upload(path):
            log.info("  Uploaded to Hippius: %s", task["instance_id"])
        else:
            log.debug("  Hippius upload skipped (no credentials or not configured)")
    except Exception:
        log.warning("  Hippius upload failed (non-fatal)", exc_info=True)

    return "task", task


def _phase_eval(task: dict, filters: FilterConfig) -> None:
    """Phase 6: Run agent evaluation on the generated task.

    Invokes the Cursor agent CLI to attempt solving the task, records
    whether it succeeded, and writes the eval_result back into the
    task JSON file on disk.

    This phase is optional and only runs when ``filters.run_eval`` is True.
    Errors are logged but never abort the pipeline.
    """
    if not filters.run_eval:
        return

    instance_id = task["instance_id"]
    log.info("  Running agent evaluation on %s (model=%s, timeout=%ds)...",
             instance_id, filters.eval_model, filters.eval_timeout)

    try:
        from .eval import evaluate_new_task

        eval_result = evaluate_new_task(
            task,
            model=filters.eval_model,
            agent_timeout=filters.eval_timeout,
            cleanup=True,
        )

        # Write eval_result back into the task JSON on disk
        update_task_eval_result(eval_result, instance_id, DATASET_DIR)

        status = eval_result["status"]
        elapsed = eval_result.get("agent_time_seconds", 0.0)
        tokens = eval_result.get("token_count")
        tools = eval_result.get("tool_calls")

        status_str = status.upper()
        metrics = f"time={elapsed:.1f}s"
        if tokens is not None:
            metrics += f", tokens={tokens}"
        if tools is not None:
            metrics += f", tool_calls={tools}"

        log.info("  EVAL RESULT: %s (%s)", status_str, metrics)

    except Exception:
        log.exception("  Evaluation failed for %s (non-fatal)", instance_id)


class _SkipTask(Exception):
    """Raised by phase functions to signal that the current task should be skipped."""


def process_one(
    db_path: Path,
    filters: FilterConfig | None = None,
) -> str:
    """Two-phase processing: annotate first, then extract.

    Phase A: Annotate — enrich a PR and store metadata (fast, API-only).
    Phase B: Extract — pick best annotated candidate and do expensive work.

    Returns: "task", "skip", "empty", "annotated", or "error".
    """
    if filters is None:
        filters = FilterConfig()

    conn = get_connection(db_path)
    try:
        # Phase A: annotate next un-enriched PR (fast)
        status = annotate_one(conn, allow_no_issue=filters.allow_no_issue)
        if status == "annotated":
            return "annotated"
        # If "empty" or "skip", fall through to try extraction
    finally:
        conn.close()

    # Phase B: try to extract from annotated pool (expensive)
    return extract_one(db_path, filters)


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
        ann_stats = get_annotation_stats(conn)
        conn.close()
    except Exception:
        log.warning("Database not initialized yet.")
        return

    dataset_count = len(list(DATASET_DIR.glob("*.json"))) if DATASET_DIR.exists() else 0
    prs_total = stats.get("prs_merged_total", 0)
    prs_enriched = stats.get("prs_enriched", 0)

    print()
    print("=" * 64)
    print("  SWE Infinite — Pipeline Statistics")
    print("=" * 64)
    print(f"  GH Archive hours ingested:     {stats.get('hours_ingested', 0):>8,}")
    print(f"  Events in database:            {stats.get('events', 0):>8,}")
    if stats.get("events_by_type"):
        for t, c in stats["events_by_type"].items():
            print(f"    {t + ':':30s}  {c:>8,}")
    print(f"  Merged PRs (enriched/total):   {prs_enriched:>5,} / {prs_total:,}")
    print(f"  Unique repos seen:             {stats.get('unique_repos', 0):>8,}")

    # --- Annotation breakdown ---
    if ann_stats.get("total", 0) > 0:
        print()
        print(f"  PR Annotations:                {ann_stats['total']:>8,}")
        print(f"    With linked issue:           {ann_stats.get('with_issue', 0):>8,}")
        print(f"    Extraction-ready:            {ann_stats.get('extraction_ready', 0):>8,}")
        if ann_stats.get("by_language"):
            print(f"    By language:")
            for lang, cnt in sorted(ann_stats["by_language"].items(), key=lambda x: -x[1]):
                print(f"      {lang + ':':28s}  {cnt:>8,}")
        if ann_stats.get("by_status"):
            print(f"    By status:")
            for st, cnt in sorted(ann_stats["by_status"].items(), key=lambda x: -x[1]):
                print(f"      {st + ':':28s}  {cnt:>8,}")

    print()
    print(f"  Tasks in database:             {stats.get('tasks', 0):>8,}")
    print(f"  Dataset files on disk:         {dataset_count:>8,}")
    print(f"  Dataset directory:             {DATASET_DIR}")
    if stats.get("validations_by_status"):
        print(f"  Validation results:")
        for s, c in stats["validations_by_status"].items():
            print(f"    {s + ':':30s}  {c:>8,}")
    if stats.get("repos_quality_checked"):
        print(f"  Repos quality-checked:         {stats['repos_quality_checked']:>8,}")
        print(f"  Repos passing quality:         {stats['repos_passing_quality']:>8,}")
    print("=" * 64)
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

    # --- Pipeline phase toggles ---
    phase_group = parser.add_argument_group("pipeline phases")
    phase_group.add_argument(
        "--validate", action="store_true",
        help="Enable execution validation (populates FAIL_TO_PASS / PASS_TO_PASS)",
    )
    phase_group.add_argument(
        "--quality", action="store_true",
        help="Enable LLM quality scoring",
    )
    phase_group.add_argument(
        "--decontamination", action="store_true",
        help="Enable overlap check against known benchmarks (SWE-bench, etc.)",
    )
    phase_group.add_argument(
        "--docker", action="store_true",
        help="Use Docker for isolated validation environments",
    )
    phase_group.add_argument(
        "--no-generate-tests", action="store_true",
        help="Disable LLM test generation for PRs without test changes",
    )
    phase_group.add_argument(
        "--eval", action="store_true",
        help="Run Cursor agent evaluation on each generated task",
    )
    phase_group.add_argument(
        "--eval-model", type=str, default="opus-4.6",
        help="Model for agent evaluation (default: opus-4.6)",
    )
    phase_group.add_argument(
        "--eval-timeout", type=int, default=800,
        help="Agent timeout in seconds for evaluation (default: 800)",
    )

    # --- Repo filters ---
    repo_group = parser.add_argument_group("repo filters")
    repo_group.add_argument(
        "--min-stars", type=int, default=0,
        help="Minimum GitHub stars for repo quality gate (default: 5)",
    )
    repo_group.add_argument(
        "--min-contributors", type=int, default=0,
        help="Minimum number of contributors (default: 0)",
    )
    repo_group.add_argument(
        "--require-ci", action="store_true",
        help="Require CI/CD configuration (default: not required)",
    )
    repo_group.add_argument(
        "--require-tests", action="store_true",
        help="Require a test framework to be detected in repo",
    )
    repo_group.add_argument(
        "--no-allow-archived", action="store_true",
        help="Skip archived repositories",
    )
    repo_group.add_argument(
        "--no-allow-non-permissive", action="store_true",
        help="Reject repos with non-permissive licenses (default: allow all licenses)",
    )
    repo_group.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated list of languages to accept (default: python,typescript,javascript,java,go)",
    )

    # --- Issue / PR filters ---
    issue_group = parser.add_argument_group("issue filters")
    issue_group.add_argument(
        "--min-issue-length", type=int, default=0,
        help="Minimum issue body length in characters (default: 10)",
    )
    issue_group.add_argument(
        "--no-allow-no-issue", action="store_true",
        help="Reject PRs with no linked issue (default: accept them and generate problem statement via LLM)",
    )

    # --- Patch filters ---
    patch_group = parser.add_argument_group("patch filters")
    patch_group.add_argument(
        "--max-patch-files", type=int, default=15,
        help="Maximum number of files in solution patch (default: 15)",
    )
    patch_group.add_argument(
        "--min-patch-lines", type=int, default=3,
        help="Minimum changed lines in solution patch (default: 3)",
    )
    patch_group.add_argument(
        "--max-patch-lines", type=int, default=2000,
        help="Maximum changed lines in solution patch (default: 1000)",
    )
    patch_group.add_argument(
        "--no-patch-checks", action="store_true",
        help="Disable patch complexity checks (file count, line count, config-only, comment-only)",
    )

    # --- Quality filters ---
    quality_group = parser.add_argument_group("quality filters")
    quality_group.add_argument(
        "--min-quality-score", type=int, default=2,
        help="Minimum quality score (1-5) for issue_text and test dimensions (default: 2)",
    )

    args = parser.parse_args()

    _setup_logging()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    init_db(args.db)

    if args.status:
        print_stats(args.db)
        return

    # Build FilterConfig from CLI args
    languages = set(SUPPORTED_LANGUAGES)
    if args.languages:
        languages = {l.strip().lower() for l in args.languages.split(",")}

    filter_config = FilterConfig(
        min_stars=args.min_stars,
        min_contributors=args.min_contributors,
        require_ci=args.require_ci,
        require_tests=args.require_tests,
        allow_archived=not args.no_allow_archived,
        allow_non_permissive=not args.no_allow_non_permissive,
        languages=languages,
        min_issue_length=args.min_issue_length,
        allow_no_issue=not args.no_allow_no_issue,
        max_patch_files=args.max_patch_files,
        min_patch_lines=args.min_patch_lines,
        max_patch_lines=args.max_patch_lines,
        skip_patch_checks=args.no_patch_checks,
        min_quality_score=args.min_quality_score,
        generate_tests=not args.no_generate_tests,
        skip_quality=not args.quality,
        skip_decontamination=not args.decontamination,
        skip_validation=not args.validate,
        use_docker=args.docker,
        run_eval=args.eval,
        eval_model=args.eval_model,
        eval_timeout=args.eval_timeout,
    )

    global _shutdown  # noqa: PLW0603
    log.info("Starting SWE Infinite pipeline (pid=%d)", os.getpid())
    log.info("Filter config: %s", filter_config)

    # Seed the DB with some data
    ensure_events(args.db)

    tasks_produced = 0
    annotated_count = 0
    extraction_attempts = 0
    errors = 0
    consecutive_errors = 0
    start_time = time.monotonic()
    last_heartbeat = time.monotonic()
    last_cleanup = time.monotonic()

    # The main loop alternates between two modes:
    # 1. ANNOTATE: enrich PRs rapidly (API-only, no cloning)
    # 2. EXTRACT: pick best annotated candidate and do expensive work

    # When allow_no_issue is on, most PRs are immediately viable so we
    # annotate just one before trying extraction.  With the strict filter
    # we still batch to build up a pool of candidates.
    ANNOTATION_BATCH = 1 if filter_config.allow_no_issue else 20

    while not _shutdown:
        # --- Periodic heartbeat (every 10 min) ---
        now_mono = time.monotonic()
        if now_mono - last_heartbeat >= 600:
            _log_heartbeat(start_time, tasks_produced, annotated_count, errors, args.db)
            last_heartbeat = now_mono

        # --- Periodic disk cleanup (every hour) ---
        if now_mono - last_cleanup >= 3600:
            try:
                cleanup_old_repos()
            except Exception:
                log.exception("Error during repo cleanup")
            last_cleanup = now_mono

        # ============================
        # PHASE A: Annotate a batch
        # ============================
        annotation_empty = False
        conn = get_connection(args.db)
        try:
            for _ in range(ANNOTATION_BATCH):
                if _shutdown:
                    break
                try:
                    status = annotate_one(conn, allow_no_issue=filter_config.allow_no_issue)
                except Exception:
                    log.exception("Error during annotation")
                    errors += 1
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        backoff = min(2 ** (consecutive_errors - 10), 60)
                        log.info("Backing off %ds after %d consecutive errors", backoff, consecutive_errors)
                        time.sleep(backoff)
                    continue
                else:
                    consecutive_errors = 0

                if status == "annotated":
                    annotated_count += 1
                elif status == "empty":
                    annotation_empty = True
                    break
                # "skip" means API error — just continue to next
        finally:
            conn.close()

        if annotated_count > 0 and annotated_count % 100 == 0:
            log.info("--- Annotated %d PRs so far ---", annotated_count)

        # ============================
        # PHASE B: Try extraction
        # ============================
        try:
            status = extract_one(args.db, filter_config)
        except Exception:
            log.exception("Error during extraction")
            errors += 1
            status = "error"

        extraction_attempts += 1

        if status == "task":
            tasks_produced += 1
            log.info("=== TASK %d PRODUCED ===", tasks_produced)
            if args.once:
                break

        elif status == "empty" and annotation_empty:
            # Both annotation and extraction are empty — need more data
            log.info("All PRs annotated, no extractable candidates. Downloading more data...")
            try:
                got_new = download_next_hour(args.db)
            except Exception:
                log.exception("Error downloading next GH Archive hour")
                got_new = False
            if not got_new:
                log.info("No new data available. Waiting 5 minutes...")
                for _ in range(300):
                    if _shutdown:
                        break
                    time.sleep(1)

        # Brief pause between iterations
        time.sleep(0.05)

    # Final summary
    _log_heartbeat(start_time, tasks_produced, annotated_count, errors, args.db)
    print_stats(args.db)
    log.info("Done. %d tasks, %d annotated, %d extraction attempts, %d errors.",
             tasks_produced, annotated_count, extraction_attempts, errors)


if __name__ == "__main__":
    main()
