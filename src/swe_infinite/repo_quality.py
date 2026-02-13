"""
Repository quality gate for the SWE-rebench pipeline.

Filters out low-quality repositories that are unlikely to produce
good benchmark tasks. Checks:
  - Star count (minimum threshold)
  - Contributor count (>1 to avoid personal projects)
  - CI presence (.github/workflows/, .travis.yml, tox.ini, etc.)
  - Test framework usage (pytest/unittest in dependencies)

Results are cached in the SQLite database to avoid redundant API calls.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .db import DEFAULT_DB_PATH, get_connection

log = logging.getLogger("swe-infinite.repo_quality")

_GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Quality result
# ---------------------------------------------------------------------------


@dataclass
class RepoQuality:
    """Quality assessment of a repository."""

    repo_name: str
    stars: int = 0
    forks: int = 0
    contributors: int = 0
    has_ci: bool = False
    has_test_framework: bool = False
    is_archived: bool = False
    is_fork: bool = False
    passes: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "stars": self.stars,
            "forks": self.forks,
            "contributors": self.contributors,
            "has_ci": self.has_ci,
            "has_test_framework": self.has_test_framework,
            "is_archived": self.is_archived,
            "is_fork": self.is_fork,
        }


# ---------------------------------------------------------------------------
# Database cache
# ---------------------------------------------------------------------------
# NOTE: The repo_quality table schema is defined in db.py's _SCHEMA_SQL.
# init_db() must be called before using these cache functions.
# ---------------------------------------------------------------------------


def _get_cached_quality(
    conn: sqlite3.Connection, repo_name: str, max_age_hours: int = 168,
) -> RepoQuality | None:
    """Get cached quality data if fresh enough (default: 7 days)."""
    try:
        row = conn.execute(
            "SELECT * FROM repo_quality WHERE repo_name = ?", (repo_name,)
        ).fetchone()
    except sqlite3.OperationalError:
        # Table doesn't exist yet (init_db hasn't been called)
        return None

    if row is None:
        return None

    # Check freshness (ensure timezone-aware comparison)
    try:
        checked_at = datetime.fromisoformat(row["checked_at"])
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None  # Invalid date — treat as stale
    age_hours = (datetime.now(timezone.utc) - checked_at).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None

    return RepoQuality(
        repo_name=row["repo_name"],
        stars=row["stars"],
        forks=row["forks"],
        contributors=row["contributors"],
        has_ci=bool(row["has_ci"]),
        has_test_framework=bool(row["has_test_framework"]),
        is_archived=bool(row["is_archived"]),
        is_fork=bool(row["is_fork"]),
        passes=bool(row["passes"]),
        reason=row["reason"],
    )


def _cache_quality(conn: sqlite3.Connection, quality: RepoQuality) -> None:
    """Store quality data in the cache."""
    conn.execute(
        """
        INSERT OR REPLACE INTO repo_quality
            (repo_name, stars, forks, contributors, has_ci, has_test_framework,
             is_archived, is_fork, passes, reason, checked_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            quality.repo_name, quality.stars, quality.forks,
            quality.contributors, int(quality.has_ci),
            int(quality.has_test_framework), int(quality.is_archived),
            int(quality.is_fork), int(quality.passes), quality.reason,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def _get_gh_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _fetch_repo_info(repo_name: str) -> dict | None:
    """Fetch repository metadata from GitHub API."""
    headers = _get_gh_headers()
    try:
        with httpx.Client(timeout=15, headers=headers) as client:
            resp = client.get(f"{_GITHUB_API}/repos/{repo_name}")
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (403, 429):
                log.debug("Rate limited fetching repo info for %s", repo_name)
            else:
                log.debug("Got %d for repo info %s", resp.status_code, repo_name)
    except httpx.HTTPError as e:
        log.debug("HTTP error fetching repo info %s: %s", repo_name, e)
    return None


def _count_contributors(repo_name: str) -> int:
    """Estimate contributor count (limited to first page of 30)."""
    headers = _get_gh_headers()
    try:
        with httpx.Client(timeout=15, headers=headers) as client:
            resp = client.get(
                f"{_GITHUB_API}/repos/{repo_name}/contributors",
                params={"per_page": 5, "anon": "false"},
            )
            if resp.status_code == 200:
                return len(resp.json())
    except httpx.HTTPError:
        pass
    return 0


# ---------------------------------------------------------------------------
# CI detection (from repo tree)
# ---------------------------------------------------------------------------

_CI_INDICATORS = [
    ".github/workflows",
    ".travis.yml",
    ".circleci",
    "Jenkinsfile",
    ".gitlab-ci.yml",
    "azure-pipelines.yml",
    "tox.ini",
    "noxfile.py",
    "Makefile",
]


def _check_ci_from_tree(repo_dir: Path) -> bool:
    """Check for CI presence in a local repo clone."""
    for indicator in _CI_INDICATORS:
        path = repo_dir / indicator
        if path.exists():
            return True
    return False


def _check_test_framework(repo_dir: Path) -> bool:
    """Check if the repo uses a test framework."""
    # Check pyproject.toml
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(errors="replace").lower()
            if "pytest" in content or "unittest" in content or "nose" in content:
                return True
        except OSError:
            pass

    # Check setup.py / setup.cfg
    for name in ("setup.py", "setup.cfg"):
        path = repo_dir / name
        if path.exists():
            try:
                content = path.read_text(errors="replace").lower()
                if "pytest" in content or "unittest" in content:
                    return True
            except OSError:
                pass

    # Check requirements files
    for req_file in repo_dir.glob("requirements*.txt"):
        try:
            content = req_file.read_text(errors="replace").lower()
            if "pytest" in content:
                return True
        except OSError:
            pass

    # Check for test directories
    if (repo_dir / "tests").is_dir() or (repo_dir / "test").is_dir():
        return True

    return False


# ---------------------------------------------------------------------------
# Main quality check
# ---------------------------------------------------------------------------


def check_repo_quality(
    repo_name: str,
    repo_dir: Path | None = None,
    min_stars: int = 5,
    min_contributors: int = 1,
    require_ci: bool = False,
    require_tests: bool = True,
    allow_archived: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
) -> RepoQuality:
    """Check if a repository meets quality thresholds.

    Uses cached data when available to minimize API calls.

    Args:
        allow_archived: If True, don't reject archived repositories.
    """
    # Check cache first
    conn = get_connection(db_path)
    try:
        cached = _get_cached_quality(conn, repo_name)
        if cached is not None:
            # Re-evaluate with current thresholds (cache stores raw data,
            # thresholds may have changed via CLI flags)
            cached = _apply_thresholds(
                cached,
                min_stars=min_stars,
                min_contributors=min_contributors,
                require_ci=require_ci,
                require_tests=require_tests,
                allow_archived=allow_archived,
            )
            log.debug("Using cached quality for %s (passes=%s)", repo_name, cached.passes)
            return cached
    finally:
        conn.close()

    quality = RepoQuality(repo_name=repo_name)

    # Fetch from GitHub API
    repo_info = _fetch_repo_info(repo_name)
    if repo_info:
        quality.stars = repo_info.get("stargazers_count", 0)
        quality.forks = repo_info.get("forks_count", 0)
        quality.is_archived = repo_info.get("archived", False)
        quality.is_fork = repo_info.get("fork", False)

        # Quick contributor count
        quality.contributors = _count_contributors(repo_name)
    else:
        log.debug("Could not fetch repo info for %s, using defaults", repo_name)

    # Check local repo for CI and test framework
    if repo_dir and repo_dir.exists():
        quality.has_ci = _check_ci_from_tree(repo_dir)
        quality.has_test_framework = _check_test_framework(repo_dir)

    # --- Apply thresholds ---
    quality = _apply_thresholds(
        quality,
        min_stars=min_stars,
        min_contributors=min_contributors,
        require_ci=require_ci,
        require_tests=require_tests,
        allow_archived=allow_archived,
    )

    # Cache result
    conn = get_connection(db_path)
    try:
        _cache_quality(conn, quality)
    finally:
        conn.close()

    return quality


def _apply_thresholds(
    quality: RepoQuality,
    *,
    min_stars: int = 5,
    min_contributors: int = 1,
    require_ci: bool = False,
    require_tests: bool = True,
    allow_archived: bool = False,
) -> RepoQuality:
    """Apply quality thresholds to a RepoQuality object."""
    if not allow_archived and quality.is_archived:
        quality.passes = False
        quality.reason = "Repository is archived"
    elif quality.stars < min_stars:
        quality.passes = False
        quality.reason = f"Too few stars ({quality.stars} < {min_stars})"
    elif quality.contributors < min_contributors:
        quality.passes = False
        quality.reason = f"Too few contributors ({quality.contributors} < {min_contributors})"
    elif require_ci and not quality.has_ci:
        quality.passes = False
        quality.reason = "No CI/CD detected"
    elif require_tests and not quality.has_test_framework:
        quality.passes = False
        quality.reason = "No test framework detected"
    else:
        quality.passes = True
        quality.reason = "OK"
    return quality
