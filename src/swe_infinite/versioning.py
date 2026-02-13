"""
Version grouping for the SWE-rebench pipeline.

Analyzes git tags to determine project versions, normalizes them to
major.minor format, and groups task instances by (repo, version).
For each group, selects the most recent base_commit as the environment
setup commit (per paper Section 2.2).
"""

from __future__ import annotations

import logging
import re
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("swe-infinite.versioning")


# ---------------------------------------------------------------------------
# Git tag → version mapping
# ---------------------------------------------------------------------------

# Matches version-like tags: v1.2.3, 1.2.3, release-1.2, ver1.2.3, etc.
_VERSION_TAG_RE = re.compile(
    r"(?:^|[\-_/])"          # start of string or separator
    r"v?(?:er(?:sion)?[\-_]?)?"  # optional v/ver/version prefix
    r"(\d+)\.(\d+)"          # major.minor (required)
    r"(?:\.(\d+))?",         # .patch (optional)
    re.IGNORECASE,
)


def _normalize_version(tag: str) -> str | None:
    """Normalize a git tag to major.minor format.

    Examples:
        v1.2.3  -> 1.2
        1.2     -> 1.2
        release-3.4.5 -> 3.4
        not-a-version -> None
    """
    m = _VERSION_TAG_RE.search(tag)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}"


def get_repo_tags(repo_dir: Path) -> list[tuple[str, str, str]]:
    """Get all tags from a repo with their associated commits and dates.

    Returns list of (tag_name, commit_sha, date) sorted by date descending.
    """
    result = subprocess.run(
        ["git", "tag", "-l", "--format=%(refname:short)\t%(objectname:short)\t%(*objectname:short)\t%(creatordate:iso-strict)"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        log.warning("git tag failed for %s: %s", repo_dir, result.stderr[:200])
        return []

    tags = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        tag_name = parts[0]
        # For annotated tags, *objectname is the commit; for lightweight, objectname
        commit = parts[2] if parts[2] else parts[1]
        date = parts[3]
        tags.append((tag_name, commit, date))

    # Sort by date descending (most recent first)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


def build_version_map(repo_dir: Path) -> dict[str, str]:
    """Build a mapping from commit_sha -> normalized version for a repo.

    For each tag, finds the normalized major.minor version. If multiple
    tags map to the same commit, the most specific version wins.
    """
    tags = get_repo_tags(repo_dir)
    commit_to_version: dict[str, str] = {}

    for tag_name, commit_sha, _date in tags:
        version = _normalize_version(tag_name)
        if version and commit_sha not in commit_to_version:
            commit_to_version[commit_sha] = version

    return commit_to_version


def find_version_for_commit(
    repo_dir: Path,
    commit_sha: str,
    version_map: dict[str, str] | None = None,
) -> str | None:
    """Find the version a commit belongs to using git describe.

    First checks the version_map (tag -> version), then falls back to
    `git describe --tags --abbrev=0` to find the nearest ancestor tag.
    """
    if version_map is None:
        version_map = build_version_map(repo_dir)

    # Direct match
    short_sha = commit_sha[:7]
    for sha, ver in version_map.items():
        if sha.startswith(short_sha) or short_sha.startswith(sha):
            return ver

    # Use git describe to find the nearest tag ancestor
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", commit_sha],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 0:
        tag = result.stdout.strip()
        return _normalize_version(tag)

    return None


# ---------------------------------------------------------------------------
# Group tasks by version
# ---------------------------------------------------------------------------


def group_tasks_by_version(
    extracted_tasks: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    """Group extracted task dicts by (repo_name, version).

    For tasks without a detectable version, assigns a unique version string
    so each gets its own environment.

    Also selects the environment_setup_commit for each group (the most
    recent base_commit in the group, per paper Section 2.2).
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    version_caches: dict[str, dict[str, str]] = {}  # repo_dir -> version_map

    for task in extracted_tasks:
        repo_name = task["repo_name"]
        repo_dir = Path(task["repo_dir"])
        base_commit = task["base_commit"]

        # Build or reuse version map for this repo
        if str(repo_dir) not in version_caches:
            version_caches[str(repo_dir)] = build_version_map(repo_dir)
        vmap = version_caches[str(repo_dir)]

        version = find_version_for_commit(repo_dir, base_commit, vmap)

        if version is None:
            # Assign unique version so task gets its own environment
            version = f"unique-{uuid.uuid4().hex[:8]}"
            log.debug(
                "No version tag for %s @ %s — assigned %s",
                repo_name, base_commit[:8], version,
            )

        task["version"] = version
        groups[(repo_name, version)].append(task)

    # Select environment_setup_commit for each group
    for key, tasks in groups.items():
        # Sort by created_at (from PR event) descending, pick most recent base_commit
        # Since we may not have created_at on the task dict, use base_commit ordering
        # The paper says: "select the base_commit of the PR with the most recent date"
        # For now, use the last task's base_commit (they're already ordered by PR number)
        setup_commit = tasks[-1]["base_commit"]
        for t in tasks:
            t["environment_setup_commit"] = setup_commit

    log.info(
        "Grouped %d tasks into %d version groups",
        len(extracted_tasks), len(groups),
    )
    return dict(groups)
