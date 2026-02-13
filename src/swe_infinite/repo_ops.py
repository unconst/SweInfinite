"""
Repository operations for the SWE-rebench pipeline.

Handles:
  - Cloning GitHub repos (full history for diff computation)
  - Computing diffs between base_commit and head_commit (or merge_commit)
  - Splitting patches into solution patch (non-test) and test patch
  - License detection against the permissive allowlist
  - File count validation (1-15 files in solution patch)
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("swe-infinite.repo_ops")

from .paths import REPOS_DIR

# ---------------------------------------------------------------------------
# Permissive license allowlist (from paper, Appendix D)
# ---------------------------------------------------------------------------

PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0", "apache 2.0", "apache license 2.0", "apache license, version 2.0",
    "bsd-2-clause", "bsd 2-clause",
    "bsd-3-clause", "bsd 3-clause",
    "bsd-4-clause", "bsd 4-clause",
    "isc",
    "cc0-1.0", "cc0 1.0",
    "zpl-2.1", "zpl 2.1",
}

# Keywords that strongly indicate a permissive license in LICENSE file text
_LICENSE_KEYWORDS = [
    ("mit license", "MIT"),
    ("apache license", "Apache-2.0"),
    ("bsd 2-clause", "BSD-2-Clause"),
    ("bsd 3-clause", "BSD-3-Clause"),
    ("bsd 4-clause", "BSD-4-Clause"),
    ("isc license", "ISC"),
    ("creative commons zero", "CC0-1.0"),
    ("cc0 1.0 universal", "CC0-1.0"),
    ("permission is hereby granted, free of charge", "MIT"),
]

# ---------------------------------------------------------------------------
# Test file detection heuristics (multi-language)
# ---------------------------------------------------------------------------

_TEST_PATH_PATTERNS = [
    # Python
    re.compile(r"(^|/)tests?/"),           # files inside test/ or tests/ dir
    re.compile(r"(^|/)test_[^/]+\.py$"),   # test_foo.py at any level
    re.compile(r"(^|/)[^/]+_test\.py$"),   # foo_test.py at any level
    re.compile(r"(^|/)conftest\.py$"),      # pytest conftest
    re.compile(r"(^|/)testing/"),           # testing/ directory
    # TypeScript/JavaScript
    re.compile(r"(^|/)__tests__/"),
    re.compile(r"\.(test|spec)\.(ts|tsx|js|jsx|mjs)$"),
    # Java
    re.compile(r"(^|/)src/test/"),
    re.compile(r"Test\.java$"),
    # Go
    re.compile(r"_test\.go$"),
]


def _is_test_file(path: str) -> bool:
    """Heuristic: does this file path look like a test file?

    Supports Python, TypeScript, JavaScript, Java, and Go test conventions.
    """
    return any(pat.search(path) for pat in _TEST_PATH_PATTERNS)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(
    args: list[str], cwd: Path, timeout: int = 300
) -> subprocess.CompletedProcess:
    cmd = ["git"] + args
    log.debug("$ %s  (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def clone_repo(repo_name: str) -> Path:
    """Clone a GitHub repo into workspace/repos/{owner}__{name}/.

    If already cloned, fetches latest and returns the path.
    Uses full clone (not shallow) so we can compute diffs between arbitrary commits.
    """
    safe_name = repo_name.replace("/", "__")
    repo_dir = REPOS_DIR / safe_name

    if repo_dir.exists():
        log.info("Repo %s already cloned. Fetching latest...", repo_name)
        result = _run_git(["fetch", "--all"], cwd=repo_dir, timeout=120)
        if result.returncode != 0:
            log.warning("git fetch failed for %s: %s", repo_name, result.stderr[:200])
        return repo_dir

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo_name}.git"
    log.info("Cloning %s → %s", url, repo_dir)

    try:
        result = _run_git(
            ["clone", "--no-checkout", "--filter=blob:none", url, str(repo_dir)],
            cwd=REPOS_DIR,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        log.warning("Clone timed out for %s (>180s). Skipping large repo.", repo_name)
        # Clean up partial clone
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        raise RuntimeError(f"git clone timed out for {repo_name}")

    if result.returncode != 0:
        log.error("Clone failed for %s: %s", repo_name, result.stderr[:500])
        # Clean up partial clone
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        raise RuntimeError(f"git clone failed for {repo_name}")

    return repo_dir


# ---------------------------------------------------------------------------
# Diff + patch splitting
# ---------------------------------------------------------------------------


def compute_diff(
    repo_dir: Path,
    base_sha: str,
    head_sha: str | None,
    merge_commit_sha: str | None,
) -> str | None:
    """Compute the unified diff for a PR.

    Prefers head_sha over merge_commit_sha (paper Section G: head_commit
    isolates PR-specific changes better).
    """
    target = head_sha or merge_commit_sha
    if not target or not base_sha:
        log.warning("Missing SHAs: base=%s head=%s merge=%s", base_sha, head_sha, merge_commit_sha)
        return None

    result = _run_git(["diff", base_sha, target], cwd=repo_dir, timeout=120)
    if result.returncode != 0:
        # Commits might not exist if clone was shallow or branch was deleted
        log.warning("git diff failed (%s..%s): %s", base_sha[:8], target[:8], result.stderr[:200])
        return None

    diff_text = result.stdout
    if not diff_text.strip():
        log.debug("Empty diff between %s and %s", base_sha[:8], target[:8])
        return None

    return diff_text


def split_patch(full_diff: str) -> tuple[str, str]:
    """Split a unified diff into (solution_patch, test_patch).

    solution_patch: changes to non-test files
    test_patch: changes to test files only
    """
    solution_hunks: list[str] = []
    test_hunks: list[str] = []

    # Split on file boundaries (diff --git a/... b/...)
    file_diffs = re.split(r"(?=^diff --git )", full_diff, flags=re.MULTILINE)

    for file_diff in file_diffs:
        if not file_diff.strip():
            continue

        # Extract the file path from "diff --git a/path b/path"
        header_match = re.match(r"diff --git a/(.+?) b/(.+?)$", file_diff, re.MULTILINE)
        if not header_match:
            continue

        file_path = header_match.group(2)

        if _is_test_file(file_path):
            test_hunks.append(file_diff)
        else:
            solution_hunks.append(file_diff)

    return "".join(solution_hunks), "".join(test_hunks)


def count_files_in_patch(patch: str) -> int:
    """Count the number of distinct files modified in a unified diff."""
    return len(re.findall(r"^diff --git ", patch, re.MULTILINE))


def count_changed_lines(patch: str) -> tuple[int, int]:
    """Count added and removed lines in a patch (excluding headers).

    Returns (added, removed).
    """
    added = len(re.findall(r"^\+[^+]", patch, re.MULTILINE))
    removed = len(re.findall(r"^-[^-]", patch, re.MULTILINE))
    return added, removed


def is_comment_only_patch(patch: str) -> bool:
    """Check if a patch only modifies comments/docstrings.

    Returns True if >90% of changed lines are comments or docstrings.
    """
    added_lines = [
        l[1:] for l in patch.split("\n")
        if l.startswith("+") and not l.startswith("+++")
    ]
    removed_lines = [
        l[1:] for l in patch.split("\n")
        if l.startswith("-") and not l.startswith("---")
    ]

    all_changed = added_lines + removed_lines
    if not all_changed:
        return False

    comment_count = 0
    for line in all_changed:
        stripped = line.strip()
        if (
            stripped.startswith("#")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
            or stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or not stripped  # blank lines
        ):
            comment_count += 1

    return comment_count / len(all_changed) > 0.9


def is_config_only_patch(patch: str) -> bool:
    """Check if a patch only modifies configuration files.

    Returns True if all modified files are config/metadata files.
    """
    config_patterns = [
        re.compile(r"\.(cfg|ini|toml|yml|yaml|json|xml|conf)$"),
        re.compile(r"(^|/)\."),  # dotfiles
        re.compile(r"(^|/)(Makefile|Dockerfile|docker-compose)"),
        re.compile(r"(^|/)(setup\.py|setup\.cfg|pyproject\.toml)$"),
        re.compile(r"(^|/)(requirements.*\.txt)$"),
        re.compile(r"(^|/)MANIFEST\.in$"),
    ]

    file_paths = re.findall(r"^diff --git a/.+? b/(.+?)$", patch, re.MULTILINE)
    if not file_paths:
        return False

    for path in file_paths:
        is_config = any(pat.search(path) for pat in config_patterns)
        if not is_config:
            return False

    return True


def validate_patch_complexity(
    solution_patch: str,
    min_changed_lines: int = 3,
    max_changed_lines: int = 1000,
) -> str | None:
    """Validate that a solution patch has appropriate complexity.

    Returns a rejection reason string, or None if the patch passes.
    """
    if not solution_patch or not solution_patch.strip():
        return "Patch is empty"

    added, removed = count_changed_lines(solution_patch)
    total = added + removed

    if total < min_changed_lines:
        return f"Patch too trivial ({total} lines changed, minimum {min_changed_lines})"

    if total > max_changed_lines:
        return f"Patch too large ({total} lines changed, maximum {max_changed_lines})"

    if is_comment_only_patch(solution_patch):
        return "Patch only modifies comments/docstrings"

    if is_config_only_patch(solution_patch):
        return "Patch only modifies configuration files"

    return None


# ---------------------------------------------------------------------------
# License detection
# ---------------------------------------------------------------------------

_LICENSE_FILE_NAMES = [
    "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE.rst",
    "LICENCE", "LICENCE.md", "LICENCE.txt",
    "COPYING", "COPYING.md", "COPYING.txt",
]


def detect_license(repo_dir: Path) -> str | None:
    """Detect the license of a repository by reading common license files.

    Returns the SPDX-like name if permissive, or None if not found/not permissive.
    """
    for name in _LICENSE_FILE_NAMES:
        license_file = repo_dir / name
        if license_file.exists():
            try:
                text = license_file.read_text(errors="replace").lower()
            except Exception:
                continue

            for keyword, spdx in _LICENSE_KEYWORDS:
                if keyword in text:
                    return spdx

    return None


def is_permissive_license(license_name: str | None) -> bool:
    """Check if a license name is in our permissive allowlist."""
    if not license_name:
        return False
    return license_name.lower() in PERMISSIVE_LICENSES


# ---------------------------------------------------------------------------
# High-level: extract a task from a candidate
# ---------------------------------------------------------------------------


def extract_candidate(candidate: dict) -> dict | None:
    """Process a single candidate into extracted task data.

    Clones the repo (if needed), computes the diff, splits patches,
    checks license, and validates file count.

    Returns a dict with extracted fields or None if the candidate fails validation.
    """
    repo_name = candidate["repo_name"]
    base_sha = candidate["base_sha"]
    head_sha = candidate["head_sha"]
    merge_sha = candidate["merge_commit_sha"]

    # 1. Clone
    try:
        repo_dir = clone_repo(repo_name)
    except RuntimeError:
        return None

    # Need to checkout so license detection works (may fetch blobs on demand with partial clone)
    _run_git(["checkout", base_sha, "--force"], cwd=repo_dir, timeout=120)

    # 2. License check
    license_name = detect_license(repo_dir)
    if not is_permissive_license(license_name):
        log.info("Skipping %s — non-permissive or unknown license: %s", repo_name, license_name)
        return None

    # 3. Compute diff
    full_diff = compute_diff(repo_dir, base_sha, head_sha, merge_sha)
    if not full_diff:
        log.info("Skipping %s PR#%s — no diff", repo_name, candidate["pr_number"])
        return None

    # 4. Split patches
    solution_patch, test_patch = split_patch(full_diff)

    # Must have BOTH solution and test changes
    if not solution_patch.strip():
        log.info("Skipping %s PR#%s — no non-test changes", repo_name, candidate["pr_number"])
        return None
    if not test_patch.strip():
        log.info("Skipping %s PR#%s — no test changes", repo_name, candidate["pr_number"])
        return None

    # 5. File count check (1-15 files in solution patch)
    file_count = count_files_in_patch(solution_patch)
    if file_count < 1 or file_count > 15:
        log.info(
            "Skipping %s PR#%s — solution patch modifies %d files (need 1-15)",
            repo_name, candidate["pr_number"], file_count,
        )
        return None

    # 6. Patch complexity check
    complexity_issue = validate_patch_complexity(solution_patch)
    if complexity_issue:
        log.info(
            "Skipping %s PR#%s — %s",
            repo_name, candidate["pr_number"], complexity_issue,
        )
        return None

    # Determine which commit was used
    commit_name = "head_commit" if head_sha else "merge_commit"

    return {
        "repo_name": repo_name,
        "pr_number": candidate["pr_number"],
        "issue_number": candidate["issue_number"],
        "issue_title": candidate.get("issue_title", ""),
        "issue_body": candidate.get("issue_body", ""),
        "pr_title": candidate.get("pr_title", ""),
        "base_commit": base_sha,
        "head_commit": head_sha,
        "merge_commit": merge_sha,
        "solution_patch": solution_patch,
        "test_patch": test_patch,
        "license_name": license_name,
        "commit_name": commit_name,
        "num_modified_files": file_count,
        "repo_dir": str(repo_dir),
    }
