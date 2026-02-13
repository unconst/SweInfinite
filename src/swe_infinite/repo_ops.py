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

    Returns the SPDX-like name if recognized (permissive or non-permissive),
    or None if not found/unrecognized.
    """
    for name in _LICENSE_FILE_NAMES:
        license_file = repo_dir / name
        if license_file.exists():
            try:
                text = license_file.read_text(errors="replace").lower()
            except Exception:
                continue

            # Check permissive first
            for keyword, spdx in _LICENSE_KEYWORDS:
                if keyword in text:
                    return spdx

            # Check non-permissive
            for keyword in _NON_PERMISSIVE_KEYWORDS:
                if keyword in text:
                    if "affero" in text:
                        return "AGPL-3.0"
                    if "lesser" in text:
                        return "LGPL-3.0"
                    if "gnu general public" in text:
                        return "GPL-3.0"
                    if "server side" in text:
                        return "SSPL-1.0"
                    if "business source" in text:
                        return "BUSL-1.1"
                    if "elastic" in text:
                        return "Elastic-2.0"

    return None


# Licenses that are explicitly non-permissive (hard reject)
_NON_PERMISSIVE_LICENSES = {
    "gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-2.1", "lgpl-3.0",
    "sspl-1.0", "busl-1.1", "elastic-2.0",
}

# Keywords that indicate a non-permissive license in LICENSE file text
_NON_PERMISSIVE_KEYWORDS = [
    "gnu general public license",
    "gnu affero general public license",
    "server side public license",
    "business source license",
    "elastic license",
]


def is_permissive_license(license_name: str | None) -> bool:
    """Check if a license name is acceptable.

    Returns True for:
    - Known permissive licenses
    - Unknown/undetected licenses (None) — treated as acceptable
    Returns False only for explicitly non-permissive licenses.
    """
    if not license_name:
        return True  # Unknown license is OK — most public repos are usable
    name = license_name.lower()
    if name in PERMISSIVE_LICENSES:
        return True
    if name in _NON_PERMISSIVE_LICENSES:
        return False
    return True  # Unknown license type — default to accept


# ---------------------------------------------------------------------------
# Language detection from patch
# ---------------------------------------------------------------------------

_LANG_BY_EXTENSION = {
    ".py": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".java": "java",
    ".go": "go",
}

# Extensions that are never source code — skip when detecting language
_NON_CODE_EXTENSIONS = {
    ".json", ".yml", ".yaml", ".toml", ".xml", ".cfg", ".ini", ".conf",
    ".lock", ".md", ".rst", ".txt", ".csv", ".html", ".css", ".scss",
    ".svg", ".png", ".jpg", ".gif", ".ico",
}


def _detect_language_from_patch(patch: str) -> str | None:
    """Detect the primary programming language from file extensions in a patch.

    Returns None if no recognized source-code extensions are found (e.g. the
    patch only touches config files like .json, .yml, .lock).
    """
    file_paths = re.findall(r"^diff --git a/.+? b/(.+?)$", patch, re.MULTILINE)
    lang_counts: dict[str, int] = {}
    for fpath in file_paths:
        ext = Path(fpath).suffix.lower()
        if ext in _NON_CODE_EXTENSIONS:
            continue
        lang = _LANG_BY_EXTENSION.get(ext)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    if not lang_counts:
        return None
    return max(lang_counts, key=lang_counts.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# High-level: extract a task from a candidate
# ---------------------------------------------------------------------------


def extract_candidate(
    candidate: dict,
    *,
    allow_non_permissive: bool = False,
    max_patch_files: int = 15,
    min_patch_lines: int = 3,
    max_patch_lines: int = 1000,
    generate_tests: bool = False,
    skip_patch_checks: bool = False,
) -> dict | None:
    """Process a single candidate into extracted task data.

    Clones the repo (if needed), computes the diff, splits patches,
    checks license, and validates file count.

    Args:
        candidate: Dict with repo_name, base_sha, head_sha, merge_commit_sha, etc.
        allow_non_permissive: If True, skip the permissive license check.
        max_patch_files: Maximum files allowed in the solution patch.
        min_patch_lines: Minimum changed lines for patch complexity check.
        max_patch_lines: Maximum changed lines for patch complexity check.
        generate_tests: If True, use LLM to generate tests when PR has no test changes.
        skip_patch_checks: If True, skip file count and complexity checks
            (config-only, comment-only, size limits).

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
    if not allow_non_permissive and not is_permissive_license(license_name):
        log.info("Skipping %s — non-permissive or unknown license: %s", repo_name, license_name)
        return None

    # 3. Compute diff
    full_diff = compute_diff(repo_dir, base_sha, head_sha, merge_sha)
    if not full_diff:
        log.info("Skipping %s PR#%s — no diff", repo_name, candidate["pr_number"])
        return None

    # 4. Split patches
    solution_patch, test_patch = split_patch(full_diff)

    # Must have solution changes
    if not solution_patch.strip():
        log.info("Skipping %s PR#%s — no non-test changes", repo_name, candidate["pr_number"])
        return None

    # If no test changes, optionally generate tests via LLM
    if not test_patch.strip():
        if not generate_tests:
            log.info("Skipping %s PR#%s — no test changes", repo_name, candidate["pr_number"])
            return None

        # Generate tests using LLM
        log.info("No test changes in %s PR#%s — generating tests via LLM...",
                 repo_name, candidate["pr_number"])
        try:
            from .test_generator import generate_test_patch

            # Determine language from candidate metadata or file extensions
            language = _detect_language_from_patch(solution_patch)
            if language is None:
                log.info("Skipping %s PR#%s — cannot determine language from patch (config-only files?)",
                         repo_name, candidate["pr_number"])
                return None

            # Temporarily apply the solution to read fixed source files
            target = head_sha or merge_sha
            _run_git(["checkout", target, "--force"], cwd=repo_dir, timeout=120)

            generated_patch = generate_test_patch(
                repo_dir=repo_dir,
                solution_patch=solution_patch,
                language=language,
                repo_name=repo_name,
                pr_title=candidate.get("pr_title", ""),
                pr_body=candidate.get("issue_body", ""),
            )

            # Restore to base commit for downstream processing
            _run_git(["checkout", base_sha, "--force"], cwd=repo_dir, timeout=120)

            if generated_patch:
                test_patch = generated_patch
                log.info("  Generated test patch for %s PR#%s (%d chars)",
                         repo_name, candidate["pr_number"], len(test_patch))
            else:
                log.info("Skipping %s PR#%s — test generation failed",
                         repo_name, candidate["pr_number"])
                return None

        except Exception as exc:
            log.warning("Skipping %s PR#%s — test generation error: %s",
                        repo_name, candidate["pr_number"], exc)
            # Restore to base commit on error
            _run_git(["checkout", base_sha, "--force"], cwd=repo_dir, timeout=120)
            return None

    # 5. File count check
    file_count = count_files_in_patch(solution_patch)
    if not skip_patch_checks and (file_count < 1 or file_count > max_patch_files):
        log.info(
            "Skipping %s PR#%s — solution patch modifies %d files (need 1-%d)",
            repo_name, candidate["pr_number"], file_count, max_patch_files,
        )
        return None

    # 6. Patch complexity check (config-only, comment-only, size limits)
    if not skip_patch_checks:
        complexity_issue = validate_patch_complexity(
            solution_patch,
            min_changed_lines=min_patch_lines,
            max_changed_lines=max_patch_lines,
        )
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
