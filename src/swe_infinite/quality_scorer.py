"""
LLM-based quality scoring for the SWE-rebench pipeline.

Evaluates and scores task candidates across three dimensions:
  1. Problem statement quality (clarity, completeness, standalone-ness)
  2. Test quality (meaningful assertions, coverage, isolation)
  3. Difficulty (conceptual complexity, domain knowledge, scope)

Also provides problem statement cleanup to remove noise (boilerplate,
images, Discord signup text, etc.) while preserving the core problem.

Uses the Cursor agent CLI as the LLM backend.
Falls back to heuristic scoring when the agent CLI is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("swe-infinite.quality_scorer")


# ---------------------------------------------------------------------------
# Score result
# ---------------------------------------------------------------------------


@dataclass
class QualityScores:
    """Quality scores for a task candidate."""

    issue_text_score: int | None = None   # 1-5
    test_score: int | None = None         # 1-5
    difficulty_score: int | None = None   # 1-5
    cleaned_problem_statement: str = ""
    rejection_reason: str = ""
    _min_quality_score: int = 2           # configurable threshold

    @property
    def passes_threshold(self) -> bool:
        """Check if the task passes minimum quality thresholds."""
        threshold = self._min_quality_score
        if self.issue_text_score is not None and self.issue_text_score < threshold:
            return False
        if self.test_score is not None and self.test_score < threshold:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "difficulty_score": self.difficulty_score,
            "issue_text_score": self.issue_text_score,
            "test_score": self.test_score,
        }


# ---------------------------------------------------------------------------
# Heuristic pre-filters (cheap, no LLM needed)
# ---------------------------------------------------------------------------

# Patterns that indicate low-quality issues
_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")
_URL_PATTERN = re.compile(r"https?://\S+")
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
_BOILERPLATE_PATTERNS = [
    re.compile(r"discord\s*:", re.IGNORECASE),
    re.compile(r"I['']d like to work on this", re.IGNORECASE),
    re.compile(r"Apertre\s+\d+\.\d+", re.IGNORECASE),
    re.compile(r"verification\s+checklist", re.IGNORECASE),
    re.compile(r"forked the repository", re.IGNORECASE),
    re.compile(r"read CONTRIBUTING", re.IGNORECASE),
    re.compile(r"To get assigned", re.IGNORECASE),
    re.compile(r"comment below with", re.IGNORECASE),
    re.compile(r"Hacktoberfest", re.IGNORECASE),
    re.compile(r"good first issue", re.IGNORECASE),
    re.compile(r"help wanted", re.IGNORECASE),
]

_SKIP_TITLE_PATTERNS = [
    re.compile(r"\bbump\b", re.IGNORECASE),
    re.compile(r"\bdeps?\b", re.IGNORECASE),
    re.compile(r"\bchore\b", re.IGNORECASE),
    # Only match standalone CI prefixes like "[CI]" or "CI:" — not words containing "ci"
    re.compile(r"(?:^|\[)\s*ci\s*[\]:]", re.IGNORECASE),
    re.compile(r"\btypo\b", re.IGNORECASE),
    re.compile(r"\bformat(?:ting)?\b", re.IGNORECASE),
    re.compile(r"\blint(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\bversion\s+bump\b", re.IGNORECASE),
]


def heuristic_issue_prefilter(title: str, body: str) -> str | None:
    """Fast heuristic pre-filter for issue quality.

    Returns a rejection reason string, or None if the issue passes.
    """
    title = title or ""
    body = body or ""

    # Skip maintenance/chore titles
    for pat in _SKIP_TITLE_PATTERNS:
        if pat.search(title):
            return f"Title matches skip pattern: {pat.pattern}"

    # Body too short
    if len(body.strip()) < 10:
        return f"Body too short ({len(body.strip())} chars)"

    # Remove code blocks to measure actual prose
    prose = _CODE_BLOCK_PATTERN.sub("", body)
    prose = _IMAGE_PATTERN.sub("", prose)
    prose_len = len(prose.strip())

    # Issue is mostly code blocks (too prescriptive / solution-in-issue)
    code_blocks = _CODE_BLOCK_PATTERN.findall(body)
    code_len = sum(len(b) for b in code_blocks)
    if code_len > 0 and code_len / max(len(body), 1) > 0.7 and prose_len < 100:
        return "Issue body is >70% code blocks with little prose"

    # Image-only issues (images but no substantial text)
    images = _IMAGE_PATTERN.findall(body)
    if images and prose_len < 50:
        return "Issue appears to be image-only with insufficient text"

    # Giant spec documents (likely feature requests, not bug reports)
    if len(body) > 8000:
        # Check if it's a spec/PRD-like document
        spec_indicators = ["acceptance criteria", "requirements", "## task",
                          "## context", "phase", "roadmap", "milestone"]
        indicator_count = sum(1 for s in spec_indicators if s.lower() in body.lower())
        if indicator_count >= 3:
            return "Issue appears to be a spec/PRD document, not a bug report"

    return None


def heuristic_patch_quality(solution_patch: str, test_patch: str) -> str | None:
    """Quick heuristic check on patch quality.

    Returns rejection reason or None.
    """
    solution_patch = solution_patch or ""
    test_patch = test_patch or ""

    # Count actual code lines changed (ignoring headers, context lines)
    added_lines = len(re.findall(r"^\+[^+]", solution_patch, re.MULTILINE))
    removed_lines = len(re.findall(r"^-[^-]", solution_patch, re.MULTILINE))
    total_changed = added_lines + removed_lines

    # Too trivial (likely typo fix)
    if total_changed < 3:
        return f"Solution patch too trivial ({total_changed} lines changed)"

    # Too large (likely refactor, not bug fix)
    if total_changed > 1000:
        return f"Solution patch too large ({total_changed} lines changed)"

    # Check if patch only modifies comments/docstrings
    # Simple heuristic: if all added lines start with # or are inside triple quotes
    added = [l[1:] for l in solution_patch.split("\n") if l.startswith("+") and not l.startswith("+++")]
    if added:
        comment_lines = sum(1 for l in added if l.strip().startswith("#") or l.strip().startswith('"""') or l.strip().startswith("'''"))
        if comment_lines / max(len(added), 1) > 0.9:
            return "Solution patch appears to only modify comments/docstrings"

    # Check test patch has actual assertions (covers multiple test frameworks)
    test_added = [l[1:] for l in test_patch.split("\n") if l.startswith("+") and not l.startswith("+++")]
    has_assert = any("assert" in l.lower() for l in test_added)
    has_pytest_raises = any("pytest.raises" in l or "raises(" in l for l in test_added)
    has_expect = any("expect(" in l or ".toBe(" in l or ".toEqual(" in l for l in test_added)
    has_should = any(".should" in l for l in test_added)
    has_junit = any("assertThat(" in l or "verify(" in l for l in test_added)
    if not (has_assert or has_pytest_raises or has_expect or has_should or has_junit):
        return "Test patch appears to have no assertions"

    return None


# ---------------------------------------------------------------------------
# Problem statement cleanup
# ---------------------------------------------------------------------------


def clean_problem_statement(problem_statement: str) -> str:
    """Remove noise from a problem statement while preserving the core problem.

    Removes:
      - Discord signup / contribution boilerplate
      - Apertre / Hacktoberfest boilerplate
      - Verification checklists
      - Image references (replace with [image])
      - Excessive formatting

    Preserves:
      - Bug description
      - Reproduction steps
      - Expected vs actual behavior
      - Error messages / tracebacks
    """
    problem_statement = problem_statement or ""
    lines = problem_statement.split("\n")
    cleaned_lines = []
    skip_section = False

    for line in lines:
        stripped = line.strip()

        # Detect boilerplate sections to skip
        if any(pat.search(stripped) for pat in _BOILERPLATE_PATTERNS):
            skip_section = True
            continue

        # Resume after blank line following boilerplate
        if skip_section and not stripped:
            skip_section = False
            continue

        if skip_section:
            continue

        # Replace image markdown with placeholder
        line = _IMAGE_PATTERN.sub("[image]", line)

        # Keep the line
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()

    # Remove trailing checklist items (unchecked boxes)
    result = re.sub(r"\n\s*-\s*\[ \]\s*.*$", "", result, flags=re.MULTILINE)

    # Collapse excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


# ---------------------------------------------------------------------------
# LLM-based scoring
# ---------------------------------------------------------------------------

_SCORE_PROMPT = """\
You are evaluating a software engineering task for a benchmark dataset. \
Score the following dimensions on a 1-5 scale.

## Problem Statement
{problem_statement}

## Solution Patch (diff)
{solution_patch_summary}

## Test Patch (diff)
{test_patch_summary}

## Scoring Criteria

### issue_text_score (1-5)
1 = Incomprehensible, image-only, or empty
2 = Vague with missing context, hard to act on
3 = Understandable but missing some details
4 = Clear with good description, mostly self-contained
5 = Excellent: clear problem, reproduction steps, expected behavior

### test_score (1-5)
1 = No meaningful assertions, trivial test
2 = Tests exist but weak coverage of the fix
3 = Decent tests, cover the main case
4 = Good tests with edge cases
5 = Comprehensive tests with assertions, edge cases, error handling

### difficulty_score (1-5)
1 = Trivial (typo, one-line change, obvious fix)
2 = Easy (localized fix, clear from error message)
3 = Medium (requires understanding module interactions)
4 = Hard (requires deep domain knowledge or multi-file changes)
5 = Very hard (architectural changes, complex algorithms)

Return ONLY a JSON object with these three fields:
```json
{{"issue_text_score": N, "test_score": N, "difficulty_score": N}}
```"""


def _truncate_patch(patch: str, max_lines: int = 50) -> str:
    """Truncate a patch for LLM context, keeping file headers and key changes."""
    lines = patch.split("\n")
    if len(lines) <= max_lines:
        return patch

    # Keep file headers and first few hunks
    result = []
    in_hunk = False
    hunk_lines = 0
    for line in lines:
        if line.startswith("diff --git"):
            result.append(line)
            in_hunk = False
            hunk_lines = 0
        elif line.startswith("@@"):
            result.append(line)
            in_hunk = True
            hunk_lines = 0
        elif in_hunk:
            hunk_lines += 1
            if hunk_lines <= 20:
                result.append(line)
            elif hunk_lines == 21:
                result.append("... (truncated)")
                in_hunk = False

        if len(result) >= max_lines:
            result.append("... (truncated)")
            break

    return "\n".join(result)


def _call_llm(prompt: str) -> str | None:
    """Call an LLM via the Cursor agent CLI.

    Requires the ``agent`` command to be available on PATH (installed with
    Cursor IDE).

    Uses a scratch directory as ``--workspace`` and ``cwd`` so that any files
    the agent creates as a side-effect land there instead of the project root.
    """
    from .paths import LLM_SCRATCH_DIR

    scratch = LLM_SCRATCH_DIR
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["agent", "-p", prompt, "--output-format", "text",
             "--workspace", str(scratch)],
            capture_output=True, text=True, timeout=60,
            cwd=scratch,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except FileNotFoundError:
        log.warning("Cursor agent CLI ('agent') not found on PATH – install Cursor IDE first")
    except subprocess.TimeoutExpired:
        log.warning("Cursor agent CLI timed out after 60 s")

    return None


def _extract_json_from_text(text: str) -> dict | None:
    """Extract a JSON object from LLM output."""
    if not text:
        return None

    # Try code blocks first
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue

    # Try raw JSON
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

    return None


_GENERATE_PROBLEM_PROMPT = """\
You are analyzing a merged pull request to write a bug report / problem statement.
Given the PR metadata and diff below, write a clear problem statement describing
what was wrong BEFORE this fix was applied. Write it as if you are filing a bug
report -- describe the symptoms, not the solution.

## PR Title
{pr_title}

## PR Body
{pr_body}

## Diff (solution patch)
{solution_patch}

Write a concise problem statement (2-6 sentences). Describe:
- What component/function is affected
- What the incorrect behavior is
- Expected vs actual behavior if inferable

Do NOT reveal the solution or mention the PR. Write as a standalone bug report.
"""


def generate_problem_statement(
    pr_title: str,
    pr_body: str,
    solution_patch: str,
) -> str:
    """Synthesize a problem statement from PR metadata and diff using an LLM.

    When a PR has no linked issue (or the issue body is too short), we can
    still create a useful problem statement by having an LLM read the diff
    and describe what was broken.

    Falls back to "PR title + PR body" if LLM is unavailable.
    """
    pr_title = pr_title or ""
    pr_body = pr_body or ""
    solution_patch = solution_patch or ""

    prompt = _GENERATE_PROBLEM_PROMPT.format(
        pr_title=pr_title[:500],
        pr_body=pr_body[:2000],
        solution_patch=_truncate_patch(solution_patch),
    )

    output = _call_llm(prompt)
    if output and len(output.strip()) > 20:
        log.info("  [quality] Generated problem statement via LLM (%d chars)", len(output.strip()))
        return output.strip()

    # Fallback: compose from PR title + body
    log.debug("  [quality] LLM unavailable for problem generation, using PR title+body")
    fallback = pr_title
    if pr_body.strip():
        fallback += "\n\n" + pr_body.strip()
    return fallback


def score_with_llm(
    problem_statement: str,
    solution_patch: str,
    test_patch: str,
) -> QualityScores:
    """Score a task using an LLM.

    Falls back to heuristic scoring if LLM is unavailable.
    """
    prompt = _SCORE_PROMPT.format(
        problem_statement=problem_statement[:3000],
        solution_patch_summary=_truncate_patch(solution_patch),
        test_patch_summary=_truncate_patch(test_patch),
    )

    output = _call_llm(prompt)
    if output:
        parsed = _extract_json_from_text(output)
        if parsed:
            return QualityScores(
                issue_text_score=_clamp(parsed.get("issue_text_score"), 1, 5),
                test_score=_clamp(parsed.get("test_score"), 1, 5),
                difficulty_score=_clamp(parsed.get("difficulty_score"), 1, 5),
            )

    log.debug("LLM scoring unavailable, falling back to heuristic scoring")
    return _heuristic_score(problem_statement, solution_patch, test_patch)


def _clamp(val, lo, hi) -> int | None:
    """Clamp a value to [lo, hi], return None if not an int."""
    if val is None:
        return None
    try:
        v = int(val)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Heuristic scoring (fallback when no LLM is available)
# ---------------------------------------------------------------------------


def _heuristic_score(
    problem_statement: str,
    solution_patch: str,
    test_patch: str,
) -> QualityScores:
    """Compute approximate quality scores using heuristics."""
    scores = QualityScores()

    # --- Issue text score ---
    ps_len = len(problem_statement)
    prose = _CODE_BLOCK_PATTERN.sub("", problem_statement)
    prose_len = len(prose.strip())

    if ps_len < 30:
        scores.issue_text_score = 1
    elif ps_len < 100:
        scores.issue_text_score = 2
    elif prose_len < 200:
        scores.issue_text_score = 3
    elif prose_len < 500:
        scores.issue_text_score = 4
    else:
        # Check for structure (headers, steps, expected behavior)
        has_structure = bool(re.search(r"##|steps|expected|actual|reproduce", problem_statement, re.IGNORECASE))
        scores.issue_text_score = 5 if has_structure else 4

    # --- Test score ---
    test_lines = [l for l in test_patch.split("\n") if l.startswith("+") and not l.startswith("+++")]
    assert_count = sum(1 for l in test_lines if "assert" in l.lower() or "raises" in l.lower())
    test_func_count = sum(1 for l in test_lines if re.match(r"\+\s*(def test_|async def test_)", l))

    if assert_count == 0:
        scores.test_score = 1
    elif assert_count <= 2 and test_func_count <= 1:
        scores.test_score = 2
    elif assert_count <= 5:
        scores.test_score = 3
    elif assert_count <= 10:
        scores.test_score = 4
    else:
        scores.test_score = 5

    # --- Difficulty score ---
    added = len(re.findall(r"^\+[^+]", solution_patch, re.MULTILINE))
    removed = len(re.findall(r"^-[^-]", solution_patch, re.MULTILINE))
    total_changed = added + removed
    files_changed = len(re.findall(r"^diff --git", solution_patch, re.MULTILINE))

    if total_changed <= 5:
        scores.difficulty_score = 1
    elif total_changed <= 20 and files_changed == 1:
        scores.difficulty_score = 2
    elif total_changed <= 50 and files_changed <= 3:
        scores.difficulty_score = 3
    elif total_changed <= 200:
        scores.difficulty_score = 4
    else:
        scores.difficulty_score = 5

    return scores


# ---------------------------------------------------------------------------
# Main entry point: full quality assessment
# ---------------------------------------------------------------------------


def assess_quality(task: dict, *, min_quality_score: int = 2) -> QualityScores:
    """Full quality assessment of a task candidate.

    Combines heuristic pre-filtering with LLM scoring.
    Also cleans the problem statement.

    Args:
        task: Task dict with problem_statement, patch, test_patch.
        min_quality_score: Minimum score (1-5) for issue_text and test dimensions.
    """
    problem_statement = task.get("problem_statement") or ""
    solution_patch = task.get("patch") or ""
    test_patch = task.get("test_patch") or ""

    # Extract title and body from problem statement
    parts = problem_statement.split("\n\n", 1)
    title = parts[0] if parts else ""
    body = parts[1] if len(parts) > 1 else ""

    # --- Heuristic pre-filters (cheap) ---
    rejection = heuristic_issue_prefilter(title, body)
    if rejection:
        log.info("  [quality] Rejected by issue pre-filter: %s", rejection)
        return QualityScores(
            issue_text_score=1,
            rejection_reason=rejection,
            _min_quality_score=min_quality_score,
        )

    rejection = heuristic_patch_quality(solution_patch, test_patch)
    if rejection:
        log.info("  [quality] Rejected by patch pre-filter: %s", rejection)
        return QualityScores(
            test_score=1,
            rejection_reason=rejection,
            _min_quality_score=min_quality_score,
        )

    # --- Clean problem statement ---
    cleaned = clean_problem_statement(problem_statement)

    # --- LLM scoring (or heuristic fallback) ---
    scores = score_with_llm(cleaned, solution_patch, test_patch)
    scores.cleaned_problem_statement = cleaned
    scores._min_quality_score = min_quality_score

    if not scores.passes_threshold:
        scores.rejection_reason = (
            f"Below quality threshold (min={min_quality_score}): "
            f"issue={scores.issue_text_score}, "
            f"test={scores.test_score}, difficulty={scores.difficulty_score}"
        )
        log.info("  [quality] %s", scores.rejection_reason)

    return scores
