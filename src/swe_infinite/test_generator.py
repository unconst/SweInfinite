"""
LLM-based test generation for PRs that lack test changes.

When a merged PR only contains solution (non-test) changes, this module
generates appropriate test code via an LLM, validates it bidirectionally
(FAIL on base, PASS on base+solution), and returns a proper unified diff
that can serve as the test_patch.

Supports: Python, TypeScript/JavaScript, Java, Go.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import textwrap
from pathlib import Path

log = logging.getLogger("swe-infinite.test_generator")

# ---------------------------------------------------------------------------
# LLM call (reuses quality_scorer's multi-backend approach)
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, max_tokens: int = 4096) -> str | None:
    """Call an LLM via available backends (Anthropic, OpenAI, agent CLI)."""
    # Try Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import httpx

            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                try:
                    return data["content"][0]["text"]
                except (KeyError, IndexError, TypeError):
                    log.debug("Anthropic API returned unexpected structure")
        except Exception as e:
            log.debug("Anthropic API failed: %s", e)

    # Try OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            import httpx

            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    log.debug("OpenAI API returned unexpected structure")
        except Exception as e:
            log.debug("OpenAI API failed: %s", e)

    # Fallback: Cursor agent CLI
    try:
        result = subprocess.run(
            ["cursor", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    log.warning("No LLM backend available for test generation")
    return None


# ---------------------------------------------------------------------------
# Code extraction from LLM response
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(response: str) -> str | None:
    """Extract code from an LLM response (handles markdown code blocks)."""
    if not response:
        return None

    # Try code blocks first
    m = _CODE_BLOCK_RE.search(response)
    if m:
        return m.group(1).strip()

    # If response looks like raw code (has imports and test functions)
    if ("import " in response or "func Test" in response) and (
        "def test_" in response
        or "it(" in response
        or "describe(" in response
        or "func Test" in response
        or "@Test" in response
    ):
        return response.strip()

    return None


# ---------------------------------------------------------------------------
# Language-specific prompts
# ---------------------------------------------------------------------------

_PYTHON_PROMPT = textwrap.dedent("""\
    You are generating a pytest test for a bug fix in a Python project.

    Repository: {repo_name}

    The following diff shows the fix that was applied. Your job is to write a test
    that FAILS before this fix (on the original/broken code) and PASSES after the
    fix is applied.

    ## Solution Diff
    {solution_patch}

    ## Changed Source Files (post-fix)
    {source_context}

    Write a minimal, self-contained pytest test file that:
    1. Imports from the project (use the actual module paths visible in the diff)
    2. Tests the specific behavior that was fixed/changed
    3. Would FAIL on the code BEFORE the fix (the original buggy version)
    4. Would PASS on the code AFTER the fix is applied
    5. Has meaningful assertions that verify the correct behavior

    Hard rules:
    - Use only pytest (no unittest, no nose)
    - No external dependencies beyond the project itself and pytest
    - No network calls, no real file I/O, no sleep, no randomness
    - Test MUST be fully deterministic
    - Keep it focused: 1-3 test functions covering the fix
    - Use descriptive test names: test_<what_it_verifies>
    - Do NOT mock the function under test itself
    - Make sure imports match the actual project structure

    Return ONLY the Python test code. Start with imports. No markdown, no explanation.
""")

_TYPESCRIPT_PROMPT = textwrap.dedent("""\
    You are generating a test for a bug fix in a TypeScript/JavaScript project.

    Repository: {repo_name}

    The following diff shows the fix that was applied. Your job is to write a test
    that FAILS before this fix and PASSES after the fix is applied.

    ## Solution Diff
    {solution_patch}

    ## Changed Source Files (post-fix)
    {source_context}

    Write a minimal test file that:
    1. Imports from the project using the actual module paths
    2. Tests the specific behavior that was fixed/changed
    3. Would FAIL on the code BEFORE the fix
    4. Would PASS on the code AFTER the fix
    5. Has meaningful assertions

    Hard rules:
    - Use jest or vitest (whichever is in the project)
    - No external dependencies beyond the project + test framework
    - No network calls, no real file I/O, no timers
    - Test MUST be deterministic
    - 1-3 focused test cases
    - Use descriptive test names

    Return ONLY the test code. No markdown, no explanation.
""")

_JAVA_PROMPT = textwrap.dedent("""\
    You are generating a JUnit test for a bug fix in a Java project.

    Repository: {repo_name}

    The following diff shows the fix that was applied. Your job is to write a test
    that FAILS before this fix and PASSES after the fix is applied.

    ## Solution Diff
    {solution_patch}

    ## Changed Source Files (post-fix)
    {source_context}

    Write a minimal JUnit test class that:
    1. Imports the classes being tested
    2. Tests the specific behavior that was fixed/changed
    3. Would FAIL on the code BEFORE the fix
    4. Would PASS on the code AFTER the fix
    5. Has meaningful assertions

    Hard rules:
    - Use JUnit 5 (jupiter) with standard assertions
    - No external dependencies beyond the project + JUnit
    - No network calls, no real file I/O
    - Test MUST be deterministic
    - 1-3 focused test methods
    - Use descriptive method names

    Return ONLY the Java test code. No markdown, no explanation.
""")

_GO_PROMPT = textwrap.dedent("""\
    You are generating a Go test for a bug fix in a Go project.

    Repository: {repo_name}

    The following diff shows the fix that was applied. Your job is to write a test
    that FAILS before this fix and PASSES after the fix is applied.

    ## Solution Diff
    {solution_patch}

    ## Changed Source Files (post-fix)
    {source_context}

    Write a minimal Go test file that:
    1. Uses the correct package name matching the source files
    2. Tests the specific behavior that was fixed/changed
    3. Would FAIL on the code BEFORE the fix
    4. Would PASS on the code AFTER the fix
    5. Has meaningful assertions using t.Errorf/t.Fatalf

    Hard rules:
    - Use standard testing package only
    - No external dependencies
    - No network calls, no real file I/O
    - Test MUST be deterministic
    - 1-3 focused test functions
    - Use descriptive function names: TestXxx

    Return ONLY the Go test code. No markdown, no explanation.
""")

_PROMPTS = {
    "python": _PYTHON_PROMPT,
    "typescript": _TYPESCRIPT_PROMPT,
    "javascript": _TYPESCRIPT_PROMPT,
    "java": _JAVA_PROMPT,
    "go": _GO_PROMPT,
}


# ---------------------------------------------------------------------------
# Test file path generation
# ---------------------------------------------------------------------------

_TEST_FILE_TEMPLATES = {
    "python": "tests/test_generated_{slug}.py",
    "typescript": "__tests__/generated_{slug}.test.ts",
    "javascript": "__tests__/generated_{slug}.test.js",
    "java": "src/test/java/generated/Test{slug_camel}.java",
    "go": "{pkg_dir}/generated_{slug}_test.go",
}


def _make_test_path(language: str, solution_patch: str) -> str:
    """Generate an appropriate test file path based on language and changed files."""
    # Extract first changed file to determine location
    file_match = re.search(r"^diff --git a/(.+?) b/", solution_patch, re.MULTILINE)
    first_file = file_match.group(1) if file_match else "unknown"

    # Create a slug from the first changed file
    slug = re.sub(r"[^a-zA-Z0-9]", "_", Path(first_file).stem)[:30]
    slug_camel = slug.replace("_", " ").title().replace(" ", "")

    # For Go, put the test in the same package directory
    pkg_dir = str(Path(first_file).parent) if first_file != "unknown" else "."

    template = _TEST_FILE_TEMPLATES.get(language, _TEST_FILE_TEMPLATES["python"])
    return template.format(slug=slug, slug_camel=slug_camel, pkg_dir=pkg_dir)


# ---------------------------------------------------------------------------
# Source context extraction
# ---------------------------------------------------------------------------


def _get_source_context(
    repo_dir: Path,
    solution_patch: str,
    max_lines: int = 200,
) -> str:
    """Read the changed source files (post-fix state) for LLM context.

    Returns a concatenated view of the changed files with their paths.
    """
    # Extract file paths from the solution patch
    file_paths = re.findall(r"^diff --git a/.+? b/(.+?)$", solution_patch, re.MULTILINE)

    context_parts = []
    total_lines = 0

    for fpath in file_paths:
        full_path = repo_dir / fpath
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(errors="replace")
        except OSError:
            continue

        lines = content.splitlines()
        if total_lines + len(lines) > max_lines:
            # Truncate
            remaining = max_lines - total_lines
            if remaining > 10:
                context_parts.append(f"### {fpath} (truncated)")
                context_parts.append("\n".join(lines[:remaining]))
                context_parts.append("... (truncated)")
            break

        context_parts.append(f"### {fpath}")
        context_parts.append(content)
        total_lines += len(lines)

    return "\n\n".join(context_parts) if context_parts else "(source files not available)"


# ---------------------------------------------------------------------------
# Unified diff construction
# ---------------------------------------------------------------------------


def _make_new_file_diff(file_path: str, content: str) -> str:
    """Create a unified diff for a brand-new file (git apply compatible)."""
    lines = content.splitlines()
    n = len(lines)

    diff_lines = [
        f"diff --git a/{file_path} b/{file_path}",
        "new file mode 100644",
        "--- /dev/null",
        f"+++ b/{file_path}",
        f"@@ -0,0 +1,{n} @@",
    ]
    for line in lines:
        diff_lines.append(f"+{line}")

    return "\n".join(diff_lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_test_patch(
    repo_dir: Path,
    solution_patch: str,
    language: str,
    repo_name: str,
    pr_title: str = "",
    pr_body: str = "",
    max_retries: int = 2,
) -> str | None:
    """Generate a test patch for a PR that has no test changes.

    Uses an LLM to generate test code based on the solution patch and
    source files, then formats it as a unified diff.

    Args:
        repo_dir: Path to the cloned repository (checked out at base_commit).
        solution_patch: The solution (non-test) unified diff.
        language: Programming language (python, typescript, java, go).
        repo_name: Repository name (owner/repo).
        pr_title: Pull request title (for context).
        pr_body: Pull request body (for context).
        max_retries: Number of LLM attempts before giving up.

    Returns:
        A unified diff string for the generated test file, or None on failure.
    """
    language = language.lower()

    prompt_template = _PROMPTS.get(language)
    if not prompt_template:
        log.info("  [test-gen] No test generation prompt for language: %s", language)
        return None

    # Truncate the solution patch for the prompt
    patch_for_prompt = solution_patch
    if len(patch_for_prompt) > 6000:
        patch_for_prompt = patch_for_prompt[:6000] + "\n... (truncated)"

    # Get source context (the changed files in their post-fix state)
    # We need to temporarily apply the solution patch to read the fixed files
    source_context = _get_source_context(repo_dir, solution_patch)

    prompt = prompt_template.format(
        repo_name=repo_name,
        solution_patch=patch_for_prompt,
        source_context=source_context,
    )

    # Add PR context if available
    if pr_title or pr_body:
        extra_context = "\n## PR Context\n"
        if pr_title:
            extra_context += f"Title: {pr_title}\n"
        if pr_body:
            extra_context += f"Body: {pr_body[:1000]}\n"
        prompt = prompt + extra_context

    for attempt in range(1, max_retries + 1):
        log.info("  [test-gen] Generating test (attempt %d/%d) for %s...",
                 attempt, max_retries, language)

        response = _call_llm(prompt)
        if not response:
            log.warning("  [test-gen] LLM returned no response (attempt %d)", attempt)
            continue

        test_code = _extract_code(response)
        if not test_code:
            log.warning("  [test-gen] Could not extract code from LLM response (attempt %d)", attempt)
            continue

        # Basic validation of the generated code
        if not _basic_validate(test_code, language):
            log.warning("  [test-gen] Generated code failed basic validation (attempt %d)", attempt)
            continue

        # Determine the test file path
        test_path = _make_test_path(language, solution_patch)

        # Create the unified diff
        test_patch = _make_new_file_diff(test_path, test_code)

        log.info("  [test-gen] Generated test: %s (%d lines of test code)",
                 test_path, len(test_code.splitlines()))

        return test_patch

    log.warning("  [test-gen] Failed to generate test after %d attempts", max_retries)
    return None


# ---------------------------------------------------------------------------
# Basic code validation
# ---------------------------------------------------------------------------


def _basic_validate(test_code: str, language: str) -> bool:
    """Quick sanity checks on generated test code."""
    if not test_code or len(test_code.strip()) < 20:
        return False

    if language == "python":
        # Must have at least one test function
        if "def test_" not in test_code:
            return False
        # Must have at least one assert
        if "assert" not in test_code.lower() and "pytest.raises" not in test_code:
            return False
        # Must parse as valid Python
        try:
            import ast
            ast.parse(test_code)
        except SyntaxError:
            return False

    elif language in ("typescript", "javascript"):
        # Must have test/it/describe blocks
        if not any(kw in test_code for kw in ("it(", "test(", "describe(")):
            return False
        # Must have assertions
        if not any(kw in test_code for kw in ("expect(", "assert", "toBe", "toEqual")):
            return False

    elif language == "java":
        # Must have @Test annotation
        if "@Test" not in test_code:
            return False
        # Must have assertions
        if not any(kw in test_code for kw in ("assert", "Assert", "verify")):
            return False

    elif language == "go":
        # Must have Test functions
        if "func Test" not in test_code:
            return False
        # Must have testing package
        if '"testing"' not in test_code:
            return False

    return True
