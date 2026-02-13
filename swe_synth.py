#!/usr/bin/env python3
"""
SWE-Synth — Self-validating SWE benchmark task generator.

Generates infinite coding challenges with guaranteed clean reward signal by:
1. Cloning a known-good repo
2. Identifying destroyable features (functions/methods) via AST
3. Generating tests via LLM that verify the feature works
4. Bidirectional validation: test PASSES on original, FAILS on broken
5. Storing valid tasks as JSON

The invariant:
    Original Repo + Generated Test → PASS
    Broken  Repo + Generated Test → FAIL

Only tasks satisfying BOTH conditions are accepted.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    # Generate up to 5 tasks (default)
    uv run python swe_synth.py

    # Generate more, with verbose output
    uv run python swe_synth.py --max-tasks 20 -v

    # List all destroyable targets without generating
    uv run python swe_synth.py --list-targets

    # Different random ordering
    uv run python swe_synth.py --seed 123

    # Resume (skips already-generated task IDs)
    uv run python swe_synth.py --max-tasks 50
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ─── Configuration ────────────────────────────────────────────────────────

REPO_URL = "https://github.com/pallets/click.git"
REPO_NAME = "pallets/click"
BASE_TAG = "8.1.7"

WORK_DIR = Path("./synth_workdir")
TASKS_DIR = Path("./synth_tasks")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Minimum lines in a function body to be considered a target
MIN_BODY_LINES = 3

log = logging.getLogger("swe-synth")


# ─── Data structures ─────────────────────────────────────────────────────


@dataclass
class Target:
    """A function/method that can be destroyed."""

    file_path: str  # relative to repo root
    name: str
    class_name: str | None
    lineno: int
    end_lineno: int
    source: str

    @property
    def qualified_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name

    @property
    def uid(self) -> str:
        """Short unique hash based on file + name + source."""
        h = hashlib.sha256(f"{self.file_path}:{self.qualified_name}:{self.source}".encode())
        return h.hexdigest()[:10]


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Repository setup
# ═══════════════════════════════════════════════════════════════════════════


def clone_repo(work_dir: Path) -> Path:
    """Clone the repo and checkout the base tag. Reuses existing clone."""
    repo_dir = work_dir / "repo"

    if repo_dir.exists():
        log.info("Repo already exists at %s — resetting to %s", repo_dir, BASE_TAG)
        subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)
        result = subprocess.run(
            ["git", "checkout", BASE_TAG],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.warning("git checkout %s failed: %s", BASE_TAG, result.stderr.strip())
            # Try as a branch
            subprocess.run(
                ["git", "checkout", "-b", f"synth-{BASE_TAG}", BASE_TAG],
                cwd=repo_dir,
                capture_output=True,
            )
        return repo_dir

    log.info("Cloning %s (tag %s)...", REPO_URL, BASE_TAG)
    subprocess.run(
        ["git", "clone", REPO_URL, str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["git", "checkout", BASE_TAG],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("Failed to checkout tag %s: %s", BASE_TAG, result.stderr.strip())
        # List available tags for debugging
        tags = subprocess.run(
            ["git", "tag", "--list", "8.*"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        log.info("Available tags: %s", tags.stdout.strip())
        sys.exit(1)

    return repo_dir


def setup_venv(repo_dir: Path) -> Path:
    """Create venv with uv and install the package + pytest."""
    venv_dir = repo_dir.parent / ".venv"
    python = venv_dir / "bin" / "python"

    if venv_dir.exists() and python.exists():
        log.info("Venv already exists at %s", venv_dir)
        return venv_dir

    log.info("Creating venv with uv...")
    subprocess.run(["uv", "venv", str(venv_dir)], check=True, capture_output=True)

    log.info("Installing package (editable) + pytest...")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "-e", ".", "pytest"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    return venv_dir


def get_python(venv_dir: Path) -> str:
    """Path to the venv's Python interpreter."""
    return str(venv_dir / "bin" / "python")


def get_base_commit(repo_dir: Path) -> str:
    """Get the HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def restore_repo(repo_dir: Path) -> None:
    """Restore repo to clean state via git."""
    subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Feature detection (AST-based)
# ═══════════════════════════════════════════════════════════════════════════


class _ParentAnnotator(ast.NodeVisitor):
    """Annotate each AST node with its parent, so we can find class context."""

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]
        super().generic_visit(node)


def find_targets(repo_dir: Path) -> list[Target]:
    """Find all public functions/methods suitable for destruction."""
    targets: list[Target] = []

    # Click uses src/ layout
    src_dir = repo_dir / "src"
    if not src_dir.exists():
        src_dir = repo_dir

    skip_patterns = {"test", "setup.py", "__pycache__", ".git", "_compat", "conftest"}

    for py_file in sorted(src_dir.rglob("*.py")):
        rel_path = str(py_file.relative_to(repo_dir))

        # Skip non-source files
        if any(skip in rel_path for skip in skip_patterns):
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Annotate parents
        _ParentAnnotator().visit(tree)

        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Skip private/dunder methods
            if node.name.startswith("_"):
                continue

            # Skip tiny functions
            end = node.end_lineno or node.lineno
            if end - node.lineno < MIN_BODY_LINES:
                continue

            # Skip functions whose body is just `pass` or `...`
            if len(node.body) == 1:
                stmt = node.body[0]
                if isinstance(stmt, ast.Pass):
                    continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if stmt.value.value is ...:
                        continue

            # Determine parent class (if method)
            parent = getattr(node, "_parent", None)
            class_name = parent.name if isinstance(parent, ast.ClassDef) else None

            # Extract source lines
            func_source = "\n".join(lines[node.lineno - 1 : end])

            targets.append(
                Target(
                    file_path=rel_path,
                    name=node.name,
                    class_name=class_name,
                    lineno=node.lineno,
                    end_lineno=end,
                    source=func_source,
                )
            )

    return targets


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Destructive operator
# ═══════════════════════════════════════════════════════════════════════════


def apply_destruction(repo_dir: Path, target: Target) -> str:
    """
    Replace the function body with `raise NotImplementedError`.
    Returns the unified diff (patch).
    """
    file_path = repo_dir / target.file_path
    source = file_path.read_text()
    lines = source.splitlines(keepends=True)

    # Re-parse to locate the function precisely
    tree = ast.parse(source)
    func_node = None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == target.name and node.lineno == target.lineno:
                func_node = node
                break

    if not func_node or not func_node.body:
        raise ValueError(f"Cannot find {target.name} at line {target.lineno} in {target.file_path}")

    # Body starts at the first statement, ends at end_lineno
    body_start = func_node.body[0].lineno - 1  # 0-indexed
    body_end = func_node.end_lineno  # 1-indexed → use as exclusive end

    # Detect indentation from the first body line
    first_body_line = lines[body_start]
    indent = ""
    for ch in first_body_line:
        if ch in (" ", "\t"):
            indent += ch
        else:
            break

    # Replacement
    replacement = f'{indent}raise NotImplementedError("Feature removed for SWE-Synth")\n'
    new_lines = lines[:body_start] + [replacement] + lines[body_end:]

    # Generate diff before writing
    patch = "".join(
        difflib.unified_diff(
            lines,
            new_lines,
            fromfile=f"a/{target.file_path}",
            tofile=f"b/{target.file_path}",
        )
    )

    # Write the destroyed version
    file_path.write_text("".join(new_lines))

    return patch


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Test generation (LLM)
# ═══════════════════════════════════════════════════════════════════════════

_TEST_GEN_PROMPT = textwrap.dedent("""\
    You are generating a pytest test for a Python function in the `{repo_name}` package.

    File: {file_path}
    Function: {qualified_name}

    Here is the full function source code:

    ```python
    {source}
    ```

    Write a minimal, self-contained pytest test that:
    1. Imports from the public package API (e.g. `import click` or `from click import ...`)
    2. Calls the function/method with valid inputs
    3. Asserts specific expected behavior (return values, side effects, output)
    4. Would FAIL if the function body was replaced with `raise NotImplementedError`

    Hard rules:
    - Use only pytest (no unittest, no nose)
    - No external dependencies beyond the package itself and pytest
    - No network calls, no file I/O to real filesystem, no sleep, no randomness
    - Test MUST be fully deterministic
    - Keep it minimal: 1-3 test functions, each focused on one behavior
    - Use descriptive test names: test_<what_it_verifies>
    - If testing a method on a class, construct the class with minimal valid arguments
    - If the function is a decorator, test applying it and calling the result
    - Prefer testing observable behavior over implementation details
    - Do NOT mock the function under test itself

    Return ONLY the Python test code. Start with imports. No markdown, no explanation.
""")


def call_llm(prompt: str) -> str:
    """Call Anthropic Messages API and return the text response."""
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def generate_test(target: Target) -> str | None:
    """Ask the LLM to generate a pytest test for the target function."""
    prompt = _TEST_GEN_PROMPT.format(
        repo_name=REPO_NAME,
        file_path=target.file_path,
        qualified_name=target.qualified_name,
        source=target.source,
    )

    try:
        response = call_llm(prompt)
    except httpx.HTTPStatusError as e:
        log.error("  LLM API error %d: %s", e.response.status_code, e.response.text[:200])
        return None
    except Exception as e:
        log.error("  LLM call failed: %s", e)
        return None

    # Extract code from markdown blocks if present
    m = _CODE_BLOCK_RE.search(response)
    if m:
        return m.group(1).strip()

    # If no code block, check if it looks like raw Python
    if "import" in response and "def test_" in response:
        return response.strip()

    log.warning("  LLM response did not contain valid test code")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Test runner
# ═══════════════════════════════════════════════════════════════════════════


def run_test(
    repo_dir: Path,
    venv_dir: Path,
    test_code: str,
    timeout: int = 60,
) -> tuple[bool, str]:
    """
    Write test to a temp file in the repo, run pytest, return (passed, output).
    Cleans up the test file afterwards.
    """
    test_file = repo_dir / "_synth_test.py"
    test_file.write_text(test_code)

    try:
        python = get_python(venv_dir)
        result = subprocess.run(
            [
                python,
                "-m",
                "pytest",
                str(test_file),
                "-x",  # stop on first failure
                "--tb=short",
                "--no-header",
                "-q",
                "-W",
                "ignore::DeprecationWarning",
            ],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        passed = result.returncode == 0
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (exceeded {}s)".format(timeout)
    finally:
        test_file.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Bidirectional validation (THE CRITICAL INVARIANT)
# ═══════════════════════════════════════════════════════════════════════════


def validate_bidirectional(
    repo_dir: Path,
    venv_dir: Path,
    target: Target,
    test_code: str,
) -> tuple[bool, str]:
    """
    The core invariant:
        Original + Test → PASS
        Broken   + Test → FAIL

    Returns (valid, patch) where patch is the destructive diff.
    If invalid, patch is empty.
    """
    # ── A. Test on ORIGINAL (must PASS) ──────────────────────────────────
    log.info("  [A] Running test on ORIGINAL repo...")
    passed_original, output_orig = run_test(repo_dir, venv_dir, test_code)

    if not passed_original:
        log.info("  ✗ REJECT — test FAILS on original (bad test)")
        log.debug("  Output:\n%s", output_orig[:500])
        return False, ""

    log.info("  ✓ Test PASSES on original")

    # ── B. Apply destruction ─────────────────────────────────────────────
    log.info("  [B] Applying destruction: %s → NotImplementedError", target.qualified_name)
    try:
        patch = apply_destruction(repo_dir, target)
    except Exception as e:
        log.error("  ✗ Failed to apply destruction: %s", e)
        restore_repo(repo_dir)
        return False, ""

    # ── C. Test on BROKEN (must FAIL) ────────────────────────────────────
    log.info("  [C] Running test on BROKEN repo...")
    passed_broken, output_broken = run_test(repo_dir, venv_dir, test_code)

    # ── D. Restore repo ─────────────────────────────────────────────────
    restore_repo(repo_dir)

    if passed_broken:
        log.info("  ✗ REJECT — test PASSES on broken (test doesn't catch the destruction)")
        log.debug("  Output:\n%s", output_broken[:500])
        return False, ""

    log.info("  ✓ Test FAILS on broken — BIDIRECTIONAL VALIDATION PASSED")
    return True, patch


# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — Static validation (additional hardening)
# ═══════════════════════════════════════════════════════════════════════════


def static_validate_test(test_code: str) -> tuple[bool, str]:
    """
    Reject tests that are likely flaky or invalid.
    Returns (ok, reason).
    """
    # Must parse as valid Python
    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # Must have at least one test function
    test_funcs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    if not test_funcs:
        return False, "No test_* functions found"

    code_lower = test_code.lower()

    # Reject if it uses sleep
    if "time.sleep" in test_code or "asyncio.sleep" in test_code:
        return False, "Uses sleep (non-deterministic)"

    # Reject if it uses network
    for net_kw in ["requests.get", "requests.post", "urllib.request", "httpx.", "socket."]:
        if net_kw in test_code:
            return False, f"Uses network ({net_kw})"

    # Reject if it uses random
    if "random." in test_code and "random.seed" not in test_code:
        return False, "Uses random without seed"

    # Reject if it writes files (basic check)
    for write_kw in ["open(", "write_text(", "write_bytes(", "os.makedirs", "shutil."]:
        if write_kw in test_code:
            # Allow StringIO/BytesIO
            if "StringIO" in test_code or "BytesIO" in test_code:
                continue
            # Click's CliRunner uses isolated_filesystem which is OK
            if "isolated_filesystem" in test_code or "CliRunner" in test_code:
                continue
            return False, f"May write to filesystem ({write_kw})"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════════════


def make_task_id(target: Target, base_commit: str) -> str:
    """Deterministic task ID from target + commit."""
    repo_slug = REPO_NAME.replace("/", "__")
    return f"{repo_slug}__{target.name}__{target.uid}__{base_commit[:8]}"


def store_task(
    task_id: str,
    target: Target,
    base_commit: str,
    patch: str,
    test_code: str,
    tasks_dir: Path,
) -> Path:
    """Store a validated task as JSON."""
    tasks_dir.mkdir(parents=True, exist_ok=True)
    path = tasks_dir / f"{task_id}.json"

    data = {
        "task_id": task_id,
        "repo": REPO_NAME,
        "repo_url": REPO_URL,
        "base_commit": base_commit,
        "base_tag": BASE_TAG,
        # Target info
        "target_function": target.qualified_name,
        "target_file": target.file_path,
        "target_lineno": target.lineno,
        "target_end_lineno": target.end_lineno,
        "original_code": target.source,
        # Patches
        "destructive_patch": patch,
        "test_code": test_code,
        # Problem statement (what the solver sees)
        "problem_statement": (
            f"The function `{target.qualified_name}` in `{target.file_path}` has been broken. "
            f"It now raises NotImplementedError instead of performing its intended behavior. "
            f"Restore the original functionality so that the provided test passes."
        ),
        # Metadata
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    path.write_text(json.dumps(data, indent=2))
    return path


def load_existing_task_ids(tasks_dir: Path) -> set[str]:
    """Load task IDs that have already been generated (for resume support)."""
    if not tasks_dir.exists():
        return set()
    ids = set()
    for f in tasks_dir.glob("*.json"):
        ids.add(f.stem)
    return ids


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-Synth: Self-validating SWE benchmark task generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Environment variables:
              ANTHROPIC_API_KEY   Required. Your Anthropic API key.
              ANTHROPIC_MODEL     Model to use (default: claude-sonnet-4-20250514)

            Examples:
              uv run python swe_synth.py --list-targets
              uv run python swe_synth.py --max-tasks 10 --seed 42
              uv run python swe_synth.py --max-tasks 50 -v
        """),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=5,
        help="Maximum number of valid tasks to generate (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for target selection order (default: 42)",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List all destroyable targets and exit (no LLM calls)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=WORK_DIR,
        help=f"Working directory for repo clone + venv (default: {WORK_DIR})",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=TASKS_DIR,
        help=f"Output directory for validated tasks (default: {TASKS_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Preflight checks ────────────────────────────────────────────────
    if not args.list_targets and not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY environment variable is required.")
        log.error("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    random.seed(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Clone + setup ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SWE-Synth — Self-validating task generator")
    log.info("Repo: %s @ %s", REPO_NAME, BASE_TAG)
    log.info("=" * 60)

    repo_dir = clone_repo(args.work_dir)
    base_commit = get_base_commit(repo_dir)
    log.info("Base commit: %s", base_commit)

    if not args.list_targets:
        venv_dir = setup_venv(repo_dir)
        log.info("Venv: %s", venv_dir)
    else:
        venv_dir = None  # not needed for listing

    # ── Step 2: Find destroyable targets ─────────────────────────────────
    targets = find_targets(repo_dir)
    log.info("Found %d destroyable targets", len(targets))

    if args.list_targets:
        print(f"\nDestroyable targets in {REPO_NAME} @ {BASE_TAG}:")
        print("-" * 70)
        for t in sorted(targets, key=lambda t: (t.file_path, t.lineno)):
            body_lines = t.end_lineno - t.lineno + 1
            print(f"  {t.file_path}:{t.lineno:<4d}  {t.qualified_name:<40s}  ({body_lines} lines)")
        print(f"\nTotal: {len(targets)} targets")
        return

    # ── Load existing tasks (resume support) ─────────────────────────────
    existing_ids = load_existing_task_ids(args.tasks_dir)
    if existing_ids:
        log.info("Found %d existing tasks (will skip)", len(existing_ids))

    # ── Shuffle for variety ──────────────────────────────────────────────
    random.shuffle(targets)

    # ── Step 3–6: Generate → Validate → Store loop ──────────────────────
    generated = 0
    attempted = 0
    rejected_static = 0
    rejected_original = 0
    rejected_broken = 0
    llm_failures = 0

    log.info("")
    log.info("Starting task generation (max %d tasks, seed %d)...", args.max_tasks, args.seed)
    log.info("")

    for target in targets:
        if generated >= args.max_tasks:
            break

        task_id = make_task_id(target, base_commit)

        # Skip already generated
        if task_id in existing_ids:
            log.debug("Skipping %s (already exists)", task_id)
            continue

        attempted += 1
        log.info(
            "━━━ [%d] %s  (%s:%d)  ━━━",
            attempted,
            target.qualified_name,
            target.file_path,
            target.lineno,
        )

        # ── Generate test via LLM ───────────────────────────────────────
        test_code = generate_test(target)
        if not test_code:
            log.info("  ✗ LLM returned no valid test code")
            llm_failures += 1
            continue

        # ── Static validation ────────────────────────────────────────────
        ok, reason = static_validate_test(test_code)
        if not ok:
            log.info("  ✗ Static validation failed: %s", reason)
            rejected_static += 1
            continue

        # ── Bidirectional validation ─────────────────────────────────────
        valid, patch = validate_bidirectional(repo_dir, venv_dir, target, test_code)

        if not valid:
            # Track which side failed (info was already logged)
            continue

        # ── Store validated task ─────────────────────────────────────────
        task_path = store_task(
            task_id=task_id,
            target=target,
            base_commit=base_commit,
            patch=patch,
            test_code=test_code,
            tasks_dir=args.tasks_dir,
        )
        generated += 1
        log.info("  ✓ STORED: %s", task_path.name)
        log.info("")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SWE-Synth — Generation Summary")
    print("=" * 60)
    print(f"  Repo:              {REPO_NAME} @ {BASE_TAG}")
    print(f"  Base commit:       {base_commit}")
    print(f"  Total targets:     {len(targets)}")
    print(f"  Attempted:         {attempted}")
    print(f"  LLM failures:      {llm_failures}")
    print(f"  Rejected (static): {rejected_static}")
    print(f"  Rejected (valid):  {attempted - generated - llm_failures - rejected_static}")
    print(f"  ✓ Generated:       {generated}")
    print(f"  Tasks directory:   {args.tasks_dir.resolve()}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
