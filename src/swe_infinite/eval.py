#!/usr/bin/env python3
"""
SWE Eval — Evaluate dataset tasks using the Cursor agent CLI.

For each task in dataset/:
  1. Clone/copy the repo, checkout base_commit
  2. Apply the test_patch (adds failing tests)
  3. Install dependencies (best-effort)
  4. Run tests — expect FAIL (sanity check)
  5. Invoke Cursor agent with the problem_statement
  6. Run tests again — expect PASS
  7. Report results

Usage:
    python swe_eval.py                           # eval all tasks in dataset/
    python swe_eval.py dataset/foo.json          # eval specific task(s)
    python swe_eval.py --agent-timeout 300       # custom agent timeout (seconds)
    python swe_eval.py --skip-install            # skip pip install step
    python swe_eval.py --model sonnet-4          # choose agent model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .paths import DATASET_DIR, EVAL_DIR, REPOS_DIR, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("swe-eval")


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(paths: list[str]) -> list[dict]:
    """Load task JSONs from explicit paths or all files in dataset/."""
    tasks = []
    if paths:
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for f in sorted(path.glob("*.json")):
                    tasks.append(json.loads(f.read_text()))
            elif path.is_file():
                tasks.append(json.loads(path.read_text()))
            else:
                log.warning("Path not found: %s", p)
    else:
        if not DATASET_DIR.exists():
            log.error("Dataset directory not found: %s", DATASET_DIR)
            return []
        for f in sorted(DATASET_DIR.glob("*.json")):
            tasks.append(json.loads(f.read_text()))
    return tasks


# ---------------------------------------------------------------------------
# Test file discovery from diff
# ---------------------------------------------------------------------------


def extract_test_files(test_patch: str) -> list[str]:
    """Parse diff headers to find test file paths.

    Looks for 'diff --git a/X b/X' headers and extracts the 'b/' path.
    """
    files = []
    for m in re.finditer(r"^diff --git a/.+? b/(.+?)$", test_patch, re.MULTILINE):
        path = m.group(1)
        if path not in files:
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path, timeout: int = 300, **kwargs) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    log.debug("$ %s  (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, **kwargs)


# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------


def setup_repo(task: dict) -> Path | None:
    """Set up a repo for evaluation.

    - Copies from workspace/repos/ cache if available, otherwise clones fresh.
    - Checks out base_commit.
    - Applies test_patch via git apply.

    Returns the repo directory, or None on failure.
    """
    instance_id = task["instance_id"]
    repo_name = task["repo"]
    base_commit = task["base_commit"]
    test_patch = task.get("test_patch", "")

    # Eval work directory
    eval_repo = EVAL_DIR / instance_id
    if eval_repo.exists():
        shutil.rmtree(eval_repo)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Check for cached clone
    safe_name = repo_name.replace("/", "__")
    cached = REPOS_DIR / safe_name

    if cached.exists():
        log.info("  Copying cached repo %s...", safe_name)
        shutil.copytree(cached, eval_repo, symlinks=True)
    else:
        log.info("  Cloning %s...", repo_name)
        url = f"https://github.com/{repo_name}.git"
        try:
            result = _run(
                ["git", "clone", "--filter=blob:none", url, str(eval_repo)],
                cwd=EVAL_DIR,
                timeout=180,
            )
            if result.returncode != 0:
                log.error("  Clone failed: %s", result.stderr[:300])
                return None
        except subprocess.TimeoutExpired:
            log.error("  Clone timed out for %s", repo_name)
            if eval_repo.exists():
                shutil.rmtree(eval_repo, ignore_errors=True)
            return None

    # Checkout base_commit
    result = _run(["git", "checkout", base_commit, "--force"], cwd=eval_repo, timeout=120)
    if result.returncode != 0:
        log.error("  Checkout failed: %s", result.stderr[:300])
        return None

    # Clean any untracked files
    _run(["git", "clean", "-fdx"], cwd=eval_repo, timeout=30)

    # Apply test_patch
    if test_patch.strip():
        patch_file = eval_repo / "_test_patch.diff"
        patch_file.write_text(test_patch)
        result = _run(
            ["git", "apply", "--allow-empty", str(patch_file)],
            cwd=eval_repo,
            timeout=30,
        )
        patch_file.unlink(missing_ok=True)
        if result.returncode != 0:
            log.warning("  git apply test_patch failed: %s", result.stderr[:300])
            # Try with --3way as fallback
            patch_file.write_text(test_patch)
            result = _run(
                ["git", "apply", "--3way", str(patch_file)],
                cwd=eval_repo,
                timeout=30,
            )
            patch_file.unlink(missing_ok=True)
            if result.returncode != 0:
                log.error("  git apply --3way also failed: %s", result.stderr[:300])
                return None

    return eval_repo


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _install_with_config(repo_dir: Path, python: str, install_config: dict) -> bool:
    """Install dependencies using a structured install_config recipe."""
    # Pre-install commands (system deps)
    pre_install = install_config.get("pre_install", [])
    if not isinstance(pre_install, list):
        pre_install = []
    for cmd_str in pre_install:
        if not isinstance(cmd_str, str):
            continue
        log.info("  pre_install: %s", cmd_str[:80])
        try:
            subprocess.run(
                cmd_str, shell=True, cwd=repo_dir,
                capture_output=True, text=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Install from requirements files
    reqs_path = install_config.get("reqs_path", [])
    if not isinstance(reqs_path, list):
        reqs_path = []
    for req_path in reqs_path:
        if not isinstance(req_path, str):
            continue
        if (repo_dir / req_path).exists():
            _run(
                ["uv", "pip", "install", "--python", python, "-r", req_path],
                cwd=repo_dir, timeout=300,
            )

    # Extra pip packages
    pip_pkgs = install_config.get("pip_packages", [])
    if not isinstance(pip_pkgs, list):
        pip_pkgs = []
    pip_pkgs = [p for p in pip_pkgs if isinstance(p, str)]
    if pip_pkgs:
        _run(
            ["uv", "pip", "install", "--python", python] + pip_pkgs,
            cwd=repo_dir, timeout=300,
        )

    # Main install command
    install_cmd = install_config.get("install", "")
    if not isinstance(install_cmd, str):
        install_cmd = ""
    if install_cmd:
        normalized = install_cmd.replace("pip install", f"uv pip install --python {python}")
        log.info("  install: %s", normalized[:80])
        try:
            subprocess.run(
                normalized, shell=True, cwd=repo_dir,
                capture_output=True, text=True, timeout=300,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    return True


def _install_best_effort(repo_dir: Path, python: str) -> bool:
    """Fallback install strategy when no install_config is available."""
    pyproject = repo_dir / "pyproject.toml"
    setup_py = repo_dir / "setup.py"
    setup_cfg = repo_dir / "setup.cfg"
    requirements = repo_dir / "requirements.txt"

    if pyproject.exists() or setup_py.exists() or setup_cfg.exists():
        result = _run(
            ["uv", "pip", "install", "--python", python, "-e", "."],
            cwd=repo_dir, timeout=300,
        )
        if result.returncode != 0:
            log.warning("  uv pip install -e . failed (non-fatal): %s", result.stderr[:200])
            _run(
                ["uv", "pip", "install", "--python", python, "."],
                cwd=repo_dir, timeout=300,
            )

    if requirements.exists():
        _run(
            ["uv", "pip", "install", "--python", python, "-r", "requirements.txt"],
            cwd=repo_dir, timeout=300,
        )

    return True


def install_deps(repo_dir: Path, install_config: dict | None = None) -> bool:
    """Install the project's dependencies.

    Creates a venv inside the repo and installs with uv pip.
    Delegates to _install_with_config() when a recipe is provided,
    or _install_best_effort() otherwise.

    Returns True if install succeeded, False otherwise.
    """
    venv_dir = repo_dir / ".eval_venv"

    # Determine Python version from install_config
    python_ver = None
    if install_config and install_config.get("python"):
        python_ver = install_config["python"]

    # Create venv with uv
    venv_cmd = ["uv", "venv", str(venv_dir)]
    if python_ver:
        venv_cmd.extend(["--python", python_ver])
    result = _run(venv_cmd, cwd=repo_dir, timeout=60)
    if result.returncode != 0:
        result = _run(["uv", "venv", str(venv_dir)], cwd=repo_dir, timeout=30)
        if result.returncode != 0:
            log.warning("  uv venv failed: %s", result.stderr[:200])
            return False

    python = _get_python(repo_dir)

    # Install pytest and common test dependencies
    common_test_deps = [
        "pytest", "pytest-cov", "pytest-asyncio", "pytest-mock",
        "pytest-xdist", "mock", "coverage",
    ]
    _run(
        ["uv", "pip", "install", "--python", python] + common_test_deps,
        cwd=repo_dir, timeout=120,
    )

    if install_config:
        return _install_with_config(repo_dir, python, install_config)

    return _install_best_effort(repo_dir, python)


def _get_python(repo_dir: Path) -> str:
    """Get the python executable path for the eval venv."""
    venv_dir = repo_dir / ".eval_venv"
    if sys.platform == "win32":
        p = venv_dir / "Scripts" / "python"
    else:
        p = venv_dir / "bin" / "python"
    if p.exists():
        return str(p)
    return "python"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def _detect_language_from_test_files(test_files: list[str]) -> str:
    """Detect the programming language from test file extensions."""
    ext_map = {
        ".py": "python",
        ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
        ".java": "java",
        ".go": "go",
    }
    lang_counts: dict[str, int] = {}
    for fpath in test_files:
        ext = Path(fpath).suffix.lower()
        lang = ext_map.get(ext)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    if not lang_counts:
        return "python"  # fallback
    return max(lang_counts, key=lang_counts.get)  # type: ignore[arg-type]


def run_tests(
    repo_dir: Path,
    test_files: list[str],
    timeout: int = 120,
    language: str | None = None,
) -> tuple[bool, str]:
    """Run tests on the specified test files using the appropriate test runner.

    Detects the language from test file extensions and delegates to the
    matching language handler (pytest for Python, jest for JS/TS, go test
    for Go, mvn/gradle for Java).  Falls back to pytest for Python.

    Returns (passed: bool, output: str).
    """
    if language is None:
        language = _detect_language_from_test_files(test_files)

    # For non-Python languages, use the language-specific handler
    if language != "python":
        try:
            from .language_support import get_handler
            handler = get_handler(language)
            if handler:
                result = handler.run_tests(repo_dir, test_files, timeout=timeout)
                return result.passed, result.output
        except Exception as exc:
            log.warning("Language handler for %s failed: %s. Falling back to pytest.", language, exc)

    # Python (or fallback): use pytest with the eval venv
    python = _get_python(repo_dir)

    # Filter to files that actually exist
    existing = [f for f in test_files if (repo_dir / f).exists()]
    if not existing:
        return False, "No test files found on disk"

    cmd = [
        python, "-m", "pytest", "-x", "--tb=short", "-q",
        "--no-header", "-o", "addopts=", "--override-ini=addopts=",
        "-p", "no:cacheprovider",
    ] + existing

    try:
        result = _run(cmd, cwd=repo_dir, timeout=timeout)
        output = result.stdout + result.stderr
        passed = result.returncode == 0
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out"


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

AGENT_PROMPT_TEMPLATE = """You are solving a software engineering task. The following issue has been reported:

{problem_statement}

Tests have been added to verify the fix. Run the tests to see what's failing, then fix the code to make them pass. Do NOT modify the test files themselves.

IMPORTANT environment setup:
- A Python virtual environment is at: .eval_venv/
- Activate it with: source .eval_venv/bin/activate
- All dependencies and pytest are pre-installed there.

The test files are: {test_files}

To run the tests correctly (avoiding pyproject.toml overrides):
  source .eval_venv/bin/activate && python -m pytest -x --tb=short --no-header -o "addopts=" --override-ini="addopts=" -p no:cacheprovider {test_files_str}

Steps:
1. First run the tests above to see the failures
2. Read the test code to understand what's expected
3. Read the relevant source code
4. Make the necessary code changes to fix the issue
5. Run the tests again to verify they pass
6. Do NOT modify test files"""


@dataclass
class AgentResult:
    """Structured result from a Cursor agent invocation."""

    success: bool
    elapsed_seconds: float
    output: str
    token_count: int | None = None
    tool_calls: int | None = None
    model: str | None = None


def _parse_agent_json_output(raw_output: str) -> tuple[str, int | None, int | None]:
    """Parse structured JSON output from agent --output-format json.

    The agent emits one JSON object per line (NDJSON).  Each object has a
    ``type`` field.  We look for ``result`` objects and ``usage`` data.

    Returns (text_output, token_count, tool_calls).
    """
    text_parts: list[str] = []
    total_tokens: int = 0
    total_tool_calls: int = 0
    found_usage = False

    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Not a JSON line — treat as plain text
            text_parts.append(line)
            continue

        if not isinstance(obj, dict):
            text_parts.append(line)
            continue

        msg_type = obj.get("type", "")

        # Collect text content from assistant messages
        if msg_type == "text" or "content" in obj:
            content = obj.get("content") or obj.get("text") or ""
            if content:
                text_parts.append(str(content))

        # Collect usage / token info
        if "usage" in obj:
            usage = obj["usage"]
            if isinstance(usage, dict):
                found_usage = True
                total_tokens += usage.get("total_tokens", 0)
                # Some formats use input_tokens + output_tokens
                if not total_tokens:
                    total_tokens += usage.get("input_tokens", 0)
                    total_tokens += usage.get("output_tokens", 0)

        # Count tool calls
        if msg_type == "tool_call" or "tool" in obj:
            total_tool_calls += 1
        if "tool_calls" in obj:
            calls = obj["tool_calls"]
            if isinstance(calls, list):
                total_tool_calls += len(calls)

    return (
        "\n".join(text_parts),
        total_tokens if found_usage else None,
        total_tool_calls if total_tool_calls > 0 else None,
    )


def invoke_agent(
    repo_dir: Path,
    problem_statement: str,
    test_files: list[str],
    timeout: int = 300,
    model: str | None = None,
) -> AgentResult:
    """Invoke the Cursor agent CLI to solve the task.

    Uses ``--output-format json`` to capture structured data including
    token counts and tool-call counts when available.

    Returns an AgentResult with success, timing, output, and metrics.
    """
    test_files_str = " ".join(test_files)
    prompt = AGENT_PROMPT_TEMPLATE.format(
        problem_statement=problem_statement,
        test_files=", ".join(test_files),
        test_files_str=test_files_str,
    )

    cmd = [
        "agent", "-p", prompt,
        "--output-format", "json",
        "--force",
        "--workspace", str(repo_dir),
    ]
    if model:
        cmd.extend(["--model", model])

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_dir,
        )
        elapsed = time.monotonic() - start
        raw_output = result.stdout + result.stderr

        # Try to parse structured JSON output for metrics
        text_output, token_count, tool_calls = _parse_agent_json_output(result.stdout)
        if not text_output:
            text_output = raw_output

        return AgentResult(
            success=result.returncode == 0,
            elapsed_seconds=elapsed,
            output=text_output,
            token_count=token_count,
            tool_calls=tool_calls,
            model=model,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return AgentResult(
            success=False,
            elapsed_seconds=elapsed,
            output=f"Agent timed out after {timeout}s",
            model=model,
        )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _make_result_dict(task: dict) -> dict:
    """Create an initial result dict for an evaluation run."""
    return {
        "instance_id": task["instance_id"],
        "repo": task.get("repo", ""),
        "status": "error",
        "agent_time_seconds": 0.0,
        "sanity_check": False,
        "test_files": [],
        "expected_fail_to_pass": task.get("FAIL_TO_PASS", []),
        "expected_pass_to_pass": task.get("PASS_TO_PASS", []),
        "error": None,
    }


def _setup_eval_environment(
    task: dict, skip_install: bool,
) -> tuple[Path, dict | None] | None:
    """Setup repo, install dependencies. Returns (repo_dir, install_config) or None."""
    repo_dir = setup_repo(task)
    if repo_dir is None:
        return None

    install_config = task.get("install_config")
    if isinstance(install_config, str):
        try:
            install_config = json.loads(install_config)
        except json.JSONDecodeError:
            install_config = None
    if install_config == {}:
        install_config = None

    if not skip_install:
        log.info("  Installing dependencies...")
        install_deps(repo_dir, install_config=install_config)
    else:
        venv_dir = repo_dir / ".eval_venv"
        _run(["uv", "venv", str(venv_dir)], cwd=repo_dir, timeout=30)
        python = _get_python(repo_dir)
        _run(["uv", "pip", "install", "--python", python, "pytest"], cwd=repo_dir, timeout=60)

    return repo_dir, install_config


def evaluate_task(
    task: dict,
    agent_timeout: int = 300,
    skip_install: bool = False,
    model: str | None = None,
) -> dict:
    """Evaluate a single task end-to-end.

    Returns a result dict with status, timing, etc.
    """
    test_patch = task.get("test_patch", "")
    problem_statement = task.get("problem_statement", "")
    result = _make_result_dict(task)

    # Discover test files from patch
    test_files = extract_test_files(test_patch)
    result["test_files"] = test_files

    if not test_files:
        result["error"] = "No test files found in test_patch"
        log.warning("  No test files in test_patch")
        return result

    if not test_patch.strip():
        result["error"] = "Empty test_patch"
        return result

    # Setup repo and install deps
    log.info("  Setting up repo...")
    env = _setup_eval_environment(task, skip_install)
    if env is None:
        result["error"] = "Repo setup failed"
        return result
    repo_dir, _ = env

    # Sanity check — tests should FAIL on base code
    log.info("  Sanity check: tests should fail on base code...")
    passed, _output = run_tests(repo_dir, test_files)
    if passed:
        result["status"] = "sanity_fail"
        result["sanity_check"] = False
        result["error"] = "Tests pass on base code (expected failure)"
        log.warning("  Sanity check FAILED — tests pass without fix")
        return result

    result["sanity_check"] = True
    log.info("  Sanity check passed — tests fail as expected")

    # Invoke agent
    log.info("  Invoking agent (timeout=%ds)...", agent_timeout)
    agent_result = invoke_agent(
        repo_dir, problem_statement, test_files, timeout=agent_timeout, model=model,
    )
    elapsed = agent_result.elapsed_seconds
    result["agent_time_seconds"] = round(elapsed, 1)
    result["token_count"] = agent_result.token_count
    result["tool_calls"] = agent_result.tool_calls
    result["model"] = agent_result.model

    if not agent_result.success and "timed out" in agent_result.output.lower():
        result["status"] = "timeout"
        result["error"] = f"Agent timed out after {agent_timeout}s"
        log.warning("  Agent timed out (%.1fs)", elapsed)
        return result

    # Run tests after agent
    log.info("  Running tests after agent intervention...")
    passed, _test_output = run_tests(repo_dir, test_files)

    if passed:
        result["status"] = "resolved"
        log.info("  RESOLVED — agent fixed the code (%.1fs)", elapsed)
    else:
        result["status"] = "unresolved"
        log.info("  UNRESOLVED — tests still fail (%.1fs)", elapsed)

    return result


# ---------------------------------------------------------------------------
# Single-task evaluation (for pipeline integration)
# ---------------------------------------------------------------------------


def evaluate_new_task(
    task: dict,
    model: str = "opus-4.6",
    agent_timeout: int = 300,
    skip_install: bool = False,
    cleanup: bool = True,
) -> dict:
    """Evaluate a single task and return an eval_result dict.

    This is a lighter-weight entry point designed for pipeline integration.
    It runs the full eval pipeline (setup, install, sanity, agent, retest)
    and returns a dict suitable for embedding in the task JSON as ``eval_result``.

    Returns:
        dict with keys: status, model, agent_time_seconds, token_count,
        tool_calls, sanity_check, evaluated_at, error.
    """
    eval_result: dict = {
        "status": "error",
        "model": model,
        "agent_time_seconds": 0.0,
        "token_count": None,
        "tool_calls": None,
        "sanity_check": False,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }

    instance_id = task.get("instance_id", "unknown")

    try:
        result = evaluate_task(
            task,
            agent_timeout=agent_timeout,
            skip_install=skip_install,
            model=model,
        )

        eval_result["status"] = result["status"]
        eval_result["agent_time_seconds"] = result.get("agent_time_seconds", 0.0)
        eval_result["token_count"] = result.get("token_count")
        eval_result["tool_calls"] = result.get("tool_calls")
        eval_result["sanity_check"] = result.get("sanity_check", False)
        eval_result["error"] = result.get("error")
        eval_result["evaluated_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as exc:
        log.exception("  Unhandled error evaluating %s", instance_id)
        eval_result["error"] = f"Unhandled exception: {exc!s:.200}"
        eval_result["evaluated_at"] = datetime.now(timezone.utc).isoformat()

    finally:
        if cleanup:
            cleanup_eval(instance_id)

    return eval_result


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_eval(instance_id: str) -> None:
    """Remove the eval working directory for a task."""
    eval_repo = EVAL_DIR / instance_id
    if eval_repo.exists():
        shutil.rmtree(eval_repo, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE Eval — Evaluate dataset tasks with Cursor agent",
    )
    parser.add_argument(
        "tasks", nargs="*", default=[],
        help="Task JSON files or directories (default: dataset/)",
    )
    parser.add_argument(
        "--agent-timeout", type=int, default=300,
        help="Agent timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip pip install step",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model for Cursor agent (e.g., sonnet-4, gpt-5)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Remove eval working directories after each task",
    )
    parser.add_argument(
        "--update-tasks", action="store_true",
        help="Write eval_result back into each task JSON file in dataset/",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SWE Eval — Agent Evaluation Pipeline")
    log.info("=" * 60)

    tasks = load_tasks(args.tasks)
    if not tasks:
        log.error("No tasks found.")
        return

    log.info("Found %d task(s) to evaluate.", len(tasks))

    results = []
    for i, task in enumerate(tasks, 1):
        instance_id = task["instance_id"]
        log.info("-" * 60)
        log.info("[%d/%d] Evaluating %s", i, len(tasks), instance_id)
        log.info("=" * 60)

        try:
            result = evaluate_task(
                task,
                agent_timeout=args.agent_timeout,
                skip_install=args.skip_install,
                model=args.model,
            )
        except Exception:
            log.exception("  Unhandled error evaluating %s", instance_id)
            result = {
                "instance_id": instance_id,
                "repo": task.get("repo", ""),
                "status": "error",
                "agent_time_seconds": 0.0,
                "sanity_check": False,
                "test_files": [],
                "error": "Unhandled exception",
            }

        results.append(result)

        # Write eval_result back into the task JSON if requested
        if args.update_tasks:
            from .task_store import update_task_eval_result

            eval_result_for_task = {
                "status": result["status"],
                "model": result.get("model") or args.model,
                "agent_time_seconds": result.get("agent_time_seconds", 0.0),
                "token_count": result.get("token_count"),
                "tool_calls": result.get("tool_calls"),
                "sanity_check": result.get("sanity_check", False),
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "error": result.get("error"),
            }
            update_task_eval_result(eval_result_for_task, instance_id, DATASET_DIR)

        if args.cleanup:
            cleanup_eval(instance_id)

    # Summary
    total = len(results)
    resolved = sum(1 for r in results if r["status"] == "resolved")
    unresolved = sum(1 for r in results if r["status"] == "unresolved")
    sanity_fail = sum(1 for r in results if r["status"] == "sanity_fail")
    errors = sum(1 for r in results if r["status"] == "error")
    timeouts = sum(1 for r in results if r["status"] == "timeout")
    agent_times = [r["agent_time_seconds"] for r in results if r["agent_time_seconds"] > 0]
    avg_time = sum(agent_times) / len(agent_times) if agent_times else 0.0

    # Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    eval_id = f"eval_{ts}"
    report = {
        "eval_id": eval_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "agent_timeout": args.agent_timeout,
            "skip_install": args.skip_install,
            "model": args.model,
        },
        "summary": {
            "total": total,
            "resolved": resolved,
            "unresolved": unresolved,
            "sanity_fail": sanity_fail,
            "errors": errors,
            "timeouts": timeouts,
            "avg_agent_time": round(avg_time, 1),
        },
        "tasks": results,
    }

    report_path = RESULTS_DIR / f"{eval_id}.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Results written to %s", report_path)

    # Print summary
    log.info("=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info("  Total tasks:      %d", total)
    log.info("  Resolved:         %d / %d  (%.0f%%)", resolved, total, 100 * resolved / total if total else 0)
    log.info("  Unresolved:       %d", unresolved)
    log.info("  Sanity failures:  %d", sanity_fail)
    log.info("  Errors:           %d", errors)
    log.info("  Agent timeouts:   %d", timeouts)
    log.info("  Avg agent time:   %.1fs", avg_time)
    log.info("=" * 60)
    log.info("")
    log.info("  %-16s %-12s %s", "TASK", "RESULT", "TIME")
    log.info("  " + "-" * 56)
    for r in results:
        short_id = r["instance_id"][:30]
        status = r["status"].upper()
        t = f"{r['agent_time_seconds']:.1f}s" if r["agent_time_seconds"] > 0 else "-"
        log.info("  %-16s %-12s %s", short_id, status, t)


if __name__ == "__main__":
    main()
