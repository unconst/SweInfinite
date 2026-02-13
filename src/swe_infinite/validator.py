"""
Execution validator for the SWE-rebench pipeline.

Validates task candidates by actually running their tests in an isolated
environment, populating the critical FAIL_TO_PASS and PASS_TO_PASS fields.

Flow:
  1. Setup environment (Docker or local venv)
  2. Install dependencies using install_config
  3. Apply ONLY test_patch to base_commit -> run pytest -> collect failures
  4. Apply solution_patch -> run pytest -> collect passes
  5. Compute FAIL_TO_PASS (tests that flipped from fail to pass)
  6. Compute PASS_TO_PASS (tests that passed in both runs)
  7. Reject tasks where FAIL_TO_PASS is empty
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("swe-infinite.validator")

from .paths import REPOS_DIR, VALIDATION_DIR


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a task candidate."""

    status: str  # validated | install_failed | tests_pass_without_fix |
    #               no_tests_flipped | patch_apply_failed | test_error
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    requirements: str = ""
    error: str = ""
    base_test_output: str = ""
    fixed_test_output: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status == "validated" and len(self.fail_to_pass) > 0


# ---------------------------------------------------------------------------
# Test result parsing
# ---------------------------------------------------------------------------


@dataclass
class TestOutcome:
    """Result of a single test."""

    nodeid: str
    status: str  # PASSED | FAILED | ERROR | SKIPPED


def _parse_pytest_json_report(report_path: Path) -> list[TestOutcome]:
    """Parse pytest-json-report output into TestOutcome list."""
    if not report_path.exists():
        return []

    try:
        data = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    outcomes = []
    for test in data.get("tests", []):
        nodeid = test.get("nodeid", "")
        outcome = test.get("outcome", "").upper()
        if nodeid and outcome:
            outcomes.append(TestOutcome(nodeid=nodeid, status=outcome))

    return outcomes


def _parse_pytest_verbose_output(output: str) -> list[TestOutcome]:
    """Fallback: parse pytest -v output to extract test outcomes.

    Looks for lines like:
        tests/test_foo.py::test_bar PASSED
        tests/test_foo.py::TestClass::test_method FAILED
    """
    outcomes = []
    # Match pytest verbose output format
    pattern = re.compile(
        r"^([\w/\.\-]+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)",
        re.MULTILINE,
    )
    for m in pattern.finditer(output):
        outcomes.append(TestOutcome(nodeid=m.group(1), status=m.group(2)))

    return outcomes


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def _run(
    cmd: list[str], cwd: Path, timeout: int = 300, env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    log.debug("$ %s  (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env,
    )


def _get_python(repo_dir: Path) -> str:
    """Get the python executable for the validation venv."""
    venv_dir = repo_dir / ".val_venv"
    if sys.platform == "win32":
        p = venv_dir / "Scripts" / "python"
    else:
        p = venv_dir / "bin" / "python"
    if p.exists():
        return str(p)
    return "python"


# ---------------------------------------------------------------------------
# Extract test file paths from a diff
# ---------------------------------------------------------------------------


def _extract_test_files(test_patch: str) -> list[str]:
    """Parse diff headers to find test file paths."""
    files = []
    for m in re.finditer(r"^diff --git a/.+? b/(.+?)$", test_patch, re.MULTILINE):
        path = m.group(1)
        if path not in files:
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _setup_validation_repo(task: dict) -> Path | None:
    """Copy and checkout the repo for validation.

    Returns the repo directory, or None on failure.
    """
    instance_id = task["instance_id"]
    repo_name = task["repo"]
    base_commit = task["base_commit"]

    val_repo = VALIDATION_DIR / instance_id
    if val_repo.exists():
        shutil.rmtree(val_repo)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Check for cached clone
    safe_name = repo_name.replace("/", "__")
    cached = REPOS_DIR / safe_name

    if cached.exists():
        log.info("  [validator] Copying cached repo %s...", safe_name)
        shutil.copytree(cached, val_repo, symlinks=True)
    else:
        log.info("  [validator] Cloning %s...", repo_name)
        url = f"https://github.com/{repo_name}.git"
        try:
            result = _run(
                ["git", "clone", "--filter=blob:none", url, str(val_repo)],
                cwd=VALIDATION_DIR, timeout=180,
            )
            if result.returncode != 0:
                log.error("  [validator] Clone failed: %s", result.stderr[:300])
                return None
        except subprocess.TimeoutExpired:
            log.error("  [validator] Clone timed out for %s", repo_name)
            if val_repo.exists():
                shutil.rmtree(val_repo, ignore_errors=True)
            return None

    # Checkout base_commit
    result = _run(["git", "checkout", base_commit, "--force"], cwd=val_repo, timeout=120)
    if result.returncode != 0:
        log.error("  [validator] Checkout failed: %s", result.stderr[:300])
        return None

    # Clean untracked files
    _run(["git", "clean", "-fdx"], cwd=val_repo, timeout=30)

    return val_repo


def _install_dependencies(
    repo_dir: Path, install_config: dict | None = None,
) -> bool:
    """Install project dependencies into a venv. Returns True on success."""
    venv_dir = repo_dir / ".val_venv"

    # Determine Python version from install_config
    python_ver = "3.10"
    if install_config and install_config.get("python"):
        python_ver = install_config["python"]

    # Create venv
    result = _run(
        ["uv", "venv", str(venv_dir), "--python", python_ver],
        cwd=repo_dir, timeout=60,
    )
    if result.returncode != 0:
        # Fallback: try without specific python version
        result = _run(["uv", "venv", str(venv_dir)], cwd=repo_dir, timeout=30)
        if result.returncode != 0:
            log.warning("  [validator] uv venv failed: %s", result.stderr[:200])
            return False

    python = _get_python(repo_dir)

    # Always install pytest and pytest-json-report
    _run(
        ["uv", "pip", "install", "--python", python,
         "pytest", "pytest-json-report"],
        cwd=repo_dir, timeout=120,
    )

    if install_config:
        # Pre-install commands (system deps)
        for cmd_str in install_config.get("pre_install", []):
            log.debug("  [validator] pre_install: %s", cmd_str)
            try:
                subprocess.run(
                    cmd_str, shell=True, cwd=repo_dir,
                    capture_output=True, text=True, timeout=120,
                )
            except (subprocess.TimeoutExpired, OSError):
                pass

        # Install from requirements files
        for req_path in install_config.get("reqs_path", []):
            if (repo_dir / req_path).exists():
                _run(
                    ["uv", "pip", "install", "--python", python, "-r", req_path],
                    cwd=repo_dir, timeout=300,
                )

        # Install extra pip packages
        pip_pkgs = install_config.get("pip_packages", [])
        if pip_pkgs:
            _run(
                ["uv", "pip", "install", "--python", python] + pip_pkgs,
                cwd=repo_dir, timeout=300,
            )

        # Main install command
        install_cmd = install_config.get("install", "")
        if install_cmd:
            # Replace pip/python references with venv paths
            install_cmd = install_cmd.replace("pip install", f"uv pip install --python {python}")
            log.debug("  [validator] install: %s", install_cmd)
            try:
                subprocess.run(
                    install_cmd, shell=True, cwd=repo_dir,
                    capture_output=True, text=True, timeout=300,
                )
            except (subprocess.TimeoutExpired, OSError):
                pass
    else:
        # Best-effort: try installing the project
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


def _pip_freeze(repo_dir: Path) -> str:
    """Get pip freeze output from the validation venv."""
    python = _get_python(repo_dir)
    result = _run(
        ["uv", "pip", "freeze", "--python", python],
        cwd=repo_dir, timeout=30,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return ""


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


def _apply_patch(repo_dir: Path, patch: str) -> bool:
    """Apply a unified diff patch to the repo. Returns True on success."""
    if not patch.strip():
        return True

    patch_file = repo_dir / "_temp_patch.diff"
    patch_file.write_text(patch)

    result = _run(
        ["git", "apply", "--allow-empty", str(patch_file)],
        cwd=repo_dir, timeout=30,
    )
    patch_file.unlink(missing_ok=True)

    if result.returncode != 0:
        # Try with --3way as fallback
        patch_file.write_text(patch)
        result = _run(
            ["git", "apply", "--3way", str(patch_file)],
            cwd=repo_dir, timeout=30,
        )
        patch_file.unlink(missing_ok=True)
        if result.returncode != 0:
            log.warning("  [validator] git apply failed: %s", result.stderr[:300])
            return False

    return True


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def _run_tests(
    repo_dir: Path, test_files: list[str], timeout: int = 120,
) -> tuple[list[TestOutcome], str]:
    """Run pytest on test files, collecting per-test results.

    Returns (outcomes, raw_output).
    """
    python = _get_python(repo_dir)
    report_path = repo_dir / ".pytest_report.json"
    report_path.unlink(missing_ok=True)

    # Filter to files that exist
    existing = [f for f in test_files if (repo_dir / f).exists()]
    if not existing:
        return [], "No test files found on disk"

    cmd = [
        python, "-m", "pytest",
        "-x", "--tb=short", "-v",
        "--json-report", f"--json-report-file={report_path}",
        "-W", "ignore::DeprecationWarning",
        "--no-header", "-rA",
    ] + existing

    try:
        result = _run(cmd, cwd=repo_dir, timeout=timeout)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return [], "Test execution timed out"

    # Try JSON report first, fall back to parsing verbose output
    outcomes = _parse_pytest_json_report(report_path)
    if not outcomes:
        outcomes = _parse_pytest_verbose_output(output)

    report_path.unlink(missing_ok=True)
    return outcomes, output


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_task(
    task: dict,
    use_docker: bool = False,
    test_timeout: int = 120,
) -> ValidationResult:
    """Full execution validation of a task candidate.

    Validates that:
      1. Dependencies can be installed
      2. Tests FAIL on base code + test_patch (without solution)
      3. Tests PASS on base code + test_patch + solution_patch
      4. FAIL_TO_PASS is non-empty (tests actually flip)

    When use_docker=True, runs validation inside a Docker container
    (requires docker_env module).
    """
    instance_id = task["instance_id"]
    test_patch = task.get("test_patch", "")
    solution_patch = task.get("patch", "")
    install_config = task.get("install_config")

    log.info("  [validator] Validating %s...", instance_id)

    if use_docker:
        try:
            from .docker_env import DockerEnvironment
            return _validate_in_docker(task, test_timeout)
        except ImportError:
            log.warning("  [validator] docker_env not available, falling back to local")

    # --- Setup repo ---
    repo_dir = _setup_validation_repo(task)
    if repo_dir is None:
        return ValidationResult(status="setup_failed", error="Could not setup repo")

    try:
        return _validate_local(
            repo_dir, task, test_patch, solution_patch,
            install_config, test_timeout,
        )
    finally:
        # Cleanup validation directory
        try:
            shutil.rmtree(repo_dir, ignore_errors=True)
        except OSError:
            pass


def _validate_local(
    repo_dir: Path,
    task: dict,
    test_patch: str,
    solution_patch: str,
    install_config: dict | None,
    test_timeout: int,
) -> ValidationResult:
    """Run validation in a local venv."""
    test_files = _extract_test_files(test_patch)
    if not test_files:
        return ValidationResult(
            status="test_error", error="No test files found in test_patch",
        )

    # --- Install dependencies ---
    log.info("  [validator] Installing dependencies...")
    install_ok = _install_dependencies(repo_dir, install_config)
    if not install_ok:
        return ValidationResult(status="install_failed", error="Dependency install failed")

    # --- Phase 1: Apply ONLY test_patch, run tests (should FAIL) ---
    log.info("  [validator] Phase 1: Running tests WITHOUT solution (expect failures)...")
    if not _apply_patch(repo_dir, test_patch):
        return ValidationResult(
            status="patch_apply_failed", error="test_patch failed to apply",
        )

    base_outcomes, base_output = _run_tests(repo_dir, test_files, timeout=test_timeout)

    if not base_outcomes:
        return ValidationResult(
            status="test_error",
            error="No test results collected (pytest may have crashed)",
            base_test_output=base_output,
        )

    # Collect failures
    base_failures = {t.nodeid for t in base_outcomes if t.status == "FAILED"}
    base_errors = {t.nodeid for t in base_outcomes if t.status == "ERROR"}
    base_passes = {t.nodeid for t in base_outcomes if t.status == "PASSED"}

    # All relevant failing tests (FAILED + ERROR)
    base_failing = base_failures | base_errors

    if not base_failing:
        return ValidationResult(
            status="tests_pass_without_fix",
            error=f"All {len(base_passes)} tests pass without the fix (expected failures)",
            base_test_output=base_output,
        )

    log.info(
        "  [validator] Phase 1 results: %d failing, %d passing, %d error",
        len(base_failures), len(base_passes), len(base_errors),
    )

    # --- Phase 2: Apply solution_patch, run tests again (should PASS) ---
    log.info("  [validator] Phase 2: Applying solution patch and re-running tests...")
    if not _apply_patch(repo_dir, solution_patch):
        return ValidationResult(
            status="patch_apply_failed",
            error="solution_patch failed to apply",
            base_test_output=base_output,
        )

    fixed_outcomes, fixed_output = _run_tests(repo_dir, test_files, timeout=test_timeout)

    if not fixed_outcomes:
        return ValidationResult(
            status="test_error",
            error="No test results after applying solution (pytest may have crashed)",
            base_test_output=base_output,
            fixed_test_output=fixed_output,
        )

    fixed_passes = {t.nodeid for t in fixed_outcomes if t.status == "PASSED"}

    # --- Compute FAIL_TO_PASS and PASS_TO_PASS ---
    fail_to_pass = sorted(base_failing & fixed_passes)
    pass_to_pass = sorted(base_passes & fixed_passes)

    log.info(
        "  [validator] Phase 2 results: FAIL_TO_PASS=%d, PASS_TO_PASS=%d",
        len(fail_to_pass), len(pass_to_pass),
    )

    if not fail_to_pass:
        return ValidationResult(
            status="no_tests_flipped",
            error="No tests flipped from fail to pass after applying solution",
            base_test_output=base_output,
            fixed_test_output=fixed_output,
        )

    # --- Collect pip freeze for reproducibility ---
    requirements = _pip_freeze(repo_dir)

    return ValidationResult(
        status="validated",
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        requirements=requirements,
        base_test_output=base_output,
        fixed_test_output=fixed_output,
    )


def _validate_in_docker(task: dict, test_timeout: int) -> ValidationResult:
    """Run validation inside a Docker container for isolation."""
    from .docker_env import DockerEnvironment

    instance_id = task["instance_id"]
    test_patch = task.get("test_patch", "")
    solution_patch = task.get("patch", "")
    install_config = task.get("install_config") or {}

    test_files = _extract_test_files(test_patch)
    if not test_files:
        return ValidationResult(
            status="test_error", error="No test files found in test_patch",
        )

    docker_env = DockerEnvironment(task)

    try:
        # Build and start container
        if not docker_env.build():
            return ValidationResult(status="install_failed", error="Docker build failed")

        # Apply test_patch only
        if not docker_env.apply_patch(test_patch):
            return ValidationResult(
                status="patch_apply_failed", error="test_patch failed to apply in Docker",
            )

        # Run tests (expect failures)
        base_outcomes, base_output = docker_env.run_tests(test_files, timeout=test_timeout)

        if not base_outcomes:
            return ValidationResult(
                status="test_error",
                error="No test results collected in Docker (pytest may have crashed)",
                base_test_output=base_output,
            )

        base_failing = {t.nodeid for t in base_outcomes if t.status in ("FAILED", "ERROR")}
        base_passes = {t.nodeid for t in base_outcomes if t.status == "PASSED"}

        if not base_failing:
            return ValidationResult(
                status="tests_pass_without_fix",
                error=f"All {len(base_passes)} tests pass without the fix",
                base_test_output=base_output,
            )

        # Apply solution patch
        if not docker_env.apply_patch(solution_patch):
            return ValidationResult(
                status="patch_apply_failed", error="solution_patch failed in Docker",
                base_test_output=base_output,
            )

        # Run tests again (expect passes)
        fixed_outcomes, fixed_output = docker_env.run_tests(test_files, timeout=test_timeout)

        fixed_passes = {t.nodeid for t in fixed_outcomes if t.status == "PASSED"}

        fail_to_pass = sorted(base_failing & fixed_passes)
        pass_to_pass = sorted(base_passes & fixed_passes)

        if not fail_to_pass:
            return ValidationResult(
                status="no_tests_flipped",
                error="No tests flipped from fail to pass",
                base_test_output=base_output,
                fixed_test_output=fixed_output,
            )

        requirements = docker_env.pip_freeze()

        return ValidationResult(
            status="validated",
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            requirements=requirements,
            base_test_output=base_output,
            fixed_test_output=fixed_output,
        )

    finally:
        docker_env.cleanup()


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


def validate_batch(
    tasks: list[dict],
    use_docker: bool = False,
    test_timeout: int = 120,
) -> list[tuple[dict, ValidationResult]]:
    """Validate a batch of tasks. Returns list of (task, result) tuples."""
    results = []
    for i, task in enumerate(tasks, 1):
        log.info(
            "[%d/%d] Validating %s...", i, len(tasks), task["instance_id"],
        )
        try:
            result = validate_task(task, use_docker=use_docker, test_timeout=test_timeout)
        except Exception:
            log.exception("Unexpected error validating %s", task["instance_id"])
            result = ValidationResult(status="test_error", error="Unexpected exception")

        results.append((task, result))
        log.info(
            "  -> %s (FAIL_TO_PASS=%d, PASS_TO_PASS=%d)",
            result.status, len(result.fail_to_pass), len(result.pass_to_pass),
        )

    # Summary
    validated = sum(1 for _, r in results if r.is_valid)
    log.info(
        "Validation complete: %d/%d tasks validated successfully",
        validated, len(results),
    )
    return results
