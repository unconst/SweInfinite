"""
Multi-language support for the SWE-rebench pipeline.

Provides language-specific handlers for:
  - Test file detection (patterns for each language's test conventions)
  - Default install recipes (package manager, test runner)
  - Test execution (framework-specific commands)
  - Validation environment setup

Currently supported:
  - Python (pytest, unittest)
  - TypeScript/JavaScript (jest, vitest, mocha)
  - Java (junit, maven/gradle)
  - Go (go test)

To add a new language:
  1. Create a LanguageHandler subclass
  2. Register it in LANGUAGE_HANDLERS
  3. Add test path patterns and install recipe defaults
"""

from __future__ import annotations

import json
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Base handler
# ---------------------------------------------------------------------------


@dataclass
class TestRunResult:
    """Result from running tests in a specific language."""
    passed: bool
    output: str
    test_outcomes: list[dict]  # [{nodeid, status}]


class LanguageHandler(ABC):
    """Abstract base for language-specific pipeline operations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Language name (lowercase)."""

    @property
    @abstractmethod
    def test_path_patterns(self) -> list[re.Pattern]:
        """Regex patterns that match test file paths."""

    def is_test_file(self, path: str) -> bool:
        """Check if a file path is a test file for this language."""
        return any(pat.search(path) for pat in self.test_path_patterns)

    @abstractmethod
    def default_install_recipe(self, repo_dir: Path) -> dict:
        """Generate a default install recipe for this language."""

    @abstractmethod
    def run_tests(
        self, repo_dir: Path, test_files: list[str], timeout: int = 120,
    ) -> TestRunResult:
        """Run tests and return structured results."""


# ---------------------------------------------------------------------------
# Python handler
# ---------------------------------------------------------------------------


class PythonHandler(LanguageHandler):
    """Handler for Python projects (pytest/unittest)."""

    @property
    def name(self) -> str:
        return "python"

    @property
    def test_path_patterns(self) -> list[re.Pattern]:
        return [
            re.compile(r"(^|/)tests?/"),
            re.compile(r"(^|/)test_[^/]+\.py$"),
            re.compile(r"(^|/)[^/]+_test\.py$"),
            re.compile(r"(^|/)conftest\.py$"),
            re.compile(r"(^|/)testing/"),
        ]

    def default_install_recipe(self, repo_dir: Path) -> dict:
        recipe = {
            "python": "3.10",
            "install": "pip install -e .",
            "test_cmd": "pytest --no-header -rA --tb=line -q",
            "pre_install": [],
            "pip_packages": ["pytest"],
            "reqs_path": [],
        }

        for req_file in sorted(repo_dir.glob("requirements*.txt")):
            recipe["reqs_path"].append(str(req_file.relative_to(repo_dir)))

        pyproject = repo_dir / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(errors="replace")
                m = re.search(r'requires-python\s*=\s*">=(\d+\.\d+)"', content)
                if m:
                    recipe["python"] = m.group(1)
            except OSError:
                pass

        return recipe

    def run_tests(
        self, repo_dir: Path, test_files: list[str], timeout: int = 120,
    ) -> TestRunResult:
        existing = [f for f in test_files if (repo_dir / f).exists()]
        if not existing:
            return TestRunResult(passed=False, output="No test files found", test_outcomes=[])

        cmd = [
            "python", "-m", "pytest", "-x", "--tb=short", "-v",
            "--no-header", "-rA",
        ] + existing

        try:
            result = subprocess.run(
                cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Parse outcomes from verbose output
            outcomes = []
            pattern = re.compile(
                r"^([\w/\.\-]+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)",
                re.MULTILINE,
            )
            for m in pattern.finditer(output):
                outcomes.append({"nodeid": m.group(1), "status": m.group(2)})

            return TestRunResult(passed=passed, output=output, test_outcomes=outcomes)
        except subprocess.TimeoutExpired:
            return TestRunResult(passed=False, output="Timeout", test_outcomes=[])


# ---------------------------------------------------------------------------
# TypeScript/JavaScript handler
# ---------------------------------------------------------------------------


class TypeScriptHandler(LanguageHandler):
    """Handler for TypeScript/JavaScript projects (jest/vitest/mocha)."""

    @property
    def name(self) -> str:
        return "typescript"

    @property
    def test_path_patterns(self) -> list[re.Pattern]:
        return [
            re.compile(r"(^|/)__tests__/"),
            re.compile(r"(^|/)tests?/"),
            re.compile(r"\.(test|spec)\.(ts|tsx|js|jsx|mjs)$"),
            re.compile(r"(^|/)test\.(ts|tsx|js|jsx|mjs)$"),
        ]

    def default_install_recipe(self, repo_dir: Path) -> dict:
        recipe = {
            "install": "npm install",
            "test_cmd": "npm test",
            "pre_install": [],
            "pip_packages": [],
            "reqs_path": [],
        }

        # Detect package manager
        if (repo_dir / "pnpm-lock.yaml").exists():
            recipe["install"] = "pnpm install"
            recipe["test_cmd"] = "pnpm test"
        elif (repo_dir / "yarn.lock").exists():
            recipe["install"] = "yarn install"
            recipe["test_cmd"] = "yarn test"
        elif (repo_dir / "bun.lockb").exists():
            recipe["install"] = "bun install"
            recipe["test_cmd"] = "bun test"

        # Check package.json for test framework
        pkg_json = repo_dir / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                scripts = data.get("scripts", {})
                if "test" in scripts:
                    test_script = scripts["test"]
                    install_parts = (recipe.get("install") or "npm install").split()
                    pm_name = install_parts[0] if install_parts else "npm"
                    if "vitest" in test_script:
                        recipe["test_cmd"] = f"{pm_name} run test"
                    elif "jest" in test_script:
                        recipe["test_cmd"] = f"{pm_name} run test"
            except (json.JSONDecodeError, OSError):
                pass

        return recipe

    def run_tests(
        self, repo_dir: Path, test_files: list[str], timeout: int = 120,
    ) -> TestRunResult:
        # Determine package manager
        if (repo_dir / "pnpm-lock.yaml").exists():
            pm = "pnpm"
        elif (repo_dir / "yarn.lock").exists():
            pm = "yarn"
        elif (repo_dir / "bun.lockb").exists():
            pm = "bun"
        else:
            pm = "npx"

        existing = [f for f in test_files if (repo_dir / f).exists()]
        if not existing:
            return TestRunResult(passed=False, output="No test files found", test_outcomes=[])

        # Try jest/vitest with JSON reporter
        if pm == "npx":
            cmd = ["npx", "jest", "--json", "--forceExit"] + existing
        else:
            # pnpm/yarn/bun: use `<pm> exec jest` to run jest directly
            cmd = [pm, "exec", "jest", "--json", "--forceExit"] + existing

        try:
            result = subprocess.run(
                cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Parse jest JSON output
            outcomes = []
            try:
                data = json.loads(result.stdout)
                for suite in data.get("testResults", []):
                    for test in suite.get("testResults", []):
                        outcomes.append({
                            "nodeid": f"{suite.get('testFilePath', '')}::{test.get('fullName', '')}",
                            "status": "PASSED" if test.get("status") == "passed" else "FAILED",
                        })
            except (json.JSONDecodeError, KeyError):
                pass

            return TestRunResult(passed=passed, output=output, test_outcomes=outcomes)
        except subprocess.TimeoutExpired:
            return TestRunResult(passed=False, output="Timeout", test_outcomes=[])
        except FileNotFoundError:
            return TestRunResult(passed=False, output="jest not found", test_outcomes=[])


# ---------------------------------------------------------------------------
# Java handler
# ---------------------------------------------------------------------------


class JavaHandler(LanguageHandler):
    """Handler for Java projects (junit with maven/gradle)."""

    @property
    def name(self) -> str:
        return "java"

    @property
    def test_path_patterns(self) -> list[re.Pattern]:
        return [
            re.compile(r"(^|/)src/test/"),
            re.compile(r"(^|/)test/.*\.java$"),
            re.compile(r"Test\.java$"),
            re.compile(r"Tests\.java$"),
            re.compile(r"IT\.java$"),  # Integration tests
        ]

    def default_install_recipe(self, repo_dir: Path) -> dict:
        recipe = {
            "install": "",
            "test_cmd": "",
            "pre_install": [],
            "pip_packages": [],
            "reqs_path": [],
        }

        if (repo_dir / "pom.xml").exists():
            recipe["install"] = "mvn install -DskipTests"
            recipe["test_cmd"] = "mvn test"
        elif (repo_dir / "build.gradle").exists() or (repo_dir / "build.gradle.kts").exists():
            recipe["install"] = "./gradlew build -x test"
            recipe["test_cmd"] = "./gradlew test"

        return recipe

    def run_tests(
        self, repo_dir: Path, test_files: list[str], timeout: int = 300,
    ) -> TestRunResult:
        # Determine build tool
        if (repo_dir / "pom.xml").exists():
            cmd = ["mvn", "test", "-pl", ".", "-am"]
        elif (repo_dir / "build.gradle").exists() or (repo_dir / "build.gradle.kts").exists():
            cmd = ["./gradlew", "test"]
        else:
            return TestRunResult(passed=False, output="No build tool found", test_outcomes=[])

        try:
            result = subprocess.run(
                cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Parse surefire/junit output (basic)
            outcomes = []
            pattern = re.compile(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)")
            for m in pattern.finditer(output):
                total = int(m.group(1))
                failures = int(m.group(2))
                errors_count = int(m.group(3))
                passes = total - failures - errors_count
                # Approximate outcomes
                for i in range(passes):
                    outcomes.append({"nodeid": f"test_{i}", "status": "PASSED"})
                for i in range(failures + errors_count):
                    outcomes.append({"nodeid": f"test_fail_{i}", "status": "FAILED"})

            return TestRunResult(passed=passed, output=output, test_outcomes=outcomes)
        except subprocess.TimeoutExpired:
            return TestRunResult(passed=False, output="Timeout", test_outcomes=[])
        except FileNotFoundError:
            return TestRunResult(passed=False, output="Build tool not found", test_outcomes=[])


# ---------------------------------------------------------------------------
# Go handler
# ---------------------------------------------------------------------------


class GoHandler(LanguageHandler):
    """Handler for Go projects (go test)."""

    @property
    def name(self) -> str:
        return "go"

    @property
    def test_path_patterns(self) -> list[re.Pattern]:
        return [
            re.compile(r"_test\.go$"),
        ]

    def default_install_recipe(self, repo_dir: Path) -> dict:
        return {
            "install": "go mod download",
            "test_cmd": "go test ./...",
            "pre_install": [],
            "pip_packages": [],
            "reqs_path": [],
        }

    def run_tests(
        self, repo_dir: Path, test_files: list[str], timeout: int = 120,
    ) -> TestRunResult:
        # Go test runs all tests in packages, not individual files
        # Find unique packages from test files
        packages = set()
        for f in test_files:
            pkg = str(Path(f).parent)
            if pkg == ".":
                packages.add("./...")
            else:
                packages.add(f"./{pkg}/...")

        cmd = ["go", "test", "-v", "-json"] + sorted(packages)

        try:
            result = subprocess.run(
                cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Parse go test JSON output
            outcomes = []
            for line in result.stdout.split("\n"):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("Action") == "pass" and event.get("Test"):
                        outcomes.append({
                            "nodeid": f"{event.get('Package', '')}::{event['Test']}",
                            "status": "PASSED",
                        })
                    elif event.get("Action") == "fail" and event.get("Test"):
                        outcomes.append({
                            "nodeid": f"{event.get('Package', '')}::{event['Test']}",
                            "status": "FAILED",
                        })
                except json.JSONDecodeError:
                    pass

            return TestRunResult(passed=passed, output=output, test_outcomes=outcomes)
        except subprocess.TimeoutExpired:
            return TestRunResult(passed=False, output="Timeout", test_outcomes=[])
        except FileNotFoundError:
            return TestRunResult(passed=False, output="go not found", test_outcomes=[])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LANGUAGE_HANDLERS: dict[str, LanguageHandler] = {
    "python": PythonHandler(),
    "javascript": TypeScriptHandler(),
    "typescript": TypeScriptHandler(),
    "java": JavaHandler(),
    "go": GoHandler(),
}


def get_handler(language: str) -> LanguageHandler | None:
    """Get the language handler for a given language name."""
    return LANGUAGE_HANDLERS.get(language.lower())


def get_supported_languages() -> set[str]:
    """Get the set of supported language names."""
    return set(LANGUAGE_HANDLERS.keys())


def detect_language(repo_dir: Path) -> str | None:
    """Try to detect the primary language of a repository from its files."""
    # Check for language-specific marker files
    markers = {
        "python": ["setup.py", "pyproject.toml", "setup.cfg", "requirements.txt"],
        "typescript": ["tsconfig.json", "package.json"],
        "javascript": ["package.json"],
        "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
        "go": ["go.mod", "go.sum"],
    }

    for lang, files in markers.items():
        for f in files:
            if (repo_dir / f).exists():
                return lang

    return None
