"""
Installation recipe generator for the SWE-rebench pipeline.

Uses the Cursor agent CLI (`agent -p`) to analyze repository files and
generate structured JSON installation recipes, following the paper's
agentless approach (Section 2.2, Appendix B).

The process has two phases:
  1. Identify relevant files (README, setup.py, pyproject.toml, etc.)
  2. Extract a structured JSON installation recipe

On failure, retries with error context (up to max_retries attempts).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import textwrap
from pathlib import Path

log = logging.getLogger("swe-infinite.recipe_generator")


# ---------------------------------------------------------------------------
# Prompts (adapted from paper Appendix B.1, B.2, B.4)
# ---------------------------------------------------------------------------

IDENTIFY_FILES_PROMPT = textwrap.dedent("""\
    You are tasked with identifying files that likely contain installation
    instructions for a GitHub repository.

    Repository: {repo_name}

    Below is a list of files in the repository:
    {file_list}

    Please analyze this list and identify the files that are most likely to
    contain information about:
    * Installation instructions
    * Dependencies or requirements
    * Setup procedures
    * Development environment configuration
    * Testing setup

    Think step by step:
    * Identify README files, as they often contain installation instructions.
    * Look for setup.py, pyproject.toml, requirements.txt, environment.yml.
    * Consider files in directories like docs that might contain installation guides.
    * Look for test configuration files that might help understand how to run tests.
    * Consider only files from the list provided above.
    * Prioritize files in the repository root (top-level directory).
    * Only include files from subdirectories if they are clearly relevant.

    Return ONLY a JSON array containing the paths to the most relevant files.
    Include only files that are directly relevant. Sort from most to least
    relevant and limit to no more than 10 files.

    For example:
    ["README.md", "setup.py", "requirements.txt"]
""")


EXTRACT_RECIPE_PROMPT = textwrap.dedent("""\
    You are tasked with extracting detailed installation instructions from
    the following repository files.

    Repository: {repo_name}

    Here are the contents of the relevant files:
    {file_contents}

    Analyze the content and extract comprehensive installation instructions.
    Return your findings as a JSON object with these fields:

    {{
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "pip install -e .[dev]",
        "test_cmd": "pytest --no-header -rA --tb=line --color=no -p no:cacheprovider -W ignore::DeprecationWarning",
        "pre_install": ["apt-get update", "apt-get install -y gcc"],
        "reqs_path": ["requirements/base.txt"],
        "env_yml_path": ["environment.yml"],
        "pip_packages": ["numpy>=1.16.0", "pandas>=1.0.0"]
    }}

    Here is how this JSON will be used in a bash script:
    ```bash
    git clone <repo_url> repo
    cd repo
    git checkout <base_sha>
    bash <pre_install>
    conda create -n <repo> python=<python> -y
    conda activate <repo>
    if <packages> == requirements.txt; then
        for path in <reqs_path>:
            pip install -r $path
    elif <packages> == environment.yml; then
        for path in <env_yml_path>:
            conda env update -f $path
    else:
        pip install <packages>
    pip install <pip_packages>
    bash <install>
    bash <test_cmd>
    ```

    IMPORTANT:
    * For "install", always use local install commands like pip install -e .[dev]
    * Do NOT include packages in pip_packages that will be installed by pip install -e .
    * Include only explicitly needed packages in pip_packages.
    * Add relevant test frameworks to pip_packages (e.g., pytest, nose).
    * Prefer direct pytest commands over general wrappers.
    * Avoid test commands with placeholders like {{test_name}}.
    * If a Makefile runs tests, extract the actual test command.

    Required fields: python, install, test_cmd
    Optional fields: packages, pre_install, reqs_path, env_yml_path, pip_packages

    First provide brief reasoning, then return the JSON in a markdown code block.
""")


CORRECTION_PROMPT = textwrap.dedent("""\
    You are an expert in fixing software installation and testing issues.
    Analyze the installation logs provided and update the installation
    configuration to fix any errors.

    Repository: {repo_name}

    Current installation configuration:
    {install_config}

    Error logs from installation/testing (last relevant lines):
    {error_logs}

    Your task:
    * Identify the root causes of the errors in the logs.
    * Modify the installation configuration to address these issues.
    * You might need to:
      - Add missing dependencies
      - Fix command syntax errors
      - Change installation order
      - Add environment variables
      - Modify test commands

    First, provide brief reasoning (<100 words) about what's causing the
    errors and your planned fixes.

    Then return the complete updated installation configuration as a valid
    JSON object in a markdown code block.
""")


# ---------------------------------------------------------------------------
# Agent CLI wrapper
# ---------------------------------------------------------------------------


def _run_agent(prompt: str, workspace: Path, timeout: int = 300) -> str | None:
    """Invoke `agent -p` with the given prompt and return stdout."""
    try:
        result = subprocess.run(
            [
                "agent", "-p", "-f",
                "--workspace", str(workspace.resolve()),
                "--output-format", "text",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            log.warning("agent exited %d", result.returncode)
            log.debug("stderr: %s", (result.stderr or "")[-500:])
        return result.stdout or ""
    except subprocess.TimeoutExpired:
        log.warning("agent timed out (%ds)", timeout)
        return None
    except FileNotFoundError:
        log.error("'agent' CLI not found. Is Cursor agent installed?")
        return None


# ---------------------------------------------------------------------------
# JSON extraction from agent output
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_from_text(text: str, expect_array: bool = False) -> dict | list | None:
    """Try to extract a JSON object or array from LLM output text."""
    if not text:
        return None

    # Try markdown code blocks first
    for m in _JSON_BLOCK_RE.finditer(text):
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # Try raw JSON in text
    pattern = _JSON_ARRAY_RE if expect_array else _JSON_OBJECT_RE
    for m in pattern.finditer(text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# File listing + content reading
# ---------------------------------------------------------------------------

_SETUP_FILE_GLOBS = [
    "README*", "readme*", "INSTALL*",
    "setup.py", "setup.cfg", "pyproject.toml",
    "requirements*.txt", "environment*.yml", "environment*.yaml",
    "Makefile", "Dockerfile", "docker-compose*.yml",
    "tox.ini", ".github/workflows/*.yml",
    "docs/install*", "docs/setup*", "docs/getting_started*",
    "CONTRIBUTING*",
]


def _list_repo_files(repo_dir: Path, max_depth: int = 3) -> list[str]:
    """List files in a repo up to max_depth, for the agent to analyze."""
    files = []
    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        # Skip hidden dirs, .git, __pycache__, etc.
        parts = rel.parts
        if any(p.startswith(".") or p == "__pycache__" or p == "node_modules" for p in parts):
            continue
        if len(parts) > max_depth:
            continue
        files.append(str(rel))
    return files


def _read_file_contents(repo_dir: Path, file_paths: list[str], max_chars: int = 100000) -> str:
    """Read and concatenate file contents for the recipe extraction prompt."""
    parts = []
    total = 0
    for fp in file_paths:
        full_path = repo_dir / fp
        if not full_path.exists() or not full_path.is_file():
            continue
        try:
            content = full_path.read_text(errors="replace")
        except Exception:
            continue

        # Truncate individual files
        if len(content) > 20000:
            content = content[:20000] + "\n... (truncated)"

        entry = f"<filename>{fp}</filename>\n<content>{content}</content>\n"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main recipe generation flow
# ---------------------------------------------------------------------------


def _identify_relevant_files(repo_dir: Path, repo_name: str) -> list[str]:
    """Phase 1: Ask agent to identify files containing setup info."""
    all_files = _list_repo_files(repo_dir)
    if not all_files:
        log.warning("No files found in %s", repo_dir)
        return []

    # Truncate file list if huge
    file_list_str = "\n".join(all_files[:2000])

    prompt = IDENTIFY_FILES_PROMPT.format(
        repo_name=repo_name,
        file_list=file_list_str,
    )

    output = _run_agent(prompt, workspace=repo_dir)
    if not output:
        log.warning("Agent returned no output for file identification")
        return _fallback_relevant_files(repo_dir)

    files = _extract_json_from_text(output, expect_array=True)
    if isinstance(files, list) and files:
        log.info("Agent identified %d relevant files for %s", len(files), repo_name)
        return [str(f) for f in files[:10]]

    log.warning("Could not parse file list from agent output. Using fallback.")
    return _fallback_relevant_files(repo_dir)


def _fallback_relevant_files(repo_dir: Path) -> list[str]:
    """Fallback: find setup-related files using glob patterns."""
    found = []
    for pattern in _SETUP_FILE_GLOBS:
        for path in repo_dir.glob(pattern):
            if path.is_file():
                found.append(str(path.relative_to(repo_dir)))
    return found[:10]


def generate_recipe(
    repo_dir: Path,
    repo_name: str,
    max_retries: int = 3,
) -> dict | None:
    """Generate an installation recipe for a repository.

    Uses cursor agent CLI in two phases:
    1. Identify relevant files
    2. Extract structured JSON recipe

    On failure, retries with error context.
    Returns the recipe dict or None.
    """
    repo_dir = Path(repo_dir)

    # Phase 1: Identify files
    relevant_files = _identify_relevant_files(repo_dir, repo_name)
    if not relevant_files:
        log.warning("No relevant files found for %s", repo_name)
        return None

    # Read file contents
    file_contents = _read_file_contents(repo_dir, relevant_files)
    if not file_contents.strip():
        log.warning("No readable content in relevant files for %s", repo_name)
        return None

    # Phase 2: Extract recipe (with retries)
    recipe = None
    last_error = ""

    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            prompt = EXTRACT_RECIPE_PROMPT.format(
                repo_name=repo_name,
                file_contents=file_contents,
            )
        else:
            # Retry with error context
            prompt = CORRECTION_PROMPT.format(
                repo_name=repo_name,
                install_config=json.dumps(recipe, indent=2) if recipe else "{}",
                error_logs=last_error[-3000:],
            )

        log.info(
            "Recipe generation attempt %d/%d for %s",
            attempt, max_retries, repo_name,
        )

        output = _run_agent(prompt, workspace=repo_dir, timeout=300)
        if not output:
            last_error = "Agent returned no output"
            continue

        parsed = _extract_json_from_text(output, expect_array=False)
        if not isinstance(parsed, dict):
            last_error = f"Could not parse JSON from agent output: {output[-500:]}"
            log.warning("Attempt %d: %s", attempt, last_error[:200])
            continue

        recipe = parsed

        # Validate required fields
        if not recipe.get("install") or not recipe.get("test_cmd"):
            last_error = f"Recipe missing required fields: {list(recipe.keys())}"
            log.warning("Attempt %d: %s", attempt, last_error)
            continue

        # Set defaults
        recipe.setdefault("python", "3.10")
        recipe.setdefault("pre_install", [])
        recipe.setdefault("pip_packages", [])
        recipe.setdefault("reqs_path", [])
        recipe.setdefault("env_yml_path", [])

        log.info("Successfully generated recipe for %s on attempt %d", repo_name, attempt)
        return recipe

    log.warning("Failed to generate recipe for %s after %d attempts", repo_name, max_retries)
    return None
