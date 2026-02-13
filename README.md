# SWE Infinite

An automated factory for generating validated [SWE-bench](https://www.swebench.com/)-style benchmark tasks. Continuously mines GitHub for merged pull requests that fix real issues, extracts solution and test patches, validates them through execution, and produces quality-scored task instances.

## Quick Start

**Prerequisites:** Install [Cursor IDE](https://www.cursor.com/) and enable the `agent` CLI (see [Installing Cursor Agent](#installing-cursor-agent)). Verify with `agent --version`.

```bash
# Create a virtual environment and install the package
uv venv
source .venv/bin/activate
uv pip install -e .

# Set required environment variables
export GITHUB_TOKEN=ghp_...

# Run the generator (continuous, fault-tolerant) tasks go to dataset/
swe-infinite

# Produce one task into dataset and exit
swe-infinite --once

# Check pipeline stats
swe-infinite --status
```

Or without installing:

```bash
uv run swe-infinite
uv run swe-eval
```

## How It Works

The pipeline runs as a two-phase loop (annotate, then extract) with several sub-phases:

1. **Annotate** (fast, API-only) -- Downloads PR events from GitHub Archive, enriches each merged PR via the GitHub API, links to the issue the PR fixes, and stores metadata.
2. **Extract** (expensive, involves cloning) -- Picks the best annotated candidate, clones the repo, computes diffs, splits into solution and test patches, checks license and repo quality.
3. **Test Generation** (optional, on by default) -- If a PR has no test changes, generates tests via LLM and formats them as a unified diff.
4. **Validation** (optional, enable with `--validate`) -- Runs the test suite before and after applying the solution patch to populate `FAIL_TO_PASS` and `PASS_TO_PASS` fields (local venv or Docker).
5. **Quality Assessment** (optional, enable with `--quality`) -- Scores problem statement clarity, test quality, and difficulty via LLM (with heuristic fallback). Cleans issue text and checks for overlap with existing SWE-bench datasets.
6. **Storage** -- Saves the final task as a JSON file to `dataset/` and records it in the SQLite database.
7. **Evaluation** (optional, enable with `--eval`) -- Runs a Cursor agent on the generated task and records whether it resolves the issue.

The process handles errors gracefully and runs indefinitely. Logs go to stderr and a rotating log file (`swe_infinite.log`, 50 MB, 5 backups). Stop cleanly with `Ctrl-C`.

## Configuration

### Generator Flags (`swe-infinite`)

**General:**

| Flag | Default | Description |
|---|---|---|
| `--once` | off | Produce one task then exit. |
| `--status` | off | Print pipeline stats and exit. |
| `--db PATH` | `pipeline.db` | Path to SQLite database. |

**Pipeline phases** (opt-in -- off by default):

| Flag | Default | Description |
|---|---|---|
| `--validate` | off | Enable execution validation (populates `FAIL_TO_PASS` / `PASS_TO_PASS`). |
| `--quality` | off | Enable LLM quality scoring. |
| `--decontamination` | off | Enable overlap check against known benchmarks (SWE-bench, etc.). |
| `--docker` | off | Use Docker for isolated validation environments (requires `--validate`). |
| `--no-generate-tests` | off | Disable LLM test generation for PRs without test changes. |
| `--eval` | off | Run Cursor agent evaluation on each generated task. |
| `--eval-model NAME` | `opus-4.6` | Model for agent evaluation. |
| `--eval-timeout N` | 300 | Agent timeout in seconds for evaluation. |

**Repo filters:**

| Flag | Default | Description |
|---|---|---|
| `--min-stars N` | 5 | Minimum GitHub stars for the repo quality gate. |
| `--min-contributors N` | 0 | Minimum number of contributors. |
| `--require-ci` | off | Require CI/CD configuration in the repo. |
| `--require-tests` | off | Require a test framework to be detected. |
| `--no-allow-archived` | off | Skip archived repositories. |
| `--no-allow-non-permissive` | off | Reject repos with non-permissive licenses (GPL, AGPL, etc.). |
| `--languages LIST` | all | Comma-separated list of languages (e.g. `python,go`). Supported: python, typescript, javascript, java, go. |

**Issue / PR filters:**

| Flag | Default | Description |
|---|---|---|
| `--min-issue-length N` | 10 | Minimum issue body length in characters. |
| `--no-allow-no-issue` | off | Reject PRs with no linked issue (default: accept them and generate a problem statement via LLM). |

**Patch filters:**

| Flag | Default | Description |
|---|---|---|
| `--no-patch-checks` | off | Disable patch complexity checks (file count, line count, config-only, comment-only). Checks are on by default. |
| `--max-patch-files N` | 15 | Maximum files in solution patch. |
| `--min-patch-lines N` | 3 | Minimum changed lines in solution patch. |
| `--max-patch-lines N` | 1000 | Maximum changed lines in solution patch. |

**Quality filters:**

| Flag | Default | Description |
|---|---|---|
| `--min-quality-score N` | 2 | Minimum quality score (1-5) for issue text and test dimensions (requires `--quality`). |

### Evaluation Flags (`swe-eval`)

| Flag | Default | Description |
|---|---|---|
| `--agent-timeout N` | 300 | Agent timeout in seconds. |
| `--skip-install` | off | Skip dependency installation step. |
| `--model NAME` | auto | Model for the Cursor agent (e.g. `sonnet-4`, `gpt-5`). |
| `--cleanup` | off | Remove eval working directories after each task. |
| `--update-tasks` | off | Write eval results back into each task JSON file. |

## Project Structure

```
src/swe_infinite/
  __init__.py            Package version
  __main__.py            python -m swe_infinite support
  cli.py                 CLI entry points (swe-infinite, swe-eval)
  paths.py               Centralized runtime path definitions
  pipeline.py            Main generation pipeline (annotate/extract orchestrator)
  eval.py                Evaluation harness (Cursor agent)
  db.py                  SQLite schema and CRUD helpers
  gharchive.py           GitHub Archive download and event parsing
  repo_ops.py            Git clone, diff computation, patch splitting
  task_store.py          Task JSON building and DB storage
  versioning.py          Git tag to version mapping
  filters.py             Legacy PR/issue candidate filtering (unused by current pipeline)
  validator.py           Execution validation (FAIL_TO_PASS / PASS_TO_PASS)
  docker_env.py          Docker-based isolated execution environments
  recipe_generator.py    Install recipe generation (LLM + heuristics)
  test_generator.py      LLM-based test generation for PRs without test changes
  quality_scorer.py      LLM-based quality scoring and issue cleanup
  repo_quality.py        Repository quality gate (stars, CI, tests)
  decontamination.py     Overlap check against existing SWE-bench datasets
  language_support.py    Multi-language handlers (Python, TS, Java, Go)

dataset/                 Generated task JSON files (runtime)
results/                 Evaluation result files (runtime)
workspace/               Cached repos, validation dirs, eval dirs (runtime)
pipeline.db              SQLite database (runtime)
```

## Requirements

### System Tools

| Tool | Required | Used For |
|---|---|---|
| **Python 3.11+** | Yes | Runtime |
| **[uv](https://docs.astral.sh/uv/)** | Yes | Virtual environments and package installation (`uv venv`, `uv pip install`) |
| **[git](https://git-scm.com/)** | Yes | Cloning repos, computing diffs, checking out commits, reading tags |
| **[Cursor](https://www.cursor.com/)** (provides the `agent` CLI) | **Yes** | All LLM operations: synthetic problem statements, test generation, quality scoring, install recipe generation, and evaluation. See [Installing Cursor Agent](#installing-cursor-agent) below. |
| **[Docker](https://www.docker.com/)** | Optional | Isolated validation environments (enable with `--docker`) |

### Installing Cursor Agent

The Cursor agent CLI (`agent`) is **required** — it powers all LLM operations in the pipeline (synthetic problem statement generation, test generation, quality scoring, install recipe generation, and evaluation).

1. **Install [Cursor IDE](https://www.cursor.com/)**
   Download and install Cursor from [cursor.com](https://www.cursor.com/). Cursor is available for macOS, Linux, and Windows.

2. **Enable the `agent` CLI command**
   Open Cursor IDE and install the CLI command via the Command Palette:
   - Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Linux/Windows)
   - Type **"Install 'agent' command"** and select it
   - This adds the `agent` binary to your PATH (typically `/usr/local/bin/agent`)

3. **Verify installation**

   ```bash
   agent --version
   ```

   If this prints a version number, you're all set. If not, make sure the Cursor CLI directory is on your `PATH`.

> **Note:** The `agent` command uses your Cursor subscription for LLM access — no separate OpenAI or Anthropic API keys are needed.

### Language-Specific Tools (for multi-language validation)

The pipeline validates tasks by running test suites. Depending on which languages you process, you need the corresponding toolchains installed:

| Language | Tools Needed |
|---|---|
| **Python** | `python3`, `pytest` (installed automatically into venvs) |
| **JavaScript/TypeScript** | `node`, and one of: `npm`, `yarn`, `pnpm`, or `bun`; test runners like `jest` are project-local |
| **Java** | `mvn` (Maven) or `gradle` / `./gradlew` |
| **Go** | `go` |

By default all supported languages are enabled. Use `--languages python` (or any comma-separated subset) to restrict to only the languages you have tooling for.

### API Keys / Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | **Recommended** | GitHub personal access token. Without it the API rate limit is 60 req/hr vs 5,000. |
> **No separate LLM API keys are needed.** All LLM operations (quality scoring, synthetic problem statements, test generation) run through the Cursor agent CLI. Make sure `agent` is on your PATH (see [Installing Cursor Agent](#installing-cursor-agent)).

## Running the Full Pipeline

Make sure all required tools are installed, then:

```bash
# 1. Create a virtual environment and install the package
uv venv
source .venv/bin/activate
uv pip install -e .

# 2. Export your keys
export GITHUB_TOKEN="ghp_..."

# 3. Run continuously (generates tasks forever, Ctrl-C to stop)
swe-infinite
```

### Common Invocations

```bash
# Full pipeline, continuous, with synthetic problem statements (default)
GITHUB_TOKEN=ghp_... swe-infinite

# Generate one task and exit
GITHUB_TOKEN=ghp_... swe-infinite --once

# Full quality pipeline (validation + scoring + decontamination)
swe-infinite --validate --quality --decontamination

# Python-only, Docker isolation, 10+ stars
swe-infinite --languages python --docker --validate --min-stars 10

# Reject PRs without linked issues (real bug reports only)
swe-infinite --no-allow-no-issue

# Generate + evaluate in one step
swe-infinite --once --eval --eval-model sonnet-4

# Check pipeline statistics
swe-infinite --status
```

### Evaluation (requires Cursor agent CLI)

```bash
# Evaluate all tasks in dataset/
swe-eval

# Evaluate specific tasks with options
swe-eval dataset/owner__repo-42.json --agent-timeout 300 --model sonnet-4
```
