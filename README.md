# SWE Infinite

An automated factory for generating validated [SWE-bench](https://www.swebench.com/)-style benchmark tasks. Continuously mines GitHub for merged pull requests that fix real issues, extracts solution and test patches, validates them through execution, and produces quality-scored task instances.

## Quick Start

```bash
# Install the package (editable)
uv pip install -e .
export GITHUB_TOKEN=ghp_...

# Run the generator (continuous, fault-tolerant)
swe-infinite

# Produce one task and exit
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

The pipeline runs in five phases:

1. **Collection** -- Downloads PR events from GitHub Archive, enriches them via the GitHub API, and links to the issue each PR fixes.
2. **Pre-filtering** -- Checks language support, repository quality (stars, CI, test framework), clones the repo, computes diffs, and validates patch complexity.
3. **Validation** -- Runs the test suite before and after applying the solution patch to populate `FAIL_TO_PASS` and `PASS_TO_PASS` fields (local venv or Docker).
4. **Quality Assessment** -- Scores problem statement clarity, test quality, and difficulty via LLM (with heuristic fallback). Cleans issue text and checks for overlap with existing SWE-bench datasets.
5. **Storage** -- Saves the final task as a JSON file to `dataset/` and records it in the SQLite database.

The process handles errors gracefully and runs indefinitely. Logs go to stderr and a rotating log file (`swe_infinite.log`, 50 MB, 5 backups). Stop cleanly with `Ctrl-C`.

## Configuration

### Generator Flags (`swe-infinite`)

| Flag | Default | Description |
|---|---|---|
| `--once` | off | Produce one task then exit. |
| `--status` | off | Print pipeline stats and exit. |
| `--db PATH` | `pipeline.db` | Path to SQLite database. |
| `--skip-validation` | off | Skip execution validation (`FAIL_TO_PASS` / `PASS_TO_PASS` will be empty). |
| `--skip-quality` | off | Skip LLM quality scoring. |
| `--docker` | off | Run validation inside Docker containers for isolation. |
| `--min-stars N` | 5 | Minimum GitHub stars for the repo quality gate. |
| `--parallel N` | 1 | Number of parallel workers for task processing. |

### Evaluation Flags (`swe-eval`)

| Flag | Default | Description |
|---|---|---|
| `--agent-timeout N` | 300 | Agent timeout in seconds. |
| `--skip-install` | off | Skip dependency installation step. |
| `--model NAME` | auto | Model for the Cursor agent (e.g. `sonnet-4`, `gpt-5`). |
| `--cleanup` | off | Remove eval working directories after each task. |

<details>
<summary>Running as a background service</summary>

### macOS (LaunchAgent)

1. Edit `com.swe-infinite.plist` and set your `GITHUB_TOKEN`.
2. Install and start:

```bash
cp com.swe-infinite.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.swe-infinite.plist
```

3. Check status / stop:

```bash
launchctl list | grep swe-infinite
launchctl unload ~/Library/LaunchAgents/com.swe-infinite.plist
```

### Linux (systemd)

Create `/etc/systemd/system/swe-infinite.service`:

```ini
[Unit]
Description=SWE Infinite dataset generator
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/SWEInfinite
ExecStart=/path/to/SWEInfinite/.venv/bin/swe-infinite
Restart=always
RestartSec=30
Environment=GITHUB_TOKEN=ghp_...

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now swe-infinite
journalctl -u swe-infinite -f
```

</details>

## Decentralized Storage (Hippius)

Generated tasks are automatically uploaded to the [Hippius](https://hippius.com) decentralized S3 bucket, giving the dataset blockchain-anchored timestamps and censorship-resistant availability.

### Setup

Set your Hippius S3 access keys as environment variables (get them from [console.hippius.com/dashboard/settings](https://console.hippius.com/dashboard/settings)):

```bash
export HIPPIUS_ACCESS_KEY="hip_your_access_key_here"
export HIPPIUS_SECRET_KEY="your_secret_key_here"
```

When credentials are set, every task saved by the pipeline is also uploaded to the `swe-infinite-dataset` bucket on `s3.hippius.com`. If the variables are not set, the pipeline runs normally without uploading -- local storage is never blocked.

### Downloading the Dataset

Anyone can pull the full dataset without credentials:

```bash
# Install the package, then:
swe-pull

# Download to a custom directory
swe-pull --output ./my-tasks

# Use a different bucket
swe-pull --bucket my-custom-bucket
```

### Public URL

Tasks are publicly browsable at:

```
https://s3.hippius.com/swe-infinite-dataset/tasks/
```

Each task is stored as `tasks/{instance_id}.json`.

## Project Structure

```
src/swe_infinite/
  __init__.py            Package version
  __main__.py            python -m swe_infinite support
  cli.py                 CLI entry points (swe-infinite, swe-eval, swe-pull)
  hippius.py             Hippius S3 decentralized storage integration
  paths.py               Centralized runtime path definitions
  pipeline.py            Main generation pipeline (5-phase orchestrator)
  eval.py                Evaluation harness
  db.py                  SQLite schema and CRUD helpers
  gharchive.py           GitHub Archive download and event parsing
  repo_ops.py            Git clone, diff computation, patch splitting
  task_store.py          Task JSON building and DB storage
  versioning.py          Git tag to version mapping
  filters.py             PR/issue candidate filtering
  validator.py           Execution validation (FAIL_TO_PASS / PASS_TO_PASS)
  docker_env.py          Docker-based isolated execution environments
  recipe_generator.py    Install recipe generation (LLM + heuristics)
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
| **[Cursor](https://www.cursor.com/)** (provides the `agent` CLI) | Yes | Install recipe generation, evaluation harness, LLM-driven code tasks. Install Cursor IDE, then the `agent` command is available on your PATH. |
| **[Docker](https://www.docker.com/)** | Optional | Isolated validation environments (enable with `--docker`) |

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
| `OPENAI_API_KEY` | **Yes (or Anthropic)** | LLM provider for quality scoring, synthetic problem statements, and install recipes. |
| `ANTHROPIC_API_KEY` | **Yes (or OpenAI)** | Alternative LLM provider via Claude. Pipeline tries Anthropic first, falls back to OpenAI. Set at least one. |
| `ANTHROPIC_MODEL` | Optional | Override the default Anthropic model (e.g. `claude-sonnet-4-20250514`). |
| `OPENAI_MODEL` | Optional | Override the default OpenAI model (e.g. `gpt-4o`). |
| `HIPPIUS_ACCESS_KEY` | Optional | Hippius S3 access key for uploading tasks to decentralized storage. |
| `HIPPIUS_SECRET_KEY` | Optional | Hippius S3 secret key (paired with access key above). |
| `HIPPIUS_MNEMONIC` | Optional | Legacy Hippius subaccount mnemonic (fallback if access keys not set). |

> **LLM keys are needed for the full pipeline.** Without `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, quality scoring falls back to heuristics and synthetic problem statement generation (for PRs with no linked issue) is disabled.

## Running the Full Pipeline

Make sure all required tools are installed, then:

```bash
# 1. Install the package
source .venv/bin/activate
uv pip install -e .

# 2. Export your keys
export GITHUB_TOKEN="ghp_..."
export OPENAI_API_KEY="sk-..."          # or ANTHROPIC_API_KEY

# 3. Run continuously (generates tasks forever, Ctrl-C to stop)
swe-infinite
```

### Common Invocations

```bash
# Full pipeline, continuous, with synthetic problem statements (default)
GITHUB_TOKEN=ghp_... OPENAI_API_KEY=sk-... swe-infinite

# Generate one task and exit
GITHUB_TOKEN=ghp_... OPENAI_API_KEY=sk-... swe-infinite --once

# Python-only, Docker isolation, 10+ stars
swe-infinite --languages python --docker --min-stars 10

# Skip LLM scoring (heuristic fallback, no synthetic bug reports)
swe-infinite --skip-quality --no-allow-no-issue

# Reject PRs without linked issues (real bug reports only)
swe-infinite --no-allow-no-issue

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
