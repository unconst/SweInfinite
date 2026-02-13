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

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | Recommended | GitHub personal access token. Without it, API rate limit is 60 req/hr vs 5,000. |
| `ANTHROPIC_API_KEY` | Optional | Enables LLM-based quality scoring via Claude. |
| `OPENAI_API_KEY` | Optional | Fallback LLM provider for quality scoring. |

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

## Evaluation

Evaluate generated tasks using the Cursor agent CLI:

```bash
# Evaluate all tasks in dataset/
swe-eval

# Evaluate specific tasks
swe-eval dataset/owner__repo-42.json

# With options
swe-eval --agent-timeout 300 --model sonnet-4
```

Results are written to `results/eval_<timestamp>.json`.

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

## Project Structure

```
src/swe_infinite/
  __init__.py            Package version
  __main__.py            python -m swe_infinite support
  cli.py                 CLI entry points (swe-infinite, swe-eval)
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

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker (optional, for isolated validation via `--docker`)
