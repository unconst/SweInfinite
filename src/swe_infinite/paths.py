"""Centralized runtime path definitions.

All data directories (workspace, dataset, results, database, logs)
resolve relative to the current working directory at import time.
This means the tool stores data wherever you run it from.
"""

from __future__ import annotations

from pathlib import Path

#: Root for all runtime data — the working directory.
PROJECT_ROOT = Path.cwd()

#: Generated task JSON files.
DATASET_DIR = PROJECT_ROOT / "dataset"

#: Evaluation result files.
RESULTS_DIR = PROJECT_ROOT / "results"

#: Top-level workspace (repos, eval, validation, caches).
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

#: Cached repo clones.
REPOS_DIR = WORKSPACE_DIR / "repos"

#: Evaluation working directories.
EVAL_DIR = WORKSPACE_DIR / "eval"

#: Validation working directories.
VALIDATION_DIR = WORKSPACE_DIR / "validation"

#: Decontamination index cache.
DECONTAMINATION_CACHE_DIR = WORKSPACE_DIR / "decontamination_cache"

#: SQLite database.
DEFAULT_DB_PATH = PROJECT_ROOT / "pipeline.db"

#: Rotating log file.
LOG_FILE = PROJECT_ROOT / "swe_infinite.log"
