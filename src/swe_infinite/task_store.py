"""
Task storage for the SWE-rebench pipeline.

Writes task instances as JSON files matching the SWE-rebench dataset schema
(paper Table 5, Appendix K). Also persists tasks to the SQLite database.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .db import DEFAULT_DB_PATH, get_connection, insert_task
from .paths import DATASET_DIR as TASKS_DIR

log = logging.getLogger("swe-infinite.task_store")


# ---------------------------------------------------------------------------
# Instance ID generation (paper format: owner__repo-PR_number)
# ---------------------------------------------------------------------------


def make_instance_id(repo_name: str, pr_number: int) -> str:
    """Generate a SWE-rebench-style instance_id.

    Format: owner__repo-PR_number
    Example: python-humanize__humanize-42
    """
    # repo_name is "owner/repo" -> "owner__repo"
    safe = repo_name.replace("/", "__")
    return f"{safe}-{pr_number}"


# ---------------------------------------------------------------------------
# Build SWE-rebench task dict
# ---------------------------------------------------------------------------


def build_task_instance(
    extracted: dict,
    version: str | None = None,
    install_config: dict | None = None,
    environment_setup_commit: str | None = None,
    validation_result: object | None = None,
    quality_scores: object | None = None,
) -> dict:
    """Build a task instance dict matching SWE-rebench schema.

    Args:
        extracted: Output from repo_ops.extract_candidate()
        version: Normalized version string from versioning module
        install_config: Installation recipe from recipe_generator
        environment_setup_commit: The commit used for env setup (from version group)
        validation_result: Output from validator.validate_task() (ValidationResult)
        quality_scores: Output from quality_scorer.assess_quality() (QualityScores)

    Returns:
        dict matching SWE-rebench schema (paper Table 5 / Appendix K)
    """
    repo_name = extracted["repo_name"]
    pr_number = extracted["pr_number"]
    instance_id = make_instance_id(repo_name, pr_number)

    # Build problem_statement from issue title + body
    issue_title = extracted.get("issue_title", "")
    issue_body = extracted.get("issue_body", "")
    problem_statement = f"{issue_title}\n\n{issue_body}".strip()

    # Use cleaned problem statement if available from quality scoring
    if quality_scores and hasattr(quality_scores, "cleaned_problem_statement"):
        cleaned = quality_scores.cleaned_problem_statement
        if cleaned:
            problem_statement = cleaned

    # Extract LLM scores if available
    llm_score = {
        "difficulty_score": None,
        "issue_text_score": None,
        "test_score": None,
    }
    if quality_scores and hasattr(quality_scores, "to_dict"):
        llm_score = quality_scores.to_dict()

    # Build meta dict
    meta = {
        "commit_name": extracted.get("commit_name", "head_commit"),
        "num_modified_files": extracted.get("num_modified_files", 0),
        "has_test_patch": bool(extracted.get("test_patch")),
        "is_lite": extracted.get("num_modified_files", 0) <= 3,
        "llm_score": llm_score,
    }

    # Extract FAIL_TO_PASS and PASS_TO_PASS from validation result
    fail_to_pass = []
    pass_to_pass = []
    requirements = ""
    if validation_result and hasattr(validation_result, "fail_to_pass"):
        fail_to_pass = validation_result.fail_to_pass or []
        pass_to_pass = validation_result.pass_to_pass or []
        requirements = getattr(validation_result, "requirements", "") or ""

    task = {
        "instance_id": instance_id,
        "repo": repo_name,
        "base_commit": extracted.get("base_commit", ""),
        "version": version or extracted.get("version", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "problem_statement": problem_statement,
        "patch": extracted.get("solution_patch", ""),
        "test_patch": extracted.get("test_patch", ""),
        "hints_text": "",  # Could be populated from issue comments
        "environment_setup_commit": environment_setup_commit or extracted.get("base_commit", ""),
        "install_config": install_config or {},
        "meta": meta,
        "license_name": extracted.get("license_name", ""),
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
        "requirements": requirements,
        "environment": "",    # Populated after successful env setup (conda env export)
    }

    return task


# ---------------------------------------------------------------------------
# Write to disk + database
# ---------------------------------------------------------------------------


def store_task_json(task: dict, output_dir: Path = TASKS_DIR) -> Path:
    """Write a task instance to a JSON file on disk.

    File: tasks/{instance_id}.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_id = task["instance_id"]
    # Make filename filesystem-safe
    safe_name = instance_id.replace("/", "__")
    out_path = output_dir / f"{safe_name}.json"

    # Serialize install_config and meta as nested objects (not strings)
    out_path.write_text(json.dumps(task, indent=2, default=str))
    log.info("Stored task JSON: %s", out_path)
    return out_path


def store_task_db(task: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Persist a task instance to the SQLite database."""
    conn = get_connection(db_path)
    try:
        db_row = {
            "instance_id": task["instance_id"],
            "repo_name": task["repo"],
            "pr_number": int(task["instance_id"].rsplit("-", 1)[-1]),
            "base_commit": task["base_commit"],
            "version": task["version"],
            "problem_statement": task["problem_statement"],
            "patch": task["patch"],
            "test_patch": task["test_patch"],
            "install_config": json.dumps(task.get("install_config", {})),
            "license_name": task["license_name"],
            "meta": json.dumps(task.get("meta", {})),
            "created_at": task["created_at"],
        }
        insert_task(conn, db_row)
    finally:
        conn.close()


def store_task(
    task: dict,
    output_dir: Path = TASKS_DIR,
    db_path: Path = DEFAULT_DB_PATH,
) -> Path:
    """Store a task to both JSON file and database."""
    json_path = store_task_json(task, output_dir)
    store_task_db(task, db_path)
    return json_path


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


def store_batch(
    tasks: list[dict],
    output_dir: Path = TASKS_DIR,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[Path]:
    """Store multiple tasks. Returns list of JSON file paths."""
    paths = []
    for task in tasks:
        try:
            p = store_task(task, output_dir, db_path)
            paths.append(p)
        except Exception:
            log.exception("Failed to store task %s", task.get("instance_id", "?"))
    log.info("Stored %d/%d tasks", len(paths), len(tasks))
    return paths


def load_task(path: Path) -> dict:
    """Load a task instance from a JSON file."""
    return json.loads(path.read_text())


def load_all_tasks(tasks_dir: Path = TASKS_DIR) -> list[dict]:
    """Load all task JSON files from the tasks directory."""
    if not tasks_dir.exists():
        return []
    tasks = []
    for p in sorted(tasks_dir.glob("*.json")):
        try:
            tasks.append(load_task(p))
        except Exception:
            log.warning("Failed to load %s", p)
    return tasks
