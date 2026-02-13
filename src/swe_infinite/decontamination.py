"""
Decontamination check for the SWE-rebench pipeline.

Ensures generated tasks don't overlap with existing well-known benchmarks:
  - princeton-nlp/SWE-bench
  - princeton-nlp/SWE-bench_Lite
  - princeton-nlp/SWE-bench_Verified

Downloads and caches known instance IDs, then checks new tasks
for overlap on (repo, pr_number) or (repo, base_commit).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger("swe-infinite.decontamination")

from .paths import DECONTAMINATION_CACHE_DIR as CACHE_DIR

# Known benchmark datasets on HuggingFace
_BENCHMARK_DATASETS = [
    {
        "name": "SWE-bench",
        "hf_path": "princeton-nlp/SWE-bench",
        "split": "test",
    },
    {
        "name": "SWE-bench_Lite",
        "hf_path": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
    },
    {
        "name": "SWE-bench_Verified",
        "hf_path": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
    },
]


# ---------------------------------------------------------------------------
# Known instance cache
# ---------------------------------------------------------------------------


class DecontaminationIndex:
    """Index of known benchmark instances for overlap detection."""

    def __init__(self) -> None:
        # Sets for fast lookup
        self.instance_ids: set[str] = set()
        self.repo_pr_pairs: set[tuple[str, int]] = set()
        self.repo_commit_pairs: set[tuple[str, str]] = set()
        self._loaded = False

    def load(self, force_refresh: bool = False) -> None:
        """Load known instances from cache or download fresh."""
        if self._loaded and not force_refresh:
            return

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / "known_instances.json"

        # Use cache if fresh (less than 7 days old)
        if cache_file.exists() and not force_refresh:
            try:
                age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                if age_hours < 168:
                    self._load_from_cache(cache_file)
                    return
            except OSError:
                pass

        # Download fresh data
        all_instances = []
        for dataset_info in _BENCHMARK_DATASETS:
            instances = _download_dataset_instances(dataset_info)
            all_instances.extend(instances)
            log.info(
                "Loaded %d instances from %s",
                len(instances), dataset_info["name"],
            )

        if all_instances:
            # Cache to disk
            cache_file.write_text(json.dumps(all_instances, indent=2))
            log.info("Cached %d total known instances", len(all_instances))

        self._build_index(all_instances)
        self._loaded = True

    def _load_from_cache(self, cache_file: Path) -> None:
        """Load from local cache file."""
        try:
            instances = json.loads(cache_file.read_text())
            self._build_index(instances)
            self._loaded = True
            log.info("Loaded %d known instances from cache", len(instances))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load decontamination cache: %s", e)

    def _build_index(self, instances: list[dict]) -> None:
        """Build lookup sets from instance list."""
        self.instance_ids.clear()
        self.repo_pr_pairs.clear()
        self.repo_commit_pairs.clear()

        for inst in instances:
            instance_id = inst.get("instance_id", "")
            repo = inst.get("repo", "")
            base_commit = inst.get("base_commit", "")

            if instance_id:
                self.instance_ids.add(instance_id)

            # Extract PR number from instance_id (format: owner__repo-PR_number)
            if "-" in instance_id:
                try:
                    pr_num = int(instance_id.rsplit("-", 1)[-1])
                    if repo:
                        self.repo_pr_pairs.add((repo, pr_num))
                except ValueError:
                    pass

            if repo and base_commit:
                self.repo_commit_pairs.add((repo, base_commit))

    def is_contaminated(
        self, instance_id: str, repo: str, pr_number: int, base_commit: str,
    ) -> str | None:
        """Check if a task overlaps with known benchmarks.

        Returns a reason string if contaminated, None if clean.
        """
        if not self._loaded:
            self.load()

        if instance_id in self.instance_ids:
            return f"Exact instance_id match: {instance_id}"

        if (repo, pr_number) in self.repo_pr_pairs:
            return f"Repo+PR overlap: {repo} PR#{pr_number}"

        if (repo, base_commit) in self.repo_commit_pairs:
            return f"Repo+commit overlap: {repo} @ {base_commit[:12]}"

        return None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_dataset_instances(dataset_info: dict) -> list[dict]:
    """Download instance metadata from a HuggingFace dataset.

    Uses the HuggingFace datasets API to fetch instance_id, repo,
    and base_commit fields.
    """
    hf_path = dataset_info["hf_path"]
    split = dataset_info["split"]

    # Try HuggingFace datasets library first
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_path, split=split)
        instances = []
        for row in ds:
            instances.append({
                "instance_id": row.get("instance_id", ""),
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", ""),
            })
        return instances
    except Exception as e:
        log.debug("datasets library failed for %s: %s", hf_path, e)

    # Fallback: try HuggingFace API directly
    try:
        api_url = f"https://datasets-server.huggingface.co/rows?dataset={hf_path}&config=default&split={split}&offset=0&length=1000"
        with httpx.Client(timeout=30) as client:
            resp = client.get(api_url)
            if resp.status_code == 200:
                data = resp.json()
                instances = []
                for row_data in data.get("rows", []):
                    row = row_data.get("row", {})
                    instances.append({
                        "instance_id": row.get("instance_id", ""),
                        "repo": row.get("repo", ""),
                        "base_commit": row.get("base_commit", ""),
                    })

                # Handle pagination
                total = data.get("num_rows_total", 0)
                offset = len(instances)
                while offset < total:
                    api_url = f"https://datasets-server.huggingface.co/rows?dataset={hf_path}&config=default&split={split}&offset={offset}&length=1000"
                    resp = client.get(api_url)
                    if resp.status_code != 200:
                        break
                    batch = resp.json()
                    for row_data in batch.get("rows", []):
                        row = row_data.get("row", {})
                        instances.append({
                            "instance_id": row.get("instance_id", ""),
                            "repo": row.get("repo", ""),
                            "base_commit": row.get("base_commit", ""),
                        })
                    new_count = len(batch.get("rows", []))
                    if new_count == 0:
                        break
                    offset += new_count

                return instances
    except Exception as e:
        log.debug("HuggingFace API failed for %s: %s", hf_path, e)

    log.warning("Could not download dataset %s", hf_path)
    return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_index: DecontaminationIndex | None = None


def get_index() -> DecontaminationIndex:
    """Get or create the global decontamination index."""
    global _index
    if _index is None:
        _index = DecontaminationIndex()
    return _index


def check_contamination(
    instance_id: str, repo: str, pr_number: int, base_commit: str,
) -> str | None:
    """Check if a task is contaminated (overlaps with known benchmarks).

    Returns reason string if contaminated, None if clean.
    Lazy-loads the decontamination index on first call.
    """
    index = get_index()
    try:
        index.load()
    except Exception as e:
        log.warning("Decontamination check failed (index load error): %s", e)
        return None  # Don't reject tasks if we can't check

    return index.is_contaminated(instance_id, repo, pr_number, base_commit)
