"""Hippius decentralized S3 storage integration.

Uploads generated task JSON files to a public Hippius S3 bucket so the
dataset accumulates on decentralized storage with blockchain-anchored
timestamps.  Also provides helpers for listing and downloading tasks.

Authentication (checked in order):

1. **Access keys** (recommended) -- Set ``HIPPIUS_ACCESS_KEY`` and
   ``HIPPIUS_SECRET_KEY`` env vars.  Keys start with ``hip_``.
2. **Legacy mnemonic** -- Set ``HIPPIUS_MNEMONIC`` (subaccount seed
   phrase).  Uses ``base64(mnemonic)`` as access key.

If none of the above are configured, all operations silently return
without error so the pipeline can run without Hippius.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path

log = logging.getLogger("swe-infinite.hippius")

HIPPIUS_S3_ENDPOINT = "s3.hippius.com"
DEFAULT_BUCKET = "swe-infinite-dataset"

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def get_client():
    """Create a ``minio.Minio`` client using Hippius credentials.

    Checks for access keys first (``HIPPIUS_ACCESS_KEY`` /
    ``HIPPIUS_SECRET_KEY``), then falls back to the legacy mnemonic
    (``HIPPIUS_MNEMONIC``).  Returns ``None`` if no credentials are set
    or the ``minio`` package is not installed.
    """
    try:
        from minio import Minio
    except ImportError:
        log.warning("minio package not installed — pip install minio")
        return None

    # Preferred: modern access keys (hip_...)
    ak = os.environ.get("HIPPIUS_ACCESS_KEY", "").strip()
    sk = os.environ.get("HIPPIUS_SECRET_KEY", "").strip()
    if ak and sk:
        log.debug("Using Hippius access-key auth")
        return Minio(
            HIPPIUS_S3_ENDPOINT,
            access_key=ak,
            secret_key=sk,
            secure=True,
            region="decentralized",
        )

    # Fallback: legacy subaccount mnemonic
    mnemonic = os.environ.get("HIPPIUS_MNEMONIC", "").strip()
    if mnemonic:
        log.debug("Using Hippius legacy mnemonic auth")
        access_key = base64.b64encode(mnemonic.encode("utf-8")).decode("utf-8")
        return Minio(
            HIPPIUS_S3_ENDPOINT,
            access_key=access_key,
            secret_key=mnemonic,
            secure=True,
            region="decentralized",
        )

    return None


def ensure_bucket(client, bucket: str = DEFAULT_BUCKET) -> bool:
    """Create *bucket* if it doesn't exist and set a public-read policy.

    Returns ``True`` if the bucket is ready, ``False`` on error.
    """
    if client is None:
        return False

    try:
        if not client.bucket_exists(bucket):
            log.info("Creating Hippius bucket: %s", bucket)
            client.make_bucket(bucket)

            # Set public-read policy so anyone can download tasks
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket}/*"],
                    }
                ],
            }
            client.set_bucket_policy(bucket, json.dumps(policy))
            log.info("Bucket %s created with public-read policy", bucket)
        else:
            log.debug("Bucket %s already exists", bucket)
        return True
    except Exception:
        log.exception("Failed to ensure Hippius bucket %s", bucket)
        return False


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _object_exists(client, bucket: str, object_name: str) -> bool:
    """Return ``True`` if *object_name* already exists in *bucket*."""
    try:
        client.stat_object(bucket, object_name)
        return True
    except Exception:
        # stat_object raises an error (S3Error with code NoSuchKey) when
        # the object does not exist.  Treat any exception as "not found"
        # so the caller can proceed with the upload.
        return False


def upload_task(task_path: Path, bucket: str = DEFAULT_BUCKET) -> bool:
    """Upload a single task JSON file to the Hippius bucket.

    The object key is ``tasks/{filename}``.
    If the exact object already exists in the bucket the upload is skipped
    and the function returns ``True`` (success, no duplicate created).
    Returns ``True`` on success, ``False`` on any error (including missing
    credentials).
    """
    client = get_client()
    if client is None:
        return False

    if not ensure_bucket(client, bucket):
        return False

    object_name = f"tasks/{task_path.name}"

    # --- Deduplication check ---
    if _object_exists(client, bucket, object_name):
        log.info("Task already exists in bucket, skipping upload: s3://%s/%s", bucket, object_name)
        return True

    try:
        data = task_path.read_bytes()
        client.put_object(
            bucket,
            object_name,
            BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
        log.info("Uploaded %s -> s3://%s/%s", task_path.name, bucket, object_name)
        return True
    except Exception:
        log.exception("Failed to upload %s to Hippius", task_path.name)
        return False


# ---------------------------------------------------------------------------
# List / Download
# ---------------------------------------------------------------------------


def list_tasks(bucket: str = DEFAULT_BUCKET) -> list[str]:
    """Return a list of object names (``tasks/*.json``) in the bucket."""
    client = get_client()
    if client is None:
        return []

    try:
        objects = client.list_objects(bucket, prefix="tasks/", recursive=True)
        return [obj.object_name for obj in objects if obj.object_name.endswith(".json")]
    except Exception:
        log.exception("Failed to list tasks in bucket %s", bucket)
        return []


def download_task(object_name: str, output_dir: Path, bucket: str = DEFAULT_BUCKET) -> Path | None:
    """Download a single object to *output_dir*.  Returns the local path."""
    client = get_client()
    if client is None:
        return None

    filename = Path(object_name).name
    dest = output_dir / filename

    try:
        response = client.get_object(bucket, object_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in response.stream(1024 * 64):
                f.write(chunk)
        response.close()
        response.release_conn()
        return dest
    except Exception:
        log.exception("Failed to download %s", object_name)
        return None


def pull_all(output_dir: Path, bucket: str = DEFAULT_BUCKET) -> list[Path]:
    """Download all task JSONs from the bucket, skipping existing files.

    Returns the list of newly downloaded file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_objects = list_tasks(bucket)

    if not remote_objects:
        log.info("No tasks found in bucket %s", bucket)
        return []

    downloaded: list[Path] = []
    skipped = 0

    for obj_name in remote_objects:
        filename = Path(obj_name).name
        local_path = output_dir / filename

        if local_path.exists():
            skipped += 1
            continue

        result = download_task(obj_name, output_dir, bucket)
        if result is not None:
            downloaded.append(result)

    log.info(
        "Pull complete: %d downloaded, %d skipped (already exist), %d total remote",
        len(downloaded), skipped, len(remote_objects),
    )
    return downloaded
