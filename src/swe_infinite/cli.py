"""CLI entry points for swe-infinite and swe-eval commands."""

from __future__ import annotations


def main() -> None:
    """Entry point for the ``swe-infinite`` command (dataset generator)."""
    from swe_infinite.pipeline import main as _pipeline_main

    _pipeline_main()


def eval_main() -> None:
    """Entry point for the ``swe-eval`` command (evaluation harness)."""
    from swe_infinite.eval import main as _eval_main

    _eval_main()


def pull_main() -> None:
    """Entry point for the ``swe-pull`` command (download tasks from Hippius)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="swe-pull",
        description="Download SWE-Infinite tasks from the Hippius decentralized bucket.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./dataset",
        help="Local directory to save downloaded tasks (default: ./dataset)",
    )
    parser.add_argument(
        "--bucket", "-b",
        type=str,
        default="swe-infinite-dataset",
        help="Hippius S3 bucket name (default: swe-infinite-dataset)",
    )
    args = parser.parse_args()

    from pathlib import Path

    from swe_infinite.hippius import pull_all

    output_dir = Path(args.output)
    downloaded = pull_all(output_dir, bucket=args.bucket)

    if downloaded:
        print(f"Downloaded {len(downloaded)} new task(s) to {output_dir}/")
        for p in downloaded:
            print(f"  {p.name}")
    else:
        print(f"No new tasks to download (output dir: {output_dir}/)")

    # Also report totals
    existing = list(output_dir.glob("*.json")) if output_dir.exists() else []
    print(f"Total local tasks: {len(existing)}")
    sys.exit(0)
