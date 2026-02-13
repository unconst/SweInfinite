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
