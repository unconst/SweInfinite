"""
Docker-based isolated execution environment for the SWE-rebench pipeline.

Provides reproducible, sandboxed environments for:
  - Task validation (FAIL_TO_PASS / PASS_TO_PASS collection)
  - Agent evaluation
  - Install recipe testing

Each environment runs inside a Docker container built from a
per-(repo, version) image, ensuring reproducibility and safety.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("swe-infinite.docker_env")

from .paths import REPOS_DIR

# ---------------------------------------------------------------------------
# Dockerfile template
# ---------------------------------------------------------------------------

_DOCKERFILE_TEMPLATE = """\
FROM python:{python_version}-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git build-essential curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Pre-install commands (system deps from install_config)
{pre_install_commands}

WORKDIR /repo

# Copy the repository
COPY repo/ /repo/

# Create venv and install dependencies
RUN uv venv /repo/.venv --python python{python_version}
ENV VIRTUAL_ENV=/repo/.venv
ENV PATH="/repo/.venv/bin:$PATH"

# Install pytest and json report plugin
RUN uv pip install pytest pytest-json-report

# Install project dependencies
{install_commands}

# Default command
CMD ["bash"]
"""


# ---------------------------------------------------------------------------
# TestOutcome (canonical definition in validator.py)
# ---------------------------------------------------------------------------

from .validator import TestOutcome  # noqa: E402


# ---------------------------------------------------------------------------
# Docker environment
# ---------------------------------------------------------------------------


class DockerEnvironment:
    """Manages a Docker container for task validation/evaluation."""

    def __init__(self, task: dict) -> None:
        self.task = task
        self.instance_id = task["instance_id"]
        self.repo_name = task["repo"]
        self.base_commit = task["base_commit"]
        self.install_config = task.get("install_config") or {}
        self.container_id: str | None = None
        self.image_tag: str | None = None
        self._build_dir: tempfile.TemporaryDirectory | None = None

    def build(self) -> bool:
        """Build a Docker image for this task and start a container.

        Returns True on success.
        """
        python_ver = self.install_config.get("python", "3.10")
        safe_name = self.repo_name.replace("/", "__").lower()
        self.image_tag = f"swe-infinite/{safe_name}:{self.base_commit[:12]}"

        # Check if image already exists
        result = subprocess.run(
            ["docker", "image", "inspect", self.image_tag],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log.info("  [docker] Reusing existing image %s", self.image_tag)
            return self._start_container()

        # Build new image
        self._build_dir = tempfile.TemporaryDirectory(prefix="swe-docker-")
        build_path = Path(self._build_dir.name)

        # Copy repo into build context
        cached = REPOS_DIR / self.repo_name.replace("/", "__")
        repo_dest = build_path / "repo"
        if cached.exists():
            clone_result = subprocess.run(
                ["git", "clone", "--local", str(cached), str(repo_dest)],
                capture_output=True, text=True, timeout=120,
            )
            if clone_result.returncode != 0:
                log.error("  [docker] Local clone failed: %s", (clone_result.stderr or "")[:200])
                return False
        else:
            url = f"https://github.com/{self.repo_name}.git"
            clone_result = subprocess.run(
                ["git", "clone", "--filter=blob:none", url, str(repo_dest)],
                capture_output=True, text=True, timeout=180,
            )
            if clone_result.returncode != 0:
                log.error("  [docker] Remote clone failed: %s", (clone_result.stderr or "")[:200])
                return False

        if not repo_dest.exists():
            log.error("  [docker] Failed to clone repo for Docker build")
            return False

        # Checkout base commit
        checkout_result = subprocess.run(
            ["git", "checkout", self.base_commit, "--force"],
            cwd=repo_dest, capture_output=True, text=True, timeout=60,
        )
        if checkout_result.returncode != 0:
            log.error("  [docker] Checkout failed: %s", (checkout_result.stderr or "")[:200])
            return False

        # Generate Dockerfile
        pre_install = ""
        for cmd in self.install_config.get("pre_install", []):
            pre_install += f"RUN {cmd}\n"
        if not pre_install:
            pre_install = "# No pre-install commands"

        install_cmds = self._generate_install_commands(python_ver)

        dockerfile_content = _DOCKERFILE_TEMPLATE.format(
            python_version=python_ver,
            pre_install_commands=pre_install,
            install_commands=install_cmds,
        )
        (build_path / "Dockerfile").write_text(dockerfile_content)

        # Build image
        log.info("  [docker] Building image %s...", self.image_tag)
        result = subprocess.run(
            ["docker", "build", "-t", self.image_tag, "."],
            cwd=build_path, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.error("  [docker] Build failed: %s", result.stderr[-500:])
            return False

        log.info("  [docker] Image built successfully")
        return self._start_container()

    def _generate_install_commands(self, python_ver: str) -> str:
        """Generate Dockerfile RUN commands for installing dependencies."""
        lines = []

        # Requirements files
        for req_path in self.install_config.get("reqs_path", []):
            lines.append(f"RUN uv pip install -r {req_path} || true")

        # Extra pip packages
        pip_pkgs = self.install_config.get("pip_packages", [])
        if pip_pkgs:
            pkgs_str = " ".join(f'"{p}"' for p in pip_pkgs)
            lines.append(f"RUN uv pip install {pkgs_str} || true")

        # Main install command
        install_cmd = self.install_config.get("install", "")
        if install_cmd:
            # Normalize install commands
            install_cmd = install_cmd.replace("pip install", "uv pip install")
            lines.append(f"RUN {install_cmd} || true")
        else:
            # Best-effort install
            lines.append("RUN uv pip install -e . 2>/dev/null || uv pip install . 2>/dev/null || true")
            lines.append("RUN [ -f requirements.txt ] && uv pip install -r requirements.txt || true")

        if not lines:
            return "# No install commands"

        return "\n".join(lines)

    def _start_container(self) -> bool:
        """Start a container from the built image."""
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--network=none",  # No network access for safety
                "--memory=2g",
                "--cpus=2",
                "--name", f"swe-val-{self.instance_id[:40]}",
                self.image_tag,
                "sleep", "3600",  # Keep alive for 1 hour max
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            # Container name might already exist — remove and retry
            subprocess.run(
                ["docker", "rm", "-f", f"swe-val-{self.instance_id[:40]}"],
                capture_output=True, timeout=10,
            )
            result = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--network=none", "--memory=2g", "--cpus=2",
                    "--name", f"swe-val-{self.instance_id[:40]}",
                    self.image_tag, "sleep", "3600",
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                log.error("  [docker] Failed to start container: %s", result.stderr[:300])
                return False

        self.container_id = result.stdout.strip()[:12]
        log.info("  [docker] Container started: %s", self.container_id)
        return True

    def _exec(self, cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
        """Execute a command inside the container."""
        if not self.container_id:
            raise RuntimeError("No container running")
        return subprocess.run(
            ["docker", "exec", self.container_id, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )

    def apply_patch(self, patch: str) -> bool:
        """Apply a patch inside the container. Returns True on success."""
        if not patch.strip():
            return True

        # Write patch to temp file and copy into container
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            f.write(patch)
            temp_path = f.name

        try:
            subprocess.run(
                ["docker", "cp", temp_path, f"{self.container_id}:/repo/_patch.diff"],
                capture_output=True, timeout=10,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

        result = self._exec("cd /repo && git apply --allow-empty _patch.diff && rm _patch.diff")
        if result.returncode != 0:
            # Try --3way
            result = self._exec("cd /repo && git apply --3way _patch.diff 2>/dev/null; rm -f _patch.diff")
            if result.returncode != 0:
                log.warning("  [docker] git apply failed: %s", result.stderr[:300])
                return False

        return True

    def run_tests(
        self, test_files: list[str], timeout: int = 120,
    ) -> tuple[list[TestOutcome], str]:
        """Run pytest inside the container, collecting per-test results."""
        test_args = " ".join(test_files)
        cmd = (
            f"cd /repo && python -m pytest -x --tb=short -v "
            f"--json-report --json-report-file=/repo/.report.json "
            f"-W ignore::DeprecationWarning --no-header -rA "
            f"{test_args} 2>&1"
        )

        result = self._exec(cmd, timeout=timeout)
        output = result.stdout + result.stderr

        # Try to extract JSON report
        report_result = self._exec("cat /repo/.report.json 2>/dev/null", timeout=10)
        outcomes = []

        if report_result.returncode == 0 and report_result.stdout.strip():
            try:
                data = json.loads(report_result.stdout)
                for test in data.get("tests", []):
                    nodeid = test.get("nodeid", "")
                    outcome = test.get("outcome", "").upper()
                    if nodeid and outcome:
                        outcomes.append(TestOutcome(nodeid=nodeid, status=outcome))
            except json.JSONDecodeError:
                pass

        # Fallback to parsing verbose output
        if not outcomes:
            pattern = re.compile(
                r"^([\w/\.\-]+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)",
                re.MULTILINE,
            )
            for m in pattern.finditer(output):
                outcomes.append(TestOutcome(nodeid=m.group(1), status=m.group(2)))

        return outcomes, output

    def pip_freeze(self) -> str:
        """Get pip freeze output from the container."""
        result = self._exec("pip freeze 2>/dev/null || uv pip freeze 2>/dev/null")
        return result.stdout.strip() if result.returncode == 0 else ""

    def cleanup(self) -> None:
        """Stop and remove the container."""
        if self.container_id:
            subprocess.run(
                ["docker", "rm", "-f", self.container_id],
                capture_output=True, timeout=30,
            )
            self.container_id = None

        if self._build_dir:
            self._build_dir.cleanup()
            self._build_dir = None

    def __del__(self) -> None:
        self.cleanup()


# ---------------------------------------------------------------------------
# Utility: check Docker availability
# ---------------------------------------------------------------------------


def docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
