import os
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest


class GitMinirepo(NamedTuple):
    root: Path
    base_commit: str
    head_commit: str


@pytest.fixture
def minirepo_git(tmp_path: Path) -> GitMinirepo:
    """Two reproducible commits in an isolated, disposable copy of Postbag."""
    root = tmp_path / "postbag"
    shutil.copytree(
        Path(__file__).parent / "fixtures" / "minirepo",
        root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )
    env = {
        **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_AUTHOR_DATE": "2024-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2024-01-01T00:00:00Z",
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", f"core.hooksPath={os.devnull}", *args],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--initial-branch=main")
    git("add", ".")
    git("commit", "-m", "Add Postbag fixture")
    base = git("rev-parse", "HEAD")
    service = root / "postbag" / "express.py"
    service.write_text(service.read_text().replace('SERVICE_CODE = "X1"', 'SERVICE_CODE = "X2"'))
    git("commit", "-am", "Update express service code")
    return GitMinirepo(root, base, git("rev-parse", "HEAD"))
