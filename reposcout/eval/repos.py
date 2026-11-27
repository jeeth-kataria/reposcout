"""One bare clone per upstream and one detached checkout per instance."""

import fcntl
import logging
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from reposcout.contracts.models import Instance

LOG = logging.getLogger(__name__)


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()


@contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@contextmanager
def worktree(
    instance: Instance, cache_dir: Path, *, cleanup: bool = False, source: Path | None = None
) -> Iterator[Path]:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", instance.repo):
        raise ValueError("invalid GitHub owner/repository")
    if not re.fullmatch(r"[\w.-]+", instance.instance_id) or instance.instance_id in {".", ".."}:
        raise ValueError("invalid instance ID")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", instance.base_commit):
        raise ValueError("expected a full Git commit SHA")
    commit = instance.base_commit.lower()
    key = instance.repo.replace("/", "__")
    cache_dir = cache_dir.resolve()
    mirror = cache_dir / "repos" / f"{key}.git"
    tree = cache_dir / "worktrees" / instance.instance_id
    mirror.parent.mkdir(parents=True, exist_ok=True)
    tree.parent.mkdir(parents=True, exist_ok=True)
    # Keep the instance lock until readers release their checkout. The repository
    # lock is held only for git metadata updates, allowing parallel different commits.
    with locked(cache_dir / "locks" / f"tree-{instance.instance_id}.lock"):
        with locked(cache_dir / "locks" / f"repo-{key}.lock"):
            if not mirror.exists():
                with TemporaryDirectory(dir=mirror.parent) as tmp:
                    staged = Path(tmp) / "repo.git"
                    git(
                        "clone",
                        "--bare",
                        "--",
                        str(source) if source else f"https://github.com/{instance.repo}.git",
                        str(staged),
                    )
                    staged.rename(mirror)
            if tree.exists():
                if (
                    git("rev-parse", "--git-common-dir", cwd=tree) != str(mirror)
                    or git("rev-parse", "HEAD", cwd=tree) != commit
                    or git("status", "--porcelain", cwd=tree)
                ):
                    raise ValueError(f"existing worktree does not match clean base commit: {tree}")
            else:
                try:
                    git("cat-file", "-e", f"{commit}^{{commit}}", cwd=mirror)
                except subprocess.CalledProcessError:
                    git("fetch", "--no-tags", "origin", commit, cwd=mirror)
                git("worktree", "add", "--detach", str(tree), commit, cwd=mirror)
        LOG.info(
            "Disk usage (KiB): %s",
            subprocess.check_output(["du", "-sk", str(mirror), str(tree)], text=True).strip(),
        )
        try:
            yield tree
        finally:
            if cleanup:
                with locked(cache_dir / "locks" / f"repo-{key}.lock"):
                    git("worktree", "remove", str(tree), cwd=mirror)
