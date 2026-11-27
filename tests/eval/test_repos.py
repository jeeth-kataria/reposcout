from pathlib import Path

import pytest
from conftest import GitMinirepo

from reposcout.contracts.models import Instance
from reposcout.eval.repos import git, worktree


def test_cached_clone_detached_reuse_and_cleanup(minirepo_git: GitMinirepo, tmp_path: Path) -> None:
    root, base, head = minirepo_git
    cache = tmp_path / "cache"
    task = Instance(instance_id="p-1", repo="demo/p", base_commit=base, problem_statement="")
    with worktree(task, cache, source=root) as checkout:
        assert git("rev-parse", "HEAD", cwd=checkout) == base
        assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=checkout) == "HEAD"
    with worktree(task, cache, cleanup=True, source=root) as other:
        assert checkout == other
    assert not checkout.exists()
    with worktree(
        task.model_copy(update={"base_commit": head, "instance_id": "p-2"}), cache, source=root
    ) as next_tree:
        assert git("rev-parse", "HEAD", cwd=next_tree) == head
        (next_tree / "untracked").touch()
    with (
        pytest.raises(ValueError, match="clean base commit"),
        worktree(
            task.model_copy(update={"base_commit": head, "instance_id": "p-2"}), cache, source=root
        ),
    ):
        pass
    assert len(list((cache / "repos").iterdir())) == 1


def test_worktree_fetches_commit_created_after_initial_clone(
    minirepo_git: GitMinirepo, tmp_path: Path
) -> None:
    repo, base, _ = minirepo_git
    cache = tmp_path / "cached"
    task = Instance(instance_id="p-1", repo="owner/test", base_commit=base, problem_statement="")
    with worktree(task, cache, source=repo):
        pass
    (repo / "extra.py").write_text("VALUE = 1\n")
    git("add", "extra.py", cwd=repo)
    git(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Add fixture revision",
        cwd=repo,
    )
    new_commit = git("rev-parse", "HEAD", cwd=repo)
    task = task.model_copy(update={"base_commit": new_commit.upper(), "instance_id": "p-2"})
    with worktree(task, cache, source=repo) as checkout:
        assert git("rev-parse", "HEAD", cwd=checkout) == new_commit
    with worktree(task, cache, source=repo) as checkout:
        assert (checkout / "extra.py").read_text() == "VALUE = 1\n"
