import sqlite3
import subprocess
from pathlib import Path

import pytest
from conftest import GitMinirepo

from reposcout.contracts.models import Span
from reposcout.index import build_index
from reposcout.index import cache as index_cache
from reposcout.index.lexical import terms


def test_identifier_tokens() -> None:
    assert set(terms("pkg.HTTPRate.rate_band")) == {
        "pkg.httprate.rate_band",
        "pkg",
        "httprate",
        "http",
        "rate",
        "rate_band",
        "band",
    }


def test_index_reads_exact_commit_and_graph(minirepo_git: GitMinirepo, tmp_path: Path) -> None:
    root, base, _ = minirepo_git
    index = build_index(root, base, tmp_path / "cache", dense=False)
    hit = index.bm25("rate_band", 8)
    assert "postbag/rates.py::RateCard.rate_band" in [h.symbol_id for h in hit[:3]]
    assert hit == index.bm25("rate_band", 8)
    assert "postbag/rates.py::RateCard.rate_band" in [
        s.id for s in index.callees("postbag/express.py::ExpressRateCard.describe")
    ]
    assert "postbag/labels.py::shipping_label" in [
        s.id for s in index.callers("postbag/rates.py::RateCard.quote")
    ]
    assert index.list_dir(".") == ["postbag/", "tests/"]
    assert "empty.py" in index.list_dir("postbag")
    assert index.symbols("postbag/empty.py") == []
    span = Span(path="postbag/express.py", start_line=3, end_line=100)
    read = index.read(span, max_lines=1)
    assert (read.text, read.span.end_line, read.truncated) == ('SERVICE_CODE = "X1"', 3, True)
    with pytest.raises(ValueError):
        index.read(Span(path="postbag/empty.py", start_line=1, end_line=1))
    with pytest.raises(ValueError):
        index.list_dir("../")
    with pytest.raises(ValueError):
        index.dense("rate")
    assert any(s.qualname == "customs_reference" for s in index.symbols("postbag/unfinished.py"))


def test_commit_reuse(
    minirepo_git: GitMinirepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, base, head = minirepo_git
    store = tmp_path / "cache"
    index = build_index(root, base, store, dense=False)
    index.close()
    second = build_index(root, head, store, dense=False)
    assert second.manifest["counts"]["parsed_files"] == 1
    assert second.manifest["counts"]["reused_files"] == len(list(root.rglob("*.py"))) - 1
    monkeypatch.setattr(index_cache, "_populate", lambda *args: pytest.fail("unneeded rebuild"))
    loaded = build_index(root, head, store, dense=False)
    assert loaded.directory == second.directory


def test_symlink_skipped(minirepo_git: GitMinirepo, tmp_path: Path) -> None:
    root = minirepo_git.root
    (root / "postbag" / "unsafe.py").symlink_to(tmp_path / "outside")
    subprocess.run(["git", "add", "postbag/unsafe.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=f@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-m",
            "Add link",
        ],
        cwd=root,
        check=True,
    )
    index = build_index(root, "HEAD", tmp_path / "cache", dense=False)
    with pytest.raises(FileNotFoundError):
        index.read(Span(path="postbag/unsafe.py", start_line=1, end_line=1))
    assert index.manifest["counts"]["skipped_special"] == 1
    db = sqlite3.connect(index.directory / "index.sqlite")
    assert db.execute("select count(*) from calls where callee is null").fetchone()[0] > 0


def test_module_chunks_skip_definitions_and_cover_changed_module_code(
    minirepo_git: GitMinirepo,
    tmp_path: Path,
) -> None:
    index = build_index(
        minirepo_git.root, minirepo_git.base_commit, tmp_path / "cache", dense=False
    )
    hits = index.bm25("SERVICE_CODE", 20)
    module_hits = [hit for hit in hits if hit.symbol_id == "postbag/express.py::<module>"]
    assert any(hit.span.start_line <= 3 <= hit.span.end_line < 6 for hit in module_hits)
    assert any("[…]" in hit.snippet for hit in module_hits)
