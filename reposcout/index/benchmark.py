"""Measure a commit build, per-stage artifact sizes, and warm reopen."""

import argparse
import json
import sqlite3
import time
from pathlib import Path

from .repository import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("commit")
    parser.add_argument("--cache", type=Path, default=Path("cache/index"))
    parser.add_argument("--no-dense", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    index = build_index(args.repo, args.commit, args.cache, dense=not args.no_dense)
    elapsed = time.monotonic() - started
    manifest = index.manifest.copy()
    path = index.directory
    index.close()
    warm = time.monotonic()
    build_index(args.repo, args.commit, args.cache, dense=not args.no_dense).close()
    manifest["reopen_seconds"] = time.monotonic() - warm
    manifest["open_or_build_seconds"] = elapsed
    # dbstat attributes occupied pages to tables and secondary indexes.
    with sqlite3.connect(path / "index.sqlite") as db:
        pages = dict(db.execute("select name,sum(pgsize) from dbstat group by name"))
    manifest["bytes"] = {
        "postings": sum(size for name, size in pages.items() if name.startswith("postings")),
        "calls_and_imports": sum(
            pages.get(name, 0) for name in ("calls", "bases", "targets", "callers", "imports")
        ),
        "snapshot": (path / "index.sqlite").stat().st_size,
        "vectors": (path / "vectors.f16").stat().st_size if manifest["dense"] else 0,
        "shared_content_store": (path.parent / "content.sqlite").stat().st_size,
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
