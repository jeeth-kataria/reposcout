"""Build immutable commit indexes and reuse parsing and embeddings across worktrees."""

import contextlib
import fcntl
import hashlib
import io
import json
import logging
import os
import sqlite3
import subprocess
import tempfile
import time
import tokenize
import zlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, Field

from reposcout.contracts.models import Symbol

from . import dense as embeddings
from . import graph, lexical
from .parser import _is_test, extract_symbols

INDEX_VERSION = 1
GRAPH_VERSION = 2
_LOG = logging.getLogger(__name__)


class IndexConfig(BaseModel):
    max_file_bytes: int = Field(default=1_000_000, gt=0)
    module_chunk_lines: int = Field(default=60, gt=0, le=60)
    snippet_lines: int = Field(default=3, gt=0)
    model_id: str = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
    token_budget: int = Field(default=128, gt=0)
    batch_size: int = Field(default=32, gt=0)


SCHEMA = """
create table files(path text primary key, hash text);
create table symbols(id text primary key, path text, qualname text, name text,
                     start integer, data text);
create index symbol_paths on symbols(path,start);
create table imports(path text,alias text,target text);
create table bases(class_id text,base text);
create table calls(path text,caller text,target text,line integer,callee text);
create table units(path text, symbol text, start integer, end integer);
create virtual table postings using fts5(path, qualname, body, content='',
                                         tokenize="ascii tokenchars '._'");
"""


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def _cached_symbols(raw: list[dict], path: str) -> list[Symbol]:
    result = []
    for value in raw:
        qualname = value["qualname"]
        parent = value["parent"]
        result.append(
            Symbol(
                id=f"{path}::{qualname}",
                kind=value["kind"],
                name=value["name"],
                qualname=qualname,
                parent_id=f"{path}::{parent}" if parent else None,
                is_test=_is_test(path),
                span={"path": path, "start_line": value["start"], "end_line": value["end"]},
            )
        )
    return result


def _get_file(
    store: sqlite3.Connection,
    reader: BinaryIO,
    writer: BinaryIO,
    blob: str,
    path: str,
) -> tuple[str, str, list[Symbol], graph.Facts, bool]:
    row = store.execute(
        "select hash,source,symbols,facts from objects join aliases using(hash) where blob=?",
        (blob,),
    ).fetchone()
    if row:
        digest, packed, symbols, facts = row
        return (
            digest,
            zlib.decompress(packed).decode(),
            _cached_symbols(json.loads(symbols), path),
            graph.Facts.model_validate_json(facts),
            True,
        )
    writer.write(blob.encode() + b"\n")
    writer.flush()
    oid, kind, size = reader.readline().split()
    if kind != b"blob" or oid.decode() != blob:
        raise ValueError(f"unexpected git object for {path}")
    data = reader.read(int(size))
    if reader.read(1) != b"\n" or len(data) != int(size):
        raise OSError(f"short git blob read for {path}")
    digest = hashlib.sha256(data).hexdigest()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    source = data.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
    symbols = extract_symbols(source, path)
    facts = graph.collect(source, path)
    records = [
        {
            "qualname": s.qualname,
            "name": s.name,
            "kind": s.kind,
            "start": s.span.start_line,
            "end": s.span.end_line,
            "parent": s.parent_id.rsplit("::", 1)[1] if s.parent_id else None,
        }
        for s in symbols
    ]
    store.execute(
        "insert or ignore into objects values(?,?,?,?)",
        (digest, zlib.compress(source.encode()), json.dumps(records), facts.model_dump_json()),
    )
    store.execute("insert or ignore into aliases values(?,?)", (blob, digest))
    return digest, source, symbols, facts, False


def _add_file(
    db: sqlite3.Connection,
    path: str,
    digest: str,
    source: str,
    symbols: list[Symbol],
    facts: graph.Facts,
    config: IndexConfig,
) -> None:
    db.execute("insert into files values(?,?)", (path, digest))
    db.executemany(
        "insert or replace into symbols values(?,?,?,?,?,?)",
        ((s.id, path, s.qualname, s.name, s.span.start_line, s.model_dump_json()) for s in symbols),
    )
    db.executemany(
        "insert into calls(path,caller,target,line) values(?,?,?,?)",
        ((path, f"{path}::{caller}", target, line) for caller, target, line in facts.calls),
    )
    db.executemany(
        "insert into bases values(?,?)",
        ((f"{path}::{cls}", base) for cls, bases in facts.bases.items() for base in bases),
    )
    db.executemany(
        "insert into imports values(?,?,?)",
        ((path, alias, target) for alias, target in facts.imports),
    )
    lines = source.split("\n")
    if lines[-1] == "":
        lines.pop()
    path_terms = " ".join(lexical.terms(path))
    for symbol, start, end in lexical.units(symbols, lines, config.module_chunk_lines):
        row = db.execute("insert into units values(?,?,?,?)", (path, symbol.id, start, end))
        tokens = lexical.terms("\n".join(lines[start - 1 : end]))
        db.execute(
            "insert into postings(rowid,path,qualname,body) values(?,?,?,?)",
            (row.lastrowid, path_terms, " ".join(lexical.terms(symbol.qualname)), " ".join(tokens)),
        )


def _texts(db: sqlite3.Connection, store: sqlite3.Connection) -> Iterator[str]:
    previous = ""
    lines = []
    for path, digest, symbol, start, end in db.execute(
        "select u.path,f.hash,u.symbol,u.start,u.end from units u "
        "join files f on f.path=u.path order by u.rowid"
    ):
        if digest != previous:
            (packed,) = store.execute(
                "select source from objects where hash=?", (digest,)
            ).fetchone()
            lines = zlib.decompress(packed).decode().split("\n")
            previous = digest
        yield path + "\n" + symbol.rsplit("::", 1)[1] + "\n" + "\n".join(lines[start - 1 : end])


def _populate(
    root: Path,
    commit: str,
    destination: Path,
    store: sqlite3.Connection,
    db: sqlite3.Connection,
    config: IndexConfig,
    dense: bool,
) -> dict:
    started = time.monotonic()
    stats = Counter()
    tree = _git(root, "ls-tree", "-rlz", "--full-tree", commit)
    db.executescript(SCHEMA)
    with subprocess.Popen(
        ["git", "-C", str(root), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as process:
        assert process.stdin and process.stdout
        try:
            for entry in tree.split(b"\x00"):
                if not entry:
                    continue
                meta, raw_path = entry.split(b"\t", 1)
                mode, kind, blob, size = meta.split()
                path = os.fsdecode(raw_path)
                if kind != b"blob" or mode not in (b"100644", b"100755"):
                    stats["skipped_special"] += 1
                    continue
                if not path.endswith(".py"):
                    stats["skipped_non_python"] += 1
                    continue
                if int(size) > config.max_file_bytes:
                    stats["skipped_size"] += 1
                    continue
                try:
                    digest, source, symbols, facts, reused = _get_file(
                        store, process.stdout, process.stdin, blob.decode(), path
                    )
                except (SyntaxError, LookupError, UnicodeError) as exc:
                    _LOG.warning("Cannot decode %r: %s", path, exc)
                    stats["skipped_encoding"] += 1
                    continue
                _add_file(db, path, digest, source, symbols, facts, config)
                stats["reused_files" if reused else "parsed_files"] += 1
        finally:
            process.stdin.close()
    if process.returncode:
        raise OSError(f"git cat-file exited with status {process.returncode}")
    parse_finished = time.monotonic()
    graph.resolve(db)
    db.executescript(
        "create index targets on calls(callee); create index callers on calls(caller);"
    )
    graph_finished = time.monotonic()
    if dense:
        stats.update(
            embeddings.build_vectors(
                _texts(db, store),
                store,
                destination / "vectors.f16",
                config.model_id,
                config.token_budget,
                config.batch_size,
            )
        )
    store.commit()
    db.commit()
    return {
        "commit": commit,
        "version": INDEX_VERSION,
        "config": config.model_dump(),
        "dense": dense,
        "counts": dict(stats),
        "seconds": {
            "parse_and_lexical": parse_finished - started,
            "graph": graph_finished - parse_finished,
            "dense": time.monotonic() - graph_finished,
        },
    }


def build_cache(
    repo_root: Path,
    commit: str,
    cache_dir: Path,
    dense: bool,
    config: IndexConfig,
) -> tuple[Path, Path]:
    commit = (
        _git(repo_root, "rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}")
        .decode()
        .strip()
    )
    common = (
        _git(repo_root, "rev-parse", "--path-format=absolute", "--git-common-dir").decode().strip()
    )
    remote = subprocess.run(
        ["git", "-C", str(repo_root), "config", "--get", "remote.origin.url"],
        capture_output=True,
        check=False,
    )
    repo_id = remote.stdout.strip() or os.fsencode(common)
    repo_key = hashlib.sha256(repo_id).hexdigest()
    store_dir = cache_dir / f"v{INDEX_VERSION}" / repo_key
    store_dir.mkdir(parents=True, exist_ok=True)
    cfg_key = hashlib.sha256((config.model_dump_json() + str(dense)).encode()).hexdigest()
    destination = store_dir / f"{commit}-{cfg_key}-graph{GRAPH_VERSION}"
    store_path = store_dir / "content.sqlite"
    if (destination / "manifest.json").is_file():
        return destination, store_path
    with (store_dir / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "manifest.json").is_file():
            return destination, store_path
        with tempfile.TemporaryDirectory(dir=store_dir) as tmp:
            working = Path(tmp) / "index"
            working.mkdir()
            with (
                contextlib.closing(sqlite3.connect(store_path)) as store,
                contextlib.closing(sqlite3.connect(working / "index.sqlite")) as db,
            ):
                store.execute("pragma journal_mode=WAL")
                store.executescript("""
                    create table if not exists objects(hash text primary key,source blob,
                                                        symbols text,facts text);
                    create table if not exists aliases(blob text primary key,hash text);
                """)
                manifest = _populate(repo_root, commit, working, store, db, config, dense)
            (working / "manifest.json").write_text(json.dumps(manifest, indent=2))
            working.rename(destination)
    return destination, store_path
