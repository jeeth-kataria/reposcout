import json
import os
import sqlite3
import zlib
from pathlib import Path

from pydantic import TypeAdapter

from reposcout.contracts.index import ReadResult
from reposcout.contracts.models import RepoPath, SearchHit, Span, Symbol

from . import lexical
from .cache import IndexConfig, build_cache
from .dense import DenseSearch

_FILE_PATH = TypeAdapter(RepoPath)


class RepositoryIndex:
    def __init__(self, directory: Path, store: Path, config: IndexConfig) -> None:
        self.directory = directory
        self.config = config
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self._db = sqlite3.connect(
            (directory / "index.sqlite").resolve().as_uri() + "?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self._store = sqlite3.connect(
            store.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False
        )
        self._dense: DenseSearch | None = None

    def close(self) -> None:
        self._db.close()
        self._store.close()

    def symbols(self, path: str | None = None) -> list[Symbol]:
        if path is None:
            rows = self._db.execute("select data from symbols order by path,start,qualname")
        else:
            rows = self._db.execute(
                "select data from symbols where path=? order by start,qualname",
                (_FILE_PATH.validate_python(path),),
            )
        return [Symbol.model_validate_json(row[0]) for row in rows]

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        row = self._db.execute("select data from symbols where id=?", (symbol_id,)).fetchone()
        return Symbol.model_validate_json(row[0]) if row else None

    def _neighbors(self, symbol_id: str, limit: int, incoming: bool) -> list[Symbol]:
        src, dst = ("callee", "caller") if incoming else ("caller", "callee")
        rows = self._db.execute(
            f"select data from symbols where id in (select {dst} from calls where {src}=?) "
            "order by path,start,qualname limit ?",
            (symbol_id, max(0, limit)),
        )
        return [Symbol.model_validate_json(row[0]) for row in rows]

    def callers(self, symbol_id: str, limit: int = 20) -> list[Symbol]:
        return self._neighbors(symbol_id, limit, True)

    def callees(self, symbol_id: str, limit: int = 20) -> list[Symbol]:
        return self._neighbors(symbol_id, limit, False)

    def read(self, span: Span, max_lines: int = 200) -> ReadResult:
        if max_lines <= 0:
            raise ValueError("max_lines must be positive")
        row = self._db.execute("select hash from files where path=?", (span.path,)).fetchone()
        if row is None:
            raise FileNotFoundError(span.path)
        (packed,) = self._store.execute("select source from objects where hash=?", row).fetchone()
        lines = zlib.decompress(packed).decode().split("\n")
        if lines[-1] == "":
            lines.pop()
        end = min(span.end_line, len(lines), span.start_line + max_lines - 1)
        if span.start_line > end:
            raise ValueError(f"{span.path} has no line {span.start_line}")
        return ReadResult(
            text="\n".join(lines[span.start_line - 1 : end]),
            span=Span(path=span.path, start_line=span.start_line, end_line=end),
            truncated=end < span.end_line,
        )

    def list_dir(self, path: str) -> list[str]:
        prefix = "" if path in {"", "."} else _FILE_PATH.validate_python(path) + "/"
        rows = self._db.execute(
            "select path from files where substr(path,1,?)=?", (len(prefix), prefix)
        )
        entries = set()
        for (name,) in rows:
            first, slash, _ = name[len(prefix) :].partition("/")
            entries.add(first + slash)
        if not entries and prefix:
            raise FileNotFoundError(path)
        return sorted(entries)

    def _hit(self, rowid: int, score: float, source: str) -> SearchHit:
        path, symbol, start, end = self._db.execute(
            "select path,symbol,start,end from units where rowid=?", (rowid,)
        ).fetchone()
        span = Span(path=path, start_line=start, end_line=end)
        shown = self.read(span, self.config.snippet_lines)
        snippet = shown.text + (" […]" if shown.truncated else "")
        return SearchHit(symbol_id=symbol, span=span, score=score, source=source, snippet=snippet)

    def bm25(self, query: str, k: int = 20) -> list[SearchHit]:
        return [self._hit(i, score, "bm25") for i, score in lexical.search(self._db, query, k)]

    def dense(self, query: str, k: int = 20) -> list[SearchHit]:
        if not self.manifest["dense"]:
            raise ValueError("dense retrieval disabled when building this index")
        dim = self.manifest["counts"].get("dimensions", 0)
        if not dim or not query.strip() or k <= 0:
            return []
        if self._dense is None:
            self._dense = DenseSearch(
                self.directory / "vectors.f16",
                dim,
                self.config.model_id,
                self.config.token_budget,
                self.config.batch_size,
            )
        scores = self._dense.scores(query)
        # FAISS returns all equal-score rows in index-specific order. Resolve the cutoff ties.
        cutoff = scores[min(k, len(scores)) - 1][1]
        above = [(i, s) for i, s in scores if s >= cutoff]

        def rank_key(item: tuple[int, float]) -> tuple[float, str, int, str]:
            rowid, score = item
            path, start, symbol = self._db.execute(
                "select path,start,symbol from units where rowid=?", (rowid,)
            ).fetchone()
            return -score, path, start, symbol or ""

        ranked = sorted(above, key=rank_key)[:k]
        return [self._hit(rowid, score, "dense") for rowid, score in ranked]


def build_index(
    repo_root: Path, commit: str, cache_dir: Path, *, dense: bool = True
) -> RepositoryIndex:
    overrides = {}
    if model := os.getenv("REPOSCOUT_EMBEDDING_MODEL"):
        overrides["model_id"] = model
    config = IndexConfig(**overrides)
    return configured_index(repo_root, commit, cache_dir, dense=dense, config=config)


def configured_index(
    repo_root: Path, commit: str, cache_dir: Path, *, dense: bool, config: IndexConfig
) -> RepositoryIndex:
    directory, store = build_cache(repo_root, commit, cache_dir, dense, config)
    return RepositoryIndex(directory, store, config)
