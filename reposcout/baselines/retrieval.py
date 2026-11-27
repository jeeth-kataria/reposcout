import re
from collections.abc import Iterable
from functools import partial
from pathlib import PurePosixPath
from time import perf_counter
from typing import Literal

from reposcout.contracts.index import RepoIndex
from reposcout.contracts.localizer import LocalizerConfig
from reposcout.contracts.models import Candidate, Instance, Localization, RunStats, SearchHit
from reposcout.contracts.registry import register

type Method = Literal["bm25", "dense", "hybrid_rrf", "hybrid_stacktrace"]
METHODS: tuple[Method, ...] = ("bm25", "dense", "hybrid_rrf", "hybrid_stacktrace")
_FRAME = re.compile(r'^\s*File "([^"\n]+\.py)", line (\d+)(?:, in ([^\n]+))?\s*$', re.MULTILINE)


def traceback_frames(text: str) -> list[tuple[str, int, str]]:
    return [(p.replace("\\", "/"), int(n), (f or "").strip()) for p, n, f in _FRAME.findall(text)]


type UnitKey = tuple[str, str | None, int, int]


def _identity(hit: SearchHit) -> UnitKey:
    return hit.span.path, hit.symbol_id, hit.span.start_line, hit.span.end_line


def _order(hit: SearchHit) -> tuple[float, str, int, str]:
    return -hit.score, hit.span.path, hit.span.start_line, hit.symbol_id or ""


def reciprocal_rank_fusion(
    rankings: Iterable[list[SearchHit]], *, rank_constant: int = 60
) -> list[SearchHit]:
    scores: dict[UnitKey, float] = {}
    hits = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = _identity(hit)
            if key in seen:
                continue
            seen.add(key)
            hits.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + 1 / (rank_constant + rank)
    return sorted(
        [
            hit.model_copy(update={"score": scores[key], "source": "rrf"})
            for key, hit in hits.items()
        ],
        key=_order,
    )


def _boost_frames(hits: list[SearchHit], issue: str, index: RepoIndex) -> list[SearchHit]:
    frames = traceback_frames(issue)
    if not frames:
        return hits
    path_bonus, symbol_bonus = 0.025, 0.05
    symbols = index.symbols()
    paths = {s.span.path for s in symbols}
    matches: dict[str, list[tuple[int, str]]] = {}
    for frame, line, name in frames:
        suffixes = [path for path in paths if frame == path or frame.endswith("/" + path)]
        if suffixes:
            path = max(suffixes, key=len)
        else:
            # A shortened frame can omit leading repository directories. A bare
            # filename is useful only when it resolves to one indexed file.
            candidates = [path for path in paths if path.endswith("/" + frame)]
            if len(candidates) != 1:
                continue
            path = candidates[0]
        matches.setdefault(path, []).append((line, name))
    by_id: dict[UnitKey, SearchHit] = {}
    for hit in hits:
        if hit.span.path in matches:
            hit = hit.model_copy(
                update={"score": hit.score + path_bonus, "source": "rrf+traceback"}
            )
        by_id[_identity(hit)] = hit
    for symbol in symbols:
        span = symbol.span
        for line, name in matches.get(span.path, []):
            if not span.start_line <= line <= span.end_line:
                continue
            if symbol.kind == "module":
                citation = span.model_copy(update={"start_line": line, "end_line": line})
                symbol_id, bonus = None, 0.0
            elif symbol.name == name and symbol.kind in {"function", "method"}:
                citation, symbol_id, bonus = span, symbol.id, symbol_bonus
            else:
                continue
            hit = SearchHit(
                symbol_id=symbol_id,
                span=citation,
                source="traceback",
                score=path_bonus + bonus,
                snippet=f"{PurePosixPath(span.path).name}, line {line}, in {name}",
            )
            key = _identity(hit)
            if key in by_id:
                hit.score = by_id[key].score + bonus
            by_id[key] = hit
            break
    return sorted(by_id.values(), key=_order)


class RetrievalLocalizer:
    def __init__(self, method: Method, retrieval_depth: int = 50, rrf_k: int = 60) -> None:
        self.method, self.retrieval_depth, self.rrf_k = method, retrieval_depth, rrf_k

    def localize(
        self, instance: Instance, index: RepoIndex, config: LocalizerConfig
    ) -> Localization:
        start = perf_counter()
        issue = instance.problem_statement
        if self.method == "dense" and not config.use_dense:
            raise ValueError("the dense baseline requires a dense index")
        lexical = index.bm25(issue, self.retrieval_depth) if self.method != "dense" else []
        dense = (
            index.dense(issue, self.retrieval_depth)
            if self.method != "bm25" and config.use_dense
            else []
        )
        if self.method.startswith("hybrid"):
            hits = reciprocal_rank_fusion([lexical, dense], rank_constant=self.rrf_k)
        else:
            hits = sorted(lexical + dense, key=_order)
        if self.method == "hybrid_stacktrace":
            hits = _boost_frames(hits, issue, index)
        files: list[str] = []
        functions: list[str] = []
        candidates: list[Candidate] = []
        for hit in hits:
            file_added = hit.span.path not in files and len(files) < config.top_k_files
            if file_added:
                files.append(hit.span.path)
            symbol = index.get_symbol(hit.symbol_id) if hit.symbol_id else None
            func_added = (
                symbol is not None
                and symbol.kind in {"function", "method"}
                and symbol.id not in functions
                and len(functions) < config.top_k_functions
            )
            if func_added and symbol is not None:
                functions.append(symbol.id)
            if file_added or func_added:
                candidates.append(
                    Candidate(
                        path=hit.span.path,
                        symbol_id=hit.symbol_id,
                        score=hit.score,
                        rationale=f"Ranked by {hit.source} at depth "
                        f"{self.retrieval_depth}, L{hit.span.start_line}-{hit.span.end_line}.",
                        citations=[hit.span],
                    )
                )
        return Localization(
            instance_id=instance.instance_id,
            files=files,
            functions=functions,
            candidates=candidates,
            stats=RunStats(status="complete", latency_s=perf_counter() - start),
        )


for method in METHODS:
    register(method)(partial(RetrievalLocalizer, method))
