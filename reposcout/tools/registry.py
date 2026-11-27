"""Read-only tools for commit-pinned Python indexes."""

import json
from contextvars import ContextVar
from time import perf_counter
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictInt, ValidationError, field_validator

from reposcout.contracts.index import RepoIndex
from reposcout.contracts.localizer import LocalizerConfig
from reposcout.contracts.models import ContractModel, RepoPath, SearchHit, Span
from reposcout.contracts.tools import ToolName, ToolResult, ToolSpec

MAX_LINE = 1_000_000
_TRUNCATED = "[Output truncated.]"
_NOTE_RESERVE = len(_TRUNCATED) + 2
type Line = Annotated[StrictInt, Field(ge=1, le=MAX_LINE)]


class ToolConfig(LocalizerConfig):
    max_read_lines: int = Field(default=200, ge=1, le=200)
    max_tool_chars: int = Field(default=12_000, ge=100)
    max_doc_lines: int = Field(default=12, ge=1)


class SearchArgs(ContractModel):
    query: str = Field(min_length=1, max_length=12_000)
    k: StrictInt = Field(ge=1, le=20)


class SymbolArgs(ContractModel):
    symbol_id: str = Field(min_length=1)


class RangeArgs(ContractModel):
    path: RepoPath
    start_line: Line
    end_line: Line


class DirectoryArgs(ContractModel):
    path: Literal[".", ""] | RepoPath

    @field_validator("path", mode="before")
    @classmethod
    def normalize_root(cls, value: object) -> object:
        if value == "":
            return "."
        return value.removesuffix("/") if isinstance(value, str) and value != "/" else value


class ToolEvent(BaseModel):
    name: str
    arguments: str
    latency_s: float
    chars: int
    truncated: bool
    error: str | None


ToolLog: ContextVar[list[ToolEvent] | None] = ContextVar("reposcout_tool_log", default=None)

_SPECS: dict[ToolName, tuple[str, type[ContractModel]]] = {
    "search_code": ("Find Python symbols and lines by lexical relevance.", SearchArgs),
    "semantic_search": ("Find Python symbols and lines by semantic similarity.", SearchArgs),
    "get_symbol": ("Show the definition and nearby symbols.", SymbolArgs),
    "callers": ("List symbols that call this symbol.", SymbolArgs),
    "callees": ("List symbols called by this symbol.", SymbolArgs),
    "read_range": ("Read inclusive source lines at a repository-relative path.", RangeArgs),
    "list_dir": ("List indexed entries in a directory. Use '.' for the root.", DirectoryArgs),
}


def _enabled(name: str, config: LocalizerConfig) -> bool:
    return (
        name in _SPECS
        and (name != "semantic_search" or config.use_dense)
        and (name not in {"callers", "callees"} or config.use_graph_tools)
    )


def get_tool_specs(config: LocalizerConfig) -> list[ToolSpec]:
    return [
        ToolSpec(name=name, description=description, json_schema=args.model_json_schema())
        for name, (description, args) in _SPECS.items()
        if _enabled(name, config)
    ]


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("Duplicate argument name")
    return values


def _read(index: RepoIndex, span: Span, limit: int, cap: int) -> ToolResult:
    result = index.read(span, limit)
    lines = []
    used = 0
    for number, line in enumerate(result.text.split("\n"), result.span.start_line):
        formatted = f"{number:4d} | {line}"
        if used + len(formatted) + 1 > cap - _NOTE_RESERVE:
            break
        lines.append(formatted)
        used += len(formatted) + 1
    truncated = result.truncated or len(lines) < result.span.end_line - result.span.start_line + 1
    citations = []
    if lines:
        citations.append(
            Span(
                path=span.path,
                start_line=result.span.start_line,
                end_line=result.span.start_line + len(lines) - 1,
            )
        )
    if truncated:
        lines.append(_TRUNCATED)
    return ToolResult(text="\n".join(lines), citations=citations, truncated=truncated)


def _search(index: RepoIndex, hits: list[SearchHit], cap: int) -> ToolResult:
    blocks = []
    citations = []
    used = 0
    truncated = False
    for rank, hit in enumerate(hits, 1):
        span = hit.span
        label = hit.symbol_id or span.path
        title = f"{rank}. {label}  L{span.start_line}-{span.end_line}  score={hit.score:.3f}"
        evidence = _read(index, span, 3, cap - used - len(title) - 2)
        if not evidence.citations:
            truncated = True
            break
        source = "\n".join(line for line in evidence.text.split("\n") if line != _TRUNCATED)
        block = title + "\n" + source
        blocks.append(block)
        citations.extend(evidence.citations)
        used += len(block) + 2
        truncated |= evidence.truncated
    if truncated:
        blocks.append(_TRUNCATED)
    return ToolResult(
        text="\n\n".join(blocks) or "No matches.", citations=citations, truncated=truncated
    )


def _execute(
    name: str,
    args: ContractModel,
    index: RepoIndex,
    config: LocalizerConfig,
) -> ToolResult:
    cap = getattr(config, "max_tool_chars", 12_000)
    if isinstance(args, SearchArgs):
        search = index.bm25 if name == "search_code" else index.dense
        return _search(index, search(args.query, args.k), cap)
    if isinstance(args, RangeArgs):
        return _read(index, Span(**args.model_dump()), getattr(config, "max_read_lines", 200), cap)
    if isinstance(args, DirectoryArgs):
        rows = index.list_dir(args.path)
        return _bounded_rows(rows, cap)
    if not isinstance(args, SymbolArgs):
        raise ValueError("Unexpected argument type")
    symbol = index.get_symbol(args.symbol_id)
    if symbol is None:
        raise ValueError(f"Unknown symbol {args.symbol_id}")
    if name in {"callers", "callees"}:
        related = (index.callers if name == "callers" else index.callees)(args.symbol_id, 21)
        rows = [f"{s.id} L{s.span.start_line}-{s.span.end_line}" for s in related[:20]]
        return _bounded_rows(rows, cap, [s.span for s in related[:20]], len(related) > 20)
    # The leading source lines include the signature and the start of any docstring.
    source = _read(index, symbol.span, getattr(config, "max_doc_lines", 12), (cap - 1) // 2)
    parent = f"Parent: {symbol.parent_id}" if symbol.parent_id else "Parent: none"
    children = [s.id for s in index.symbols(symbol.span.path) if s.parent_id == symbol.id]
    hierarchy = _bounded_rows([parent, *[f"Child: {child}" for child in children]], (cap - 1) // 2)
    return ToolResult(
        text=source.text + "\n" + hierarchy.text,
        citations=source.citations,
        truncated=source.truncated or hierarchy.truncated,
    )


def _bounded_rows(
    rows: list[str], cap: int, spans: list[Span] | None = None, more: bool = False
) -> ToolResult:
    used = 0
    count = 0
    for row in rows:
        if used + len(row) + 1 > cap - _NOTE_RESERVE:
            break
        used += len(row) + 1
        count += 1
    truncated = more or count < len(rows)
    text = "\n".join(rows[:count])
    if truncated:
        text = "\n".join(filter(None, (text, _TRUNCATED)))
    return ToolResult(
        text=text or "No entries.", citations=spans[:count] if spans else [], truncated=truncated
    )


def dispatch(name: str, raw_args: str, index: RepoIndex, config: LocalizerConfig) -> ToolResult:
    start = perf_counter()
    try:
        if not _enabled(name, config):
            raise ValueError(f"Unknown or disabled tool {name!r}")
        if len(raw_args) > 24_000:
            raise ValueError("Tool arguments exceed 24000 characters")
        parser = _SPECS[name][1]
        args = parser.model_validate(json.loads(raw_args, object_pairs_hook=_unique_pairs))
        result = _execute(name, args, index, config)
    except ValidationError as exc:
        message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        result = ToolResult(text=message, citations=[], truncated=False, error=message)
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        result = ToolResult(text=message, citations=[], truncated=False, error=message)
    cap = getattr(config, "max_tool_chars", 12_000)
    if result.error and len(result.text) > cap:
        message = result.text[: cap - _NOTE_RESERVE] + "\n" + _TRUNCATED
        result = ToolResult(text=message, citations=[], truncated=True, error=message)
    log = ToolLog.get()
    if log is not None:
        log.append(
            ToolEvent(
                name=name,
                arguments=raw_args,
                latency_s=perf_counter() - start,
                chars=len(result.text),
                truncated=result.truncated,
                error=result.error,
            )
        )
    return result
