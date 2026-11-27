import re
import sqlite3
from collections.abc import Iterator

from reposcout.contracts.models import Symbol

_IDENT = re.compile(r"\w+(?:\.\w+)*", re.UNICODE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def terms(text: str) -> Iterator[str]:
    for match in _IDENT.finditer(text):
        full = match.group()
        variants = [full]
        for segment in full.split("."):
            variants.append(segment)
            for part in segment.split("_"):
                variants.extend(_CAMEL.split(part))
        yield from dict.fromkeys(value.lower() for value in variants if value)


def units(
    symbols: list[Symbol], lines: list[str], chunk_lines: int
) -> Iterator[tuple[Symbol, int, int]]:
    if not symbols:
        return
    module = symbols[0]
    cursor = 1

    def gaps(start: int, stop: int) -> Iterator[tuple[Symbol, int, int]]:
        for first in range(start, stop, chunk_lines):
            end = min(stop - 1, first + chunk_lines - 1)
            if any(line.strip() for line in lines[first - 1 : end]):
                yield module, first, end

    for symbol in symbols[1:]:
        if symbol.parent_id == module.id:
            yield from gaps(cursor, symbol.span.start_line)
            cursor = max(cursor, symbol.span.end_line + 1)
        yield symbol, symbol.span.start_line, symbol.span.end_line
    yield from gaps(cursor, len(lines) + 1)


def search(db: sqlite3.Connection, query: str, k: int) -> list[tuple[int, float]]:
    expression = " OR ".join(f'"{term}"' for term in sorted(set(terms(query))))
    if not expression or k <= 0:
        return []
    rows = db.execute(
        "select u.rowid,-f.rank from postings f join units u on u.rowid=f.rowid "
        "where postings match ? order by f.rank,u.path,u.start,u.rowid limit ?",
        (expression, k),
    )
    return list(rows)
