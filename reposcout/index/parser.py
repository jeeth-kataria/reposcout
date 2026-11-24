"""Extract Python definitions using the version-pinned tree-sitter grammar."""

import logging
from pathlib import PurePosixPath

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from reposcout.contracts.models import Span, Symbol

_LANGUAGE = Language(tree_sitter_python.language())
_LOG = logging.getLogger(__name__)


def _is_test(path: str) -> bool:
    parts = PurePosixPath(path)
    return (
        any(part in {"test", "tests"} for part in parts.parts[:-1])
        or parts.name.startswith("test_")
        or parts.name.endswith("_test.py")
    )


def extract_symbols(source: str, path: str) -> list[Symbol]:
    """Return declarations in source order, including intact nodes in broken files."""
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    if not source:
        return []
    root = Parser(_LANGUAGE).parse(source.encode()).root_node
    if root.has_error:
        _LOG.warning("Partial Python syntax tree in %s", path)
    test = _is_test(path)
    module = Symbol(
        id=f"{path}::<module>",
        kind="module",
        name="<module>",
        qualname="<module>",
        span=Span(
            path=path, start_line=1, end_line=source.count("\n") + int(not source.endswith("\n"))
        ),
        is_test=test,
    )
    symbols = [module]
    pending: list[tuple[Node, Symbol]] = [(root, module)]
    while pending:
        node, parent = pending.pop()
        first_line = node.start_point.row + 1
        if node.type == "decorated_definition":
            definition = node.child_by_field_name("definition")
            if definition is not None:
                node = definition
        if node.type in {"class_definition", "function_definition"}:
            name_node = node.child_by_field_name("name")
            if name_node is not None and name_node.text:
                name = name_node.text.decode()
                qualname = name if parent.kind == "module" else f"{parent.qualname}.{name}"
                kind = "function"
                if node.type == "class_definition":
                    kind = "class"
                elif parent.kind == "class":
                    kind = "method"
                parent = Symbol(
                    id=f"{path}::{qualname}",
                    kind=kind,
                    name=name,
                    qualname=qualname,
                    parent_id=parent.id,
                    is_test=test,
                    span=Span(
                        path=path,
                        start_line=first_line,
                        end_line=max(
                            first_line, node.end_point.row + int(node.end_point.column > 0)
                        ),
                    ),
                )
                symbols.append(parent)
        pending.extend((child, parent) for child in reversed(node.named_children))
    return symbols
