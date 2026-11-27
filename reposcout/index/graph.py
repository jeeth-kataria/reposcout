"""Path-independent Python call facts and deterministic commit-wide resolution."""

import ast
import logging
import sqlite3
from collections import defaultdict
from pathlib import PurePosixPath

from pydantic import BaseModel, Field


class Facts(BaseModel):
    imports: list[tuple[str, str]] = Field(default_factory=list)
    calls: list[tuple[str, str, int]] = Field(default_factory=list)
    bases: dict[str, list[str]] = Field(default_factory=dict)


def _dotted(expr: ast.expr) -> str:
    suffix = []
    while isinstance(expr, ast.Attribute | ast.Call):
        if isinstance(expr, ast.Attribute):
            suffix.append("." + expr.attr)
            expr = expr.value
        else:
            suffix.append("()")
            expr = expr.func
    return (expr.id if isinstance(expr, ast.Name) else "?") + "".join(reversed(suffix))


def collect(source: str, path: str) -> Facts:
    facts = Facts()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        logging.getLogger(__name__).info("Skipping AST call facts in %s: %s", path, exc)
        return facts
    pending: list[tuple[ast.AST, str]] = [(tree, "<module>")]
    while pending:
        node, owner = pending.pop()
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            scope = node.name if owner == "<module>" else f"{owner}.{node.name}"
            if isinstance(node, ast.ClassDef):
                facts.bases[scope] = [_dotted(base) for base in node.bases]
            pending.extend((child, scope) for child in reversed(node.body))
            continue
        if isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            prefix = module + "." if node.module else module
            facts.imports.extend((a.asname or a.name, prefix + a.name) for a in node.names)
        elif isinstance(node, ast.Import):
            facts.imports.extend(
                (a.asname or a.name.split(".")[0], a.name if a.asname else a.name.split(".")[0])
                for a in node.names
            )
        elif isinstance(node, ast.Call):
            facts.calls.append((owner, _dotted(node.func), node.lineno))
        pending.extend((child, owner) for child in ast.iter_child_nodes(node))
    return facts


def _module(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if len(parts) > 1 and parts[0] == "src":
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve(db: sqlite3.Connection) -> None:
    by_name: dict[str, list[str]] = defaultdict(list)
    qualified: dict[str, list[str]] = defaultdict(list)
    local: dict[tuple[str, str], list[str]] = defaultdict(list)
    ids = set()
    for sid, path, qualname, name in db.execute("select id,path,qualname,name from symbols"):
        ids.add(sid)
        by_name[name].append(sid)
        local[path, name].append(sid)
        qualified[f"{_module(path)}.{qualname}"].append(sid)

    imports: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path, alias, target in db.execute("select path, alias, target from imports"):
        if target.startswith("."):
            depth = len(target) - len(target.lstrip("."))
            module = _module(path)
            package = (
                module if PurePosixPath(path).name == "__init__.py" else module.rpartition(".")[0]
            )
            parents = package.split(".") if package else []
            if depth > len(parents):
                continue
            prefix = ".".join(parents[: len(parents) - depth + 1])
            target = ".".join(filter(None, [prefix, target.lstrip(".")]))
        imports[path, alias].add(target)
    bases: dict[str, list[str]] = defaultdict(list)
    for cls, base in db.execute("select class_id,base from bases"):
        bases[cls].append(base)

    def lookup(path: str, name: str) -> str | None:
        first, _, rest = name.partition(".")
        matches = local.get((path, name), [])
        if len(matches) == 1:
            return matches[0]
        targets = imports.get((path, first), set())
        if targets:
            imported = {
                sid
                for target in targets
                for sid in qualified.get(".".join(filter(None, [target, rest])), [])
            }
            return next(iter(imported)) if len(imported) == 1 else None
        matches = by_name.get(name, [])
        return matches[0] if len(matches) == 1 else None

    def callee(path: str, caller: str, target: str) -> str | None:
        if target.startswith(("self.", "cls.")):
            method = target.split(".", 1)[1]
            owner = caller.rpartition(".")[0]
            if not owner:
                return None
            pending = [owner]
            seen: set[str] = set()
            while pending:
                cls = pending.pop()
                if cls in seen:
                    continue
                seen.add(cls)
                sid = f"{cls}.{method}"
                if sid in ids:
                    return sid
                base_path = cls.rsplit("::", 1)[0]
                for base in reversed(bases.get(cls, [])):
                    match = lookup(base_path, base)
                    if match:
                        pending.append(match)
            return None
        if "()." in target:
            constructor, method = target.split("().", 1)
            cls = lookup(path, constructor)
            sid = f"{cls}.{method}"
            return sid if sid in ids else None
        return lookup(path, target)

    updates = (
        (callee(path, caller, target), rowid)
        for rowid, path, caller, target in db.execute(
            "select rowid,path,caller,target from calls order by rowid"
        )
    )
    db.executemany("update calls set callee=? where rowid=?", updates)
