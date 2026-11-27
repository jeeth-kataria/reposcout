import tokenize
from pathlib import Path

from pydantic import BaseModel, TypeAdapter
from unidiff import PatchSet
from unidiff.patch import Hunk

from reposcout.contracts.models import RepoPath
from reposcout.index.parser import extract_symbols

_PATH = TypeAdapter(RepoPath)


class Gold(BaseModel):
    files: list[str]
    functions: list[str]


def _changed_lines(hunk: Hunk) -> set[int]:
    removed = {line.source_line_no for line in hunk if line.is_removed}
    if removed:
        return removed
    previous = hunk.source_start if hunk.source_length == 0 else hunk.source_start - 1
    anchors = set()
    for line in hunk:
        if line.is_added and previous > 0:
            anchors.add(previous)
        elif line.is_context:
            previous = line.source_line_no
    return anchors


def extract_gold(patch: str, repo_root: Path) -> Gold:
    """Map changes in the implementation patch to pre-image paths and innermost functions."""
    files: set[str] = set()
    functions: set[str] = set()
    root = repo_root.resolve()
    for change in PatchSet(patch):
        if change.source_file == "/dev/null":
            continue
        path = _PATH.validate_python(change.source_file.removeprefix("a/"))
        files.add(path)
        lines = set().union(*(_changed_lines(hunk) for hunk in change))
        if not path.endswith(".py") or not lines:
            continue
        source = (root / path).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"patch source crosses repository boundary: {path}")
        with tokenize.open(source) as stream:
            symbols = [
                s for s in extract_symbols(stream.read(), path) if s.kind in {"function", "method"}
            ]
        # An inner definition always has a shorter span (ties prefer deeper nesting).
        symbols.sort(key=lambda s: (s.span.end_line - s.span.start_line, -s.qualname.count(".")))
        for line in lines:
            match = next((s for s in symbols if s.span.start_line <= line <= s.span.end_line), None)
            if match:
                functions.add(match.id)
    return Gold(files=sorted(files), functions=sorted(functions))
