from pathlib import Path

import pytest

from reposcout.index.parser import extract_symbols

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minirepo"


def test_async_decorated_method() -> None:
    path = "postbag/dispatch.py"
    module, desk, method = extract_symbols((FIXTURE / path).read_text(), path)
    assert [
        (s.qualname, s.kind, s.span.start_line, s.span.end_line) for s in (module, desk, method)
    ] == [
        ("<module>", "module", 1, 8),
        ("DispatchDesk", "class", 5, 8),
        ("DispatchDesk.manifest", "method", 6, 8),
    ]
    assert method.parent_id == desk.id and desk.parent_id == module.id


def test_nested_and_decorated_functions() -> None:
    path = "postbag/zones.py"
    _, outer, inner = extract_symbols((FIXTURE / path).read_text(), path)
    assert inner.qualname == "destination_zone.prefix"
    assert inner.parent_id == outer.id
    assert (inner.span.start_line, inner.span.end_line) == (2, 3)
    path = "postbag/returns.py"
    assert extract_symbols((FIXTURE / path).read_text(), path)[-1].span.start_line == 6


def test_partial_parse_and_empty(caplog: pytest.LogCaptureFixture) -> None:
    path = "postbag/unfinished.py"
    names = [s.name for s in extract_symbols((FIXTURE / path).read_text(), path)]
    assert "customs_reference" in names
    assert path in caplog.text
    assert extract_symbols("", "empty.py") == []


@pytest.mark.parametrize(
    "path", ["test/rates.py", "pkg/tests/rates.py", "pkg/test_a.py", "a_test.py"]
)
def test_test_paths(path: str) -> None:
    assert all(s.is_test for s in extract_symbols("def rate(): pass", path))
