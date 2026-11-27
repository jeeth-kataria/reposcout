import sqlite3

from reposcout.index.cache import SCHEMA
from reposcout.index.graph import collect, resolve


def test_imports_resolve_from_src_layout() -> None:
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    callee = "src/pkg/worker.py::process"
    db.execute(
        "insert into symbols(id,path,qualname,name) values(?,?,?,?)",
        (callee, "src/pkg/worker.py", "process", "process"),
    )
    for path, target in (
        ("src/pkg/main.py", "pkg.worker.process"),
        ("src/pkg/__init__.py", ".worker.process"),
    ):
        db.execute("insert into imports values(?,?,?)", (path, "alias", target))
        db.execute(
            "insert into calls(path,caller,target,line) values(?,?,?,1)",
            (path, f"{path}::<module>", "alias"),
        )
    resolve(db)
    assert db.execute("select callee from calls").fetchall() == [(callee,), (callee,)]


def test_call_facts_handle_long_attribute_chain() -> None:
    target = "obj" + ".next" * 1000 + ".send"
    facts = collect(f"def relay(obj):\n    {target}()\n", "relay.py")
    assert facts.calls == [("relay", target, 2)]
