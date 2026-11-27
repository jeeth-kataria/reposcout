import json
from pathlib import Path

import pytest
from conftest import GitMinirepo

from reposcout.contracts.localizer import LocalizerConfig
from reposcout.index import build_index
from reposcout.index.repository import RepositoryIndex
from reposcout.tools.registry import ToolConfig, ToolLog, dispatch, get_tool_specs


@pytest.fixture
def index(minirepo_git: GitMinirepo, tmp_path: Path) -> RepositoryIndex:
    return build_index(minirepo_git.root, minirepo_git.base_commit, tmp_path, dense=False)


@pytest.mark.parametrize(
    ("name", "args", "text"),
    [
        ("search_code", {"query": "rate_band", "k": 3}, "RateCard.rate_band"),
        ("list_dir", {"path": "."}, "postbag/"),
        ("get_symbol", {"symbol_id": "postbag/rates.py::RateCard"}, "Child:"),
        ("callers", {"symbol_id": "postbag/rates.py::RateCard.rate_band"}, "RateCard.quote"),
        ("callees", {"symbol_id": "postbag/rates.py::RateCard.quote"}, "RateCard.rate_band"),
    ],
)
def test_dispatch_reads_fixture(
    index: RepositoryIndex, name: str, args: dict[str, str | int], text: str
) -> None:
    result = dispatch(name, json.dumps(args), index, LocalizerConfig(use_dense=False))
    assert result.error is None
    assert text in result.text


def test_read_truncation_matches_citation_and_log(index: RepositoryIndex) -> None:
    token = ToolLog.set([])
    try:
        result = dispatch(
            "read_range",
            '{"path":"postbag/rates.py","start_line":1,"end_line":8}',
            index,
            ToolConfig(max_read_lines=2),
        )
        assert result.error is None and result.truncated
        assert result.citations[0].end_line == 2
        assert "   1 | from .parcel import Parcel" in result.text
        assert "truncated" in result.text
        events = ToolLog.get()
        assert events and events[0].chars == len(result.text)
    finally:
        ToolLog.reset(token)


@pytest.mark.parametrize(
    "args",
    [
        '{"path":"../escape","start_line":1,"end_line":2}',
        '{"path":"/escape","start_line":1,"end_line":2}',
        '{"path":"postbag/rates.py","start_line":2,"end_line":1}',
        '{"path":"postbag/rates.py","start_line":1,"end_line":1000001}',
        '{"path":"postbag/rates.py","start_line":true,"end_line":2}',
        '{"path":"postbag/rates.py","start_line":1,"end_line":2,"k":2}',
        '{"path":"postbag/rates.py","path":"x","start_line":1,"end_line":2}',
        "{",
    ],
)
def test_invalid_arguments_are_tool_errors(index: RepositoryIndex, args: str) -> None:
    assert dispatch("read_range", args, index, LocalizerConfig()).error


def test_schema_filter_and_semantic_dispatch(
    index: RepositoryIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = LocalizerConfig(use_dense=False, use_graph_tools=False)
    specs = get_tool_specs(config)
    assert {spec.name for spec in specs} == {"search_code", "get_symbol", "read_range", "list_dir"}
    assert all(spec.json_schema["additionalProperties"] is False for spec in specs)
    args = '{"query":"rate_band","k":2}'
    assert dispatch("semantic_search", args, index, config).error
    monkeypatch.setattr(index, "dense", index.bm25)
    assert dispatch("semantic_search", args, index, LocalizerConfig()).citations


def test_short_caps_and_directory_round_trip(index: RepositoryIndex) -> None:
    config = ToolConfig(max_tool_chars=100)
    assert dispatch("list_dir", '{"path":"postbag/"}', index, config).error is None
    for name, args in (
        ("search_code", {"query": "rate_band", "k": 20}),
        ("get_symbol", {"symbol_id": "postbag/rates.py::RateCard"}),
        ("get_symbol", {"symbol_id": "missing" * 100}),
    ):
        result = dispatch(name, json.dumps(args), index, config)
        assert len(result.text) <= 100
        assert result.truncated


def test_read_range_rejects_symlink_blob(minirepo_git: GitMinirepo, tmp_path: Path) -> None:
    import subprocess

    outside = tmp_path / "outside.py"
    outside.write_text("SECRET_MARKER = True\n")
    root = minirepo_git.root
    (root / "postbag/escape.py").symlink_to(outside)
    subprocess.run(["git", "add", "postbag/escape.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-m",
            "Add fixture symlink",
        ],
        cwd=root,
        capture_output=True,
        check=True,
    )
    index = build_index(root, "HEAD", tmp_path / "cache", dense=False)
    result = dispatch(
        "read_range",
        '{"path":"postbag/escape.py","start_line":1,"end_line":1}',
        index,
        LocalizerConfig(),
    )
    assert result.error
    assert not result.citations
    assert "SECRET_MARKER" not in result.text
