import pytest
from pydantic import ValidationError

from reposcout.contracts.localizer import LocalizerConfig
from reposcout.contracts.models import Candidate, Instance, Localization, RunStats, Span, Symbol


@pytest.mark.parametrize(
    "path",
    ["", ".", "..", "/tmp/x.py", "p/../x", "./x", "p//x", "p/", r"p\x", "C:/x", "C:x", "a\0b"],
)
def test_file_paths_are_repository_relative(path: str) -> None:
    with pytest.raises(ValidationError):
        Span(path=path, start_line=1, end_line=1)


@pytest.mark.parametrize("lines", [(0, 1), (1, 0), (3, 2), (True, 2), (1.5, 2)])
def test_span_bounds(lines: tuple[int | float, int]) -> None:
    with pytest.raises(ValidationError):
        Span(path="src/a.py", start_line=lines[0], end_line=lines[1])


def test_symbol_identity_and_round_trip() -> None:
    span = Span(path="pkg/rates é.py", start_line=3, end_line=3)
    symbol = Symbol(
        id="pkg/rates é.py::Card.quote",
        kind="method",
        name="quote",
        qualname="Card.quote",
        span=span,
    )
    assert Symbol.model_validate_json(symbol.model_dump_json()) == symbol
    with pytest.raises(ValidationError, match="symbol id"):
        Symbol.model_validate({**symbol.model_dump(), "id": "pkg/rates é.py::quote"})


def test_paths_in_nested_result_and_gold_records() -> None:
    with pytest.raises(ValidationError):
        Candidate(path="/local/file", score=1, rationale="issue mentions file", citations=[])
    with pytest.raises(ValidationError):
        Instance(
            instance_id="x",
            repo="a/b",
            base_commit="abc",
            problem_statement="x",
            gold_files=["bad/../path"],
        )
    with pytest.raises(ValidationError):
        Localization(
            instance_id="x",
            files=["/secret"],
            functions=[],
            candidates=[],
            stats=RunStats(status="complete"),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"llm_calls": -1},
        {"prompt_tokens": True},
        {"latency_s": float("nan")},
        {"status": "error"},
        {"status": "error", "error": "  "},
        {"status": "unknown"},
    ],
)
def test_invalid_run_stats(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunStats.model_validate({"status": "complete", **values})


def test_failed_run_has_serializable_diagnostic() -> None:
    stats = RunStats(status="error", error="provider timed out", llm_calls=1)
    assert RunStats.model_validate_json(stats.model_dump_json()) == stats


def test_budgets_and_extra_config_keys() -> None:
    defaults = LocalizerConfig()
    assert (defaults.max_llm_calls, defaults.max_tool_calls, defaults.max_explore_rounds) == (
        12,
        20,
        2,
    )
    assert LocalizerConfig(max_llm_calls=0).max_llm_calls == 0
    for overrides in ({"max_calls": 1}, {"max_llm_calls": -1}, {"top_k_files": 0}):
        with pytest.raises(ValidationError):
            LocalizerConfig.model_validate(overrides)
