import pytest

from reposcout.contracts.models import Instance, Localization, RunStats
from reposcout.eval.metrics import CUTOFFS, score, summarize


def case(id_: str, files: list[str], funcs: list[str]) -> Instance:
    return Instance(
        instance_id=id_,
        repo="p/p",
        base_commit="a",
        problem_statement="bug",
        gold_files=files,
        gold_functions=funcs,
    )


def pred(id_: str, files: list[str], funcs: list[str], stats: RunStats) -> Localization:
    return Localization(instance_id=id_, files=files, functions=funcs, candidates=[], stats=stats)


def test_rank_metrics_deduplicate_before_cutoff_and_average_per_instance() -> None:
    first = case("a", ["x", "y"], ["x::f", "y::g"])
    best = pred(
        "a",
        ["z", "z", "x", "y"],
        ["x::f", "x::f"],
        RunStats(
            status="budget_exhausted",
            format_repairs=1,
            prompt_tokens=8,
            completion_tokens=2,
            llm_calls=2,
            tool_calls=4,
            latency_s=10,
        ),
    )
    empty = pred("b", [], [], RunStats(status="error", error="timeout"))
    second = case("b", ["z"], [])
    assert score(first, best)["file_hit@1"] == 0
    assert score(first, best)["file_all@3"] == 1
    summary = summarize([(first, best), (second, empty)])
    for k in CUTOFFS:
        assert summary[f"file_hit@{k}"] == (0 if k == 1 else 0.5)
        assert summary[f"file_all@{k}"] == (0 if k == 1 else 0.5)
        assert summary[f"func_recall@{k}"] == 0.5
    assert summary["MRR"] == 0.25
    assert summary["instances"] == 2 and summary["no_gold_functions"] == 1
    for name, val in {
        "tokens": 5,
        "prompt_tokens": 4,
        "completion_tokens": 1,
        "latency_s": 5,
        "llm_calls": 1,
        "tool_calls": 2,
    }.items():
        assert summary[f"mean_{name}"] == summary[f"median_{name}"] == val
    assert (
        summary["error_rate"]
        == summary["budget_exhausted_rate"]
        == summary["format_failure_rate"]
        == 0.5
    )
    assert summary["mean_gold_files"] == 1.5 and summary["mean_gold_functions"] == 1
    assert summarize([(second, empty)])["func_recall@10"] is None
    with pytest.raises(ValueError, match="IDs differ"):
        score(first, empty)
    assert summarize([])["instances"] == 0
    assert summarize([])["MRR"] is None
