import json
from pathlib import Path

import pytest
from conftest import GitMinirepo
from llm.fake import FakeChatClient

from reposcout.agent.localizer import AgentLocalizer
from reposcout.agent.state import AgentConfig
from reposcout.contracts.llm import ChatMessage, ChatResult, ToolCall, ToolFunction, Usage
from reposcout.contracts.models import Instance
from reposcout.index import build_index
from reposcout.index.repository import RepositoryIndex
from reposcout.llm.client import LLMError

PATH = "postbag/rates.py"
SID = PATH + "::RateCard.rate_band"


def reply(content: str | dict | None = None, tools: list[ToolCall] | None = None) -> ChatResult:
    calls = tools or []
    message = ChatMessage(
        role="assistant",
        content=json.dumps(content) if isinstance(content, dict) else content,
        tool_calls=calls,
    )
    return ChatResult(
        message=message,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def rewriting() -> ChatResult:
    return reply({"queries": ["rate_band"], "identifiers": ["RateCard"], "suspected_paths": [PATH]})


def candidate(end: int = 10) -> dict:
    return {
        "candidates": [
            {
                "path": PATH,
                "symbol_id": SID,
                "rationale": "The boundary comparison excludes 500g.",
                "citations": [{"path": PATH, "start_line": 9, "end_line": end}],
            }
        ]
    }


def verdict(keep: bool, more: bool) -> ChatResult:
    return reply(
        {
            "verdicts": [{"symbol_id": SID, "keep": keep, "reason": "Check eligibility"}],
            "need_more": more,
            "next_hint": "inspect the inclusive threshold" if more else None,
        }
    )


@pytest.fixture
def index(minirepo_git: GitMinirepo, tmp_path: Path) -> RepositoryIndex:
    return build_index(minirepo_git.root, minirepo_git.base_commit, tmp_path / "cache", dense=False)


def localize(fake: FakeChatClient, index: RepositoryIndex, tmp_path: Path, **kwargs: object):
    issue = Instance(
        instance_id="bug",
        repo="fixture",
        base_commit="example",
        problem_statement="rate_band charges the wrong rate for a local 500 gram letter",
        gold_files=["secret_gold.py"],
        gold_functions=["secret_gold.py::leak"],
    )
    cfg = AgentConfig(use_dense=False, **kwargs)
    return AgentLocalizer(fake, fake.model, trace_root=tmp_path).localize(issue, index, cfg)


def test_happy_path_and_replay(tmp_path: Path, index: RepositoryIndex) -> None:
    fake = FakeChatClient([rewriting(), reply(candidate()), verdict(True, False)])
    prediction = localize(fake, index, tmp_path)
    assert prediction.stats.status == "complete", prediction.stats
    assert prediction.files[0] == PATH and prediction.functions[0] == SID
    assert prediction.stats.llm_calls == 3 and prediction.stats.tool_calls == 1
    assert prediction.stats.prompt_tokens == 30 and prediction.stats.completion_tokens == 15
    trace = (tmp_path / "interactive/bug.jsonl").read_text()
    assert "secret_gold" not in trace
    events = [json.loads(line) for line in trace.splitlines()]
    assert [e["event"] for e in events].count("llm_request") == 3
    assert events[-1]["localization"] == prediction.model_dump()
    assert [e["node"] for e in events if e["event"] == "node_end"] == [
        "rewrite",
        "seed",
        "explore",
        "verify",
        "rank",
    ]


def test_tool_error_recovery_and_forced_final(tmp_path: Path, index: RepositoryIndex) -> None:
    call = ToolCall(
        id="bad", function=ToolFunction(name="get_symbol", arguments='{"symbol_id":"missing"}')
    )
    fake = FakeChatClient([rewriting(), reply(tools=[call]), reply(candidate())])
    result = localize(
        fake, index, tmp_path, use_verifier=False, use_graph_tools=False, max_tool_calls=1
    )
    assert result.stats.status == "complete", result.stats
    assert result.stats.tool_calls == 1
    last = fake.requests[-1]
    assert last.tool_choice == "none" and last.tools
    assert last.messages[-1].tool_call_id == "bad" and "Unknown symbol" in last.messages[-1].content
    assert not {"callers", "callees", "semantic_search"} & {s.name for s in last.tools}


def test_invalid_citation_rejected_and_seed_retained(
    tmp_path: Path, index: RepositoryIndex
) -> None:
    fake = FakeChatClient([rewriting(), reply(candidate(10000))])
    pred = localize(fake, index, tmp_path)
    assert pred.stats.status == "error" and pred.files
    assert not any("excludes" in c.rationale for c in pred.candidates)


def test_format_repair_only_once_and_cannot_change_content(
    tmp_path: Path, index: RepositoryIndex
) -> None:
    raw = json.dumps(candidate())
    fenced = "```json\n" + raw + "\n```"
    fake = FakeChatClient(
        [rewriting(), reply(fenced), reply(raw), verdict(False, True), reply(fenced)]
    )
    pred = localize(fake, index, tmp_path)
    assert (
        pred.stats.status == "error"
        and pred.stats.format_repairs == 1
        and pred.stats.llm_calls == 5
    )
    assert "inclusive threshold" in fake.requests[-1].messages[-1].content
    fake = FakeChatClient([rewriting(), reply(fenced), reply(candidate(11))])
    pred = localize(fake, index, tmp_path)
    assert pred.stats.status == "error" and "changed content" in pred.stats.error


def test_verify_loop_and_budget_bound(tmp_path: Path, index: RepositoryIndex) -> None:
    fake = FakeChatClient(
        [
            rewriting(),
            reply(candidate()),
            verdict(False, True),
            reply(candidate()),
            verdict(True, False),
        ]
    )
    pred = localize(fake, index, tmp_path)
    assert pred.stats.status == "complete" and pred.stats.llm_calls == 5 and pred.files[0] == PATH
    fake = FakeChatClient([rewriting(), reply(candidate()), verdict(False, True)])
    pred = localize(fake, index, tmp_path, max_explore_rounds=1)
    assert pred.stats.status == "budget_exhausted" and pred.files


def test_zero_budget_and_provider_failure_keep_seed_hits(
    tmp_path: Path, index: RepositoryIndex
) -> None:
    fake = FakeChatClient([])
    pred = localize(fake, index, tmp_path, max_llm_calls=0)
    assert pred.stats.status == "budget_exhausted" and pred.files and not fake.requests
    fake = FakeChatClient([LLMError("rate_limit", "rate limit")])
    pred = localize(fake, index, tmp_path)
    assert pred.stats.status == "error" and pred.stats.llm_calls == 1 and pred.files


def test_duplicate_json_keys_rejected(tmp_path: Path, index: RepositoryIndex) -> None:
    bad = json.dumps(candidate()).replace('"path":', '"path":"other.py","path":', 1)
    fake = FakeChatClient([rewriting(), reply(bad), reply(bad)])
    pred = localize(fake, index, tmp_path)
    assert pred.stats.status == "error" and "duplicate JSON keys" in pred.stats.error
