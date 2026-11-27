import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from reposcout.contracts.llm import ChatMessage, ToolCall, ToolFunction
from reposcout.contracts.tools import ToolSpec
from reposcout.llm import LLMError, OpenAIChatClient

from .fake import FakeChatClient

SOURCE = (Path(__file__).parents[1] / "fixtures/minirepo/postbag/rates.py").read_text()
ISSUE = ChatMessage(role="user", content="Find the pricing error:\n" + SOURCE)
TOOL = ToolSpec(
    name="list_dir",
    description="List a directory",
    json_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
CALL = ToolCall(id="call1", function=ToolFunction(name="list_dir", arguments='{"path":"."}'))


def completion(reason: str = "stop") -> dict:
    message = {"role": "assistant", "content": "The rate band uses a strict comparison."}
    if reason == "tool_calls":
        message = {"role": "assistant", "content": None, "tool_calls": [CALL.model_dump()]}
    return {
        "id": "c-1",
        "object": "chat.completion",
        "created": 1,
        "model": "scripted",
        "choices": [{"index": 0, "finish_reason": reason, "message": message}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
    }


def client(handler: Callable[[httpx.Request], httpx.Response]) -> OpenAIChatClient:
    return OpenAIChatClient(
        OpenAI(
            api_key="fixture-secret",
            base_url="https://example.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        "scripted",
    )


def test_tool_replay_and_forced_answer_preserve_history(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies = []
    calls = []
    monkeypatch.setattr("reposcout.eval.budget.before_request", lambda: calls.append("tick"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert calls.pop() == "tick" and request.url.path == "/v1/chat/completions"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion("tool_calls" if len(bodies) == 1 else "stop"))

    sdk = client(handler)
    first = sdk.chat(
        [ISSUE], [TOOL], "scripted", 200, 8, seed=23, extra={"reasoning_effort": "low"}
    )
    assert first.usage.prompt_tokens == 20 and first.tool_calls == [CALL]
    assert bodies[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": TOOL.description,
                "parameters": TOOL.json_schema,
            },
        }
    ]
    assert bodies[0]["temperature"] == 0 and bodies[0]["seed"] == 23
    assert bodies[0]["max_completion_tokens"] == 200 and bodies[0]["reasoning_effort"] == "low"
    result = sdk.chat(
        [ISSUE, first.message, ChatMessage(role="tool", tool_call_id="call1", content="rates.py")],
        [TOOL],
        "scripted",
        200,
        8,
        tool_choice="none",
    )
    assert result.finish_reason == "stop" and result.usage.completion_tokens == 7
    assert bodies[1]["tools"] == bodies[0]["tools"] and bodies[1]["tool_choice"] == "none"
    assert (
        bodies[1]["messages"][1]["tool_calls"][0]["function"]["arguments"]
        == CALL.function.arguments
    )
    assert bodies[1]["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call1",
        "content": "rates.py",
    }
    sdk.close()


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (429, "rate_limit"),
        (413, "request_too_large"),
        (408, "timeout"),
        (503, "http_status"),
    ],
)
def test_http_errors_are_sanitized_and_attempted_once(code: int, kind: str) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(code, json={"error": {"message": "fixture-secret " + SOURCE}})

    sdk = client(handler)
    with pytest.raises(LLMError) as error:
        sdk.chat([ISSUE], [], "scripted", 50, 2)
    assert error.value.kind == kind and error.value.status_code == code
    assert "fixture-secret" not in str(error.value) and SOURCE not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("exception", "kind"), [(httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "connection")]
)
def test_unreachable_provider_is_classified_once(exception: type[Exception], kind: str) -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise exception("fixture-secret")

    with pytest.raises(LLMError) as error:
        client(handler).chat([ISSUE], [], "scripted", 10, 2)
    assert (
        error.value.kind == kind and len(attempts) == 1 and "fixture-secret" not in str(error.value)
    )


def test_unsupported_finish_retains_usage_and_missing_usage_is_an_error() -> None:
    with pytest.raises(LLMError, match="token limit") as error:
        client(lambda request: httpx.Response(200, json=completion("length"))).chat(
            [ISSUE], [], "scripted", 10, 2
        )
    assert error.value.usage.completion_tokens == 7
    malformed = completion()
    del malformed["usage"]
    with pytest.raises(LLMError, match="token usage"):
        client(lambda request: httpx.Response(200, json=malformed)).chat(
            [ISSUE], [], "scripted", 10, 2
        )


def test_extra_cannot_override_budget_and_missing_tool_ids_rejected() -> None:
    sdk = client(lambda request: pytest.fail("unexpected provider invocation"))
    with pytest.raises(LLMError, match="controlled"):
        sdk.chat([ISSUE], [], "scripted", 10, 2, extra={"max_completion_tokens": 100})
    with pytest.raises(LLMError, match="original tool call ID"):
        sdk.chat([ChatMessage(role="tool", content="")], [TOOL], "scripted", 10, 2)
    with pytest.raises(LLMError, match="at least one"):
        sdk.chat([ISSUE], [], "scripted", 10, 2, tool_choice="required")


def test_custom_endpoint_uses_role_specific_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEACHER_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-only")
    monkeypatch.setenv("TEACHER_BASE_URL", "http://localhost:9010/v1")
    with pytest.raises(LLMError, match="TEACHER_API_KEY"):
        OpenAIChatClient.from_env("teacher")
    monkeypatch.setenv("TEACHER_API_KEY", "local-only")
    monkeypatch.setenv("TEACHER_MODEL", "teacher-a")
    teacher = OpenAIChatClient.from_env("teacher")
    assert teacher.model == "teacher-a"
    teacher.close()
    monkeypatch.setenv("STUDENT_MODEL", "student-b")
    student = OpenAIChatClient.from_env("student")
    assert student.model == "student-b"
    student.close()


def test_fake_records_independent_snapshots_and_script_exhaustion() -> None:
    provider = client(lambda request: httpx.Response(200, json=completion()))
    result = provider.chat([ISSUE], [], "scripted", 10, 2)
    fake = FakeChatClient([result, LLMError("timeout", "timed out")])
    message = ISSUE.model_copy()
    fake.chat([message], [], "scripted", 100, 2)
    message.content = "different"
    assert fake.requests[0].messages[0].content == ISSUE.content
    with pytest.raises(LLMError, match="timed out"):
        fake.chat([ISSUE], [], "scripted", 100, 2)
    with pytest.raises(AssertionError, match="unscripted"):
        fake.chat([ISSUE], [], "scripted", 100, 2)
