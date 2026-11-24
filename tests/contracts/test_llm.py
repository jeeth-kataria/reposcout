import pytest
from pydantic import ValidationError

from reposcout.contracts.llm import ChatMessage, ChatResult, ToolCall, ToolFunction, Usage


def test_tool_call_survives_trace_round_trip() -> None:
    call = ToolCall(id="call1", function=ToolFunction(name="list_dir", arguments='{"path":"."}'))
    result = ChatResult(
        message=ChatMessage(role="assistant", content=None, tool_calls=[call]),
        tool_calls=[call],
        usage=Usage(prompt_tokens=10, completion_tokens=2),
        finish_reason="tool_calls",
    )
    assert ChatResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError, match="assistant message"):
        ChatResult.model_validate({**result.model_dump(), "tool_calls": []})
    with pytest.raises(ValidationError, match="finish_reason"):
        ChatResult.model_validate({**result.model_dump(), "finish_reason": "stop"})


def test_chat_request_controls_are_optional_and_can_force_final_answer() -> None:
    from inspect import signature

    from reposcout.contracts.llm import ChatClient

    chat = signature(ChatClient.chat)
    history = [ChatMessage(role="tool", content="found", tool_call_id="call1")]
    args = (None, history, [], "model", 128, 5)
    defaults = chat.bind(*args)
    defaults.apply_defaults()
    assert [defaults.arguments[k] for k in ("tool_choice", "temperature", "seed", "extra")] == [
        "auto",
        0.0,
        None,
        None,
    ]
    request = chat.bind(
        *args, tool_choice="none", temperature=0, seed=7, extra={"reasoning_effort": "low"}
    )
    assert request.arguments["messages"] == history
