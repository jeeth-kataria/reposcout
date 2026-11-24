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
