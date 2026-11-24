from typing import Literal, Protocol, Self

from pydantic import Field, JsonValue, model_validator

from .models import ContractModel, Count
from .tools import ToolSpec


class Usage(ContractModel):
    prompt_tokens: Count
    completion_tokens: Count


class ToolFunction(ContractModel):
    name: str
    arguments: str


class ToolCall(ContractModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolFunction


class ChatMessage(ContractModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ChatResult(ContractModel):
    message: ChatMessage
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: Literal["stop", "tool_calls"]

    @model_validator(mode="after")
    def validate_assistant_turn(self) -> Self:
        if self.message.role != "assistant" or self.message.tool_calls != self.tool_calls:
            raise ValueError("the assistant message must carry the returned tool calls")
        if bool(self.tool_calls) != (self.finish_reason == "tool_calls"):
            raise ValueError("finish_reason must agree with the presence of tool calls")
        return self


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        model: str,
        max_tokens: int,
        timeout: float,
        *,
        tool_choice: Literal["auto", "none", "required"] = "auto",
        temperature: float = 0.0,
        seed: int | None = None,
        extra: dict[str, JsonValue] | None = None,
    ) -> ChatResult: ...
