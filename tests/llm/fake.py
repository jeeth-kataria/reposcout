from collections import deque
from typing import Literal

from pydantic import BaseModel, JsonValue

from reposcout.contracts.llm import ChatMessage, ChatResult
from reposcout.contracts.tools import ToolSpec
from reposcout.llm.client import LLMError


class RecordedChat(BaseModel):
    messages: list[ChatMessage]
    tools: list[ToolSpec]
    model: str
    max_tokens: int
    timeout: float
    tool_choice: Literal["auto", "none", "required"]
    temperature: float
    seed: int | None
    extra: dict[str, JsonValue] | None


class FakeChatClient:
    def __init__(self, replies: list[ChatResult | LLMError], model: str = "scripted") -> None:
        self.model = model
        self.replies = deque(replies)
        self.requests: list[RecordedChat] = []

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
    ) -> ChatResult:
        request = RecordedChat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            tool_choice=tool_choice,
            temperature=temperature,
            seed=seed,
            extra=extra,
        )
        self.requests.append(request.model_copy(deep=True))
        if not self.replies:
            raise AssertionError(f"unscripted chat invocation {len(self.requests)}")
        reply = self.replies.popleft()
        if isinstance(reply, LLMError):
            raise reply
        return reply.model_copy(deep=True)

    def close(self) -> None:
        pass
