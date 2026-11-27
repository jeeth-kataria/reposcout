"""Single-attempt OpenAI-compatible text chat for Groq and local vLLM."""

import math
import os
from typing import Literal, Self, cast
from urllib.parse import urlsplit

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import JsonValue, ValidationError

from reposcout.contracts.llm import ChatMessage, ChatResult, ToolCall, Usage
from reposcout.contracts.tools import ToolSpec
from reposcout.eval import budget

type Role = Literal["teacher", "student"]
type ToolChoice = Literal["auto", "none", "required"]
type ErrorKind = Literal[
    "configuration",
    "authentication",
    "rate_limit",
    "timeout",
    "request_too_large",
    "http_status",
    "connection",
    "invalid_response",
    "finish_reason",
]

DEFAULT_TEACHER_MODEL = "openai/gpt-oss-20b"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
_CONTROLLED = {
    "model",
    "messages",
    "tools",
    "tool_choice",
    "temperature",
    "seed",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "n",
}


class LLMError(Exception):
    """Safe for run diagnostics. Provider exception bodies may contain source or credentials."""

    def __init__(
        self,
        kind: ErrorKind,
        detail: str,
        *,
        status_code: int | None = None,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(detail)
        self.kind = kind
        self.status_code = status_code
        self.usage = usage


def _status_error(code: int) -> LLMError:
    conditions: dict[int, tuple[ErrorKind, str]] = {
        401: ("authentication", "The provider rejected the API credentials."),
        403: ("authentication", "The provider denied access."),
        408: ("timeout", "The provider timed out."),
        413: ("request_too_large", "The provider rejected the request size."),
        429: ("rate_limit", "The provider rate limit was reached."),
        504: ("timeout", "The provider gateway timed out."),
    }
    kind, detail = conditions.get(code, ("http_status", f"The provider returned HTTP {code}."))
    return LLMError(kind, detail, status_code=code)


def _message(message: ChatMessage) -> ChatCompletionMessageParam:
    data = message.model_dump(exclude_none=True)
    if not message.tool_calls:
        data.pop("tool_calls")
    # A tool-calling assistant may have null content. Leave textual messages untouched.
    if message.role == "tool" and not message.tool_call_id:
        raise LLMError("configuration", "A tool message requires its original tool call ID.")
    return cast(ChatCompletionMessageParam, data)


def _result(reply: ChatCompletion) -> ChatResult:
    usage = None
    try:
        if reply.usage is None:
            raise LLMError("invalid_response", "The provider omitted token usage.")
        usage = Usage(
            prompt_tokens=reply.usage.prompt_tokens, completion_tokens=reply.usage.completion_tokens
        )
        if len(reply.choices) != 1:
            raise LLMError("invalid_response", "Expected one completion choice.", usage=usage)
        choice = reply.choices[0]
        if choice.finish_reason not in ("tool_calls", "stop"):
            detail = (
                "The provider reached the output token limit."
                if choice.finish_reason == "length"
                else "The provider returned an unsupported finish reason."
            )
            raise LLMError("finish_reason", detail, usage=usage)
        message = choice.message
        calls = [
            ToolCall.model_validate(
                call.model_dump(
                    include={"id": True, "type": True, "function": {"name", "arguments"}}
                )
            )
            for call in message.tool_calls or []
        ]
        assistant = ChatMessage(role=message.role, content=message.content, tool_calls=calls)
        return ChatResult(
            message=assistant, tool_calls=calls, usage=usage, finish_reason=choice.finish_reason
        )
    except (ValidationError, AttributeError, TypeError, KeyError):
        raise LLMError("invalid_response", "Malformed provider completion.", usage=usage) from None


class OpenAIChatClient:
    def __init__(self, sdk: OpenAI, model: str) -> None:
        self._sdk = sdk.with_options(max_retries=0)
        self.model = model

    @classmethod
    def from_env(cls, role: Role) -> Self:
        if role == "teacher":
            base = os.getenv("TEACHER_BASE_URL") or "https://api.groq.com/openai/v1"
            key = os.getenv("TEACHER_API_KEY")
            if base.rstrip("/") == "https://api.groq.com/openai/v1":
                key = key or os.getenv("GROQ_API_KEY")
            model = os.getenv("TEACHER_MODEL") or DEFAULT_TEACHER_MODEL
            if not key:
                raise LLMError("configuration", "Set TEACHER_API_KEY for the teacher endpoint.")
        elif role == "student":
            base = os.getenv("STUDENT_BASE_URL") or "http://127.0.0.1:8000/v1"
            model = os.getenv("STUDENT_MODEL") or DEFAULT_STUDENT_MODEL
            key = os.getenv("STUDENT_API_KEY") or "local-vllm"
        else:
            raise LLMError("configuration", "Unknown chat client role.")
        try:
            endpoint = urlsplit(base)
            if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
                raise ValueError
            sdk = OpenAI(base_url=base, api_key=key, max_retries=0)
        except (ValueError, APIError):
            raise LLMError("configuration", "Invalid provider endpoint configuration.") from None
        return cls(sdk, model)

    def close(self) -> None:
        self._sdk.close()

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        model: str,
        max_tokens: int,
        timeout: float,
        *,
        tool_choice: ToolChoice = "auto",
        temperature: float = 0.0,
        seed: int | None = None,
        extra: dict[str, JsonValue] | None = None,
    ) -> ChatResult:
        if not (math.isfinite(timeout) and timeout > 0 and max_tokens > 0):
            raise LLMError("configuration", "Chat timeout and token budget must be positive.")
        if not math.isfinite(temperature) or tool_choice not in {"auto", "required", "none"}:
            raise LLMError("configuration", "Invalid sampling or tool choice setting.")
        if extra and _CONTROLLED.intersection(extra):
            raise LLMError("configuration", "Extra settings override a controlled chat parameter.")
        if tool_choice == "required" and not tools:
            raise LLMError("configuration", "Required tool choice needs at least one tool.")
        payload = [_message(message) for message in messages]
        schemas: list[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema,
                },
            }
            for tool in tools
        ]
        if not model.strip():
            raise LLMError("configuration", "The chat model name is missing.")
        try:
            budget.before_request()
            reply = self._sdk.chat.completions.create(
                messages=payload,
                model=model,
                max_completion_tokens=max_tokens,
                timeout=timeout,
                temperature=temperature,
                extra_body=extra,
                **({"seed": seed} if seed is not None else {}),
                **({"tools": schemas, "tool_choice": tool_choice} if schemas else {}),
            )
        except APITimeoutError:
            raise LLMError("timeout", "The provider request timed out.") from None
        except APIStatusError as exc:
            raise _status_error(exc.status_code) from None
        except APIConnectionError:
            raise LLMError("connection", "Could not connect to the provider.") from None
        except (APIError, ValidationError, ValueError):
            raise LLMError(
                "invalid_response", "The provider response could not be decoded."
            ) from None
        return _result(reply)
