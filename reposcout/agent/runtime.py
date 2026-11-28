import json
import os
import re
import tempfile
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from time import monotonic
from typing import Literal, TextIO

from reposcout.contracts.index import RepoIndex
from reposcout.contracts.llm import ChatClient, ChatMessage, ChatResult
from reposcout.contracts.models import ContractModel
from reposcout.contracts.tools import ToolResult, ToolSpec
from reposcout.tools.registry import dispatch

from .state import AgentState


class OutOfBudget(Exception):
    pass


def prompt(name: str) -> str:
    return files("reposcout.agent").joinpath("prompts", name + ".md").read_text()


def decode[T: ContractModel](raw: str, schema: type[T]) -> T:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        values = dict(pairs)
        if len(values) != len(pairs):
            raise ValueError("duplicate JSON keys")
        return values

    def invalid(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value}")

    return schema.model_validate(json.loads(raw, object_pairs_hook=unique, parse_constant=invalid))


def wire(message: ChatMessage) -> dict:
    value = message.model_dump(exclude_none=True)
    if not message.tool_calls:
        value.pop("tool_calls", None)
    return value


def clipped(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[Input truncated at {limit} of {len(text)} characters.]"


def repairable_json(text: str) -> str:
    """Allow a Markdown wrapper and trailing commas outside quoted strings."""
    fenced = re.fullmatch(r"\s*```(?:json)?[ \t]*\n(.*?)\n```\s*", text, flags=re.DOTALL)
    if fenced:
        text = fenced[1]
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|(,)(?=\s*[}\]])', flags=re.DOTALL)
    return tokens.sub(lambda match: "" if match[1] else match[0], text)


class Session:
    def __init__(
        self,
        index: RepoIndex,
        client: ChatClient | None,
        model: str,
        role: Literal["teacher", "student"],
        stream: TextIO,
    ):
        self.index = index
        self.client = client
        self.model = model
        self.role = role
        self.stream = stream
        self.started = monotonic()

    def event(self, state: AgentState, node: str, event: str, **data: object) -> None:
        row = {"event": event, "node": node, "elapsed_s": monotonic() - self.started, **data}
        state["trace"].append(row)
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()

    def chat(
        self,
        s: AgentState,
        node: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        choice: Literal["auto", "none"] = "none",
    ) -> ChatResult:
        stats, config = s["stats"], s["config"]
        remaining = config.timeout_s - (monotonic() - self.started)
        if stats.llm_calls >= config.max_llm_calls or remaining <= 0:
            raise OutOfBudget("LLM call or wall-clock budget reached")
        if self.client is None:
            from reposcout.llm.client import OpenAIChatClient

            client = OpenAIChatClient.from_env(self.role)
            self.client, self.model = client, client.model
        stats.model = self.model
        specs = [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.json_schema,
                },
            }
            for spec in tools
        ]
        stats.llm_calls += 1
        self.event(
            s,
            node,
            "llm_request",
            call=stats.llm_calls,
            model=self.model,
            messages=[wire(m) for m in messages],
            tools=specs,
            tool_choice=choice,
            temperature=0.0,
            seed=config.seed,
            max_tokens=config.max_output_tokens,
        )
        try:
            result = self.client.chat(
                messages,
                tools,
                self.model,
                config.max_output_tokens,
                remaining,
                tool_choice=choice,
                temperature=0.0,
                seed=config.seed,
            )
        except Exception as exc:
            if usage := getattr(exc, "usage", None):
                stats.prompt_tokens += usage.prompt_tokens
                stats.completion_tokens += usage.completion_tokens
            self.event(s, node, "llm_error", call=stats.llm_calls, error=str(exc))
            raise
        stats.prompt_tokens += result.usage.prompt_tokens
        stats.completion_tokens += result.usage.completion_tokens
        self.event(
            s,
            node,
            "llm_response",
            call=stats.llm_calls,
            message=wire(result.message),
            usage=result.usage.model_dump(),
            finish_reason=result.finish_reason,
        )
        return result

    def tool(
        self, s: AgentState, node: str, name: str, args: str, call_id: str | None = None
    ) -> ToolResult:
        if s["stats"].tool_calls >= s["config"].max_tool_calls:
            raise OutOfBudget("Tool call budget reached")
        if monotonic() - self.started >= s["config"].timeout_s:
            raise OutOfBudget("Wall-clock budget reached")
        s["stats"].tool_calls += 1
        start = monotonic()
        result = dispatch(name, args, self.index, s["config"])
        s["seen"].update(c.path for c in result.citations)
        self.event(
            s,
            node,
            "tool_result",
            name=name,
            arguments=args,
            tool_call_id=call_id,
            latency_s=monotonic() - start,
            **result.model_dump(),
        )
        return result

    def timed(self, name: str, node: Callable) -> Callable:
        def invoke(state: AgentState) -> AgentState:
            before = monotonic()
            try:
                node(state, self)
            except OutOfBudget as exc:
                state["stats"].status = "budget_exhausted"
                state["stats"].error = str(exc)
            except Exception as exc:
                state["stats"].status = "error"
                state["stats"].error = f"{type(exc).__name__}: {exc}"
            self.event(
                state,
                name,
                "node_end",
                latency_s=monotonic() - before,
                status=state["stats"].status,
            )
            return state

        return invoke


def trace_file(root: Path, run: str, instance_id: str) -> tuple[TextIO, Path]:
    for name in (run, instance_id):
        if name in {"", ".", ".."} or Path(name).name != name or "\\" in name:
            raise ValueError("invalid trace run or instance name")
    directory = root / run
    directory.mkdir(parents=True, exist_ok=True)
    output = tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False, encoding="utf-8")
    return output, directory / (instance_id + ".jsonl")


def save_trace(stream: TextIO, dest: Path) -> None:
    stream.close()
    os.replace(stream.name, dest)
