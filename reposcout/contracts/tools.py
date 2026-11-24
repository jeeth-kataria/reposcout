from typing import Literal

from pydantic import JsonValue

from .models import ContractModel, Span

type ToolName = Literal[
    "search_code", "semantic_search", "get_symbol", "callers", "callees", "read_range", "list_dir"
]


class ToolSpec(ContractModel):
    name: ToolName
    description: str
    json_schema: dict[str, JsonValue]


class ToolResult(ContractModel):
    text: str
    citations: list[Span]
    truncated: bool
    error: str | None = None
