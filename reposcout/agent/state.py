from typing import Annotated, TypedDict

from pydantic import Field, StrictBool

from reposcout.contracts.llm import ChatMessage
from reposcout.contracts.models import (
    Candidate,
    ContractModel,
    Instance,
    Localization,
    PositiveInt,
    RepoPath,
    RunStats,
    SearchHit,
    Span,
)
from reposcout.tools.registry import ToolConfig


class AgentConfig(ToolConfig):
    query_limit: PositiveInt = 5
    identifier_limit: PositiveInt = 10
    path_limit: PositiveInt = 5
    retrieval_depth: PositiveInt = 30
    seed_limit: PositiveInt = 30
    explore_limit: int = Field(default=10, ge=1, le=10)
    verifier_limit: int = Field(default=5, ge=1, le=5)
    citation_limit: PositiveInt = 5
    issue_chars: PositiveInt = 12000
    verifier_lines: PositiveInt = 50
    rrf_k: PositiveInt = 60
    seed_weight: float = Field(default=0.2, ge=0)
    explore_weight: float = Field(default=0.3, ge=0)
    verifier_weight: float = Field(default=0.5, ge=0)
    seed: int | None = 42


class Rewrite(ContractModel):
    queries: list[str]
    identifiers: list[str]
    suspected_paths: list[str]


class Draft(ContractModel):
    path: RepoPath
    symbol_id: str | None
    rationale: str = Field(min_length=1)
    citations: Annotated[list[Span], Field(min_length=1)]


class Selection(ContractModel):
    candidates: Annotated[list[Draft], Field(min_length=1, max_length=10)]


class Verdict(ContractModel):
    symbol_id: str | None = None
    path: RepoPath | None = None
    keep: StrictBool
    reason: str = Field(min_length=1)


class Review(ContractModel):
    verdicts: list[Verdict]
    need_more: StrictBool
    next_hint: str | None


class AgentState(TypedDict):
    instance: Instance
    config: AgentConfig
    queries: list[str]
    identifiers: list[str]
    suspected_paths: list[str]
    seed_hits: list[SearchHit]
    pool: list[Candidate]
    messages: list[ChatMessage]
    verdicts: dict[str, Verdict]
    hint: str | None
    need_more: bool
    seen: set[str]
    rounds: int
    stats: RunStats
    trace: list[dict]
    localization: Localization | None
