from pathlib import PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _repo_path(path: str) -> str:
    if (
        any(part in {"", ".", ".."} for part in path.split("/"))
        or "\\" in path
        or "\x00" in path
        or PureWindowsPath(path).drive
    ):
        raise ValueError("expected a canonical repository-relative POSIX file path")
    return path


RepoPath = Annotated[str, AfterValidator(_repo_path)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
Count = Annotated[int, Field(strict=True, ge=0)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Span(ContractModel):
    path: RepoPath
    start_line: PositiveInt
    end_line: PositiveInt

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.start_line > self.end_line:
            raise ValueError("start_line must be at or before end_line")
        return self


class Symbol(ContractModel):
    id: str
    kind: Literal["module", "class", "function", "method"]
    name: str
    qualname: str = Field(min_length=1)
    span: Span
    parent_id: str | None = None
    is_test: bool = False

    @model_validator(mode="after")
    def validate_id(self) -> Self:
        if self.id != f"{self.span.path}::{self.qualname}":
            raise ValueError("symbol id must equal <path>::<qualname>")
        return self


class SearchHit(ContractModel):
    symbol_id: str | None = None
    span: Span
    score: float
    source: str
    snippet: str


class Candidate(ContractModel):
    path: RepoPath
    symbol_id: str | None = None
    score: float
    rationale: str
    citations: list[Span]


class RunStats(ContractModel):
    llm_calls: Count = 0
    tool_calls: Count = 0
    prompt_tokens: Count = 0
    completion_tokens: Count = 0
    latency_s: float = Field(default=0, ge=0)
    model: str | None = None
    status: Literal["complete", "budget_exhausted", "error"]
    error: str | None = None

    @model_validator(mode="after")
    def explain_failure(self) -> Self:
        if self.status == "error" and not (self.error and self.error.strip()):
            raise ValueError("an error run needs an error message")
        return self


class Localization(ContractModel):
    instance_id: str
    files: list[RepoPath]
    functions: list[str]
    candidates: list[Candidate]
    stats: RunStats


class Instance(ContractModel):
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_files: list[RepoPath] = Field(default_factory=list)
    gold_functions: list[str] = Field(default_factory=list)
