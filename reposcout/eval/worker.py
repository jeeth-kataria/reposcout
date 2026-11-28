"""Run a single instance inside a process with a parent-enforced wall-clock timeout."""

import json
import sys
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, SerializeAsAny, ValidationInfo, field_validator

from reposcout import baselines  # noqa: F401 (register built-in factories)
from reposcout.contracts.localizer import LocalizerConfig
from reposcout.contracts.models import Localization, RunStats
from reposcout.contracts.registry import LOCALIZERS, LocalizerFactory

from .dataset import DatasetRow
from .gold import extract_gold
from .repos import worktree


class Job(BaseModel):
    row: DatasetRow
    localizer: str
    cache: Path
    out: Path
    config: SerializeAsAny[LocalizerConfig]
    clean_worktrees: bool = False

    @field_validator("config", mode="before")
    @classmethod
    def restore_agent_config(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, dict) and info.data.get("localizer", "").startswith("agent"):
            from reposcout.agent.state import AgentConfig

            return AgentConfig.model_validate(value)
        return value


def failure(instance_id: str, error: str, elapsed: float) -> Localization:
    return Localization(
        instance_id=instance_id,
        files=[],
        functions=[],
        candidates=[],
        stats=RunStats(status="error", error=error, latency_s=elapsed),
    )


def resolve_localizer(name: str) -> LocalizerFactory:
    if name not in LOCALIZERS and find_spec("reposcout.agent.localizer") is not None:
        from reposcout.agent import localizer  # noqa: F401
    if name not in LOCALIZERS:
        choices = ", ".join(sorted(LOCALIZERS))
        raise ValueError(f"unknown localizer {name!r}. Available: {choices}")
    return LOCALIZERS[name]


def evaluate(job: Job) -> None:
    from reposcout.index import build_index

    started = perf_counter()
    issue = job.row.issue()
    try:
        factory = resolve_localizer(job.localizer)
        with worktree(issue, job.cache, cleanup=job.clean_worktrees) as tree:
            labels = extract_gold(job.row.patch, tree)
            gold = issue.model_copy(
                update={"gold_files": labels.files, "gold_functions": labels.functions}
            )
            (job.out / "gold.json").write_text(gold.model_dump_json())
            index = build_index(
                tree, issue.base_commit, job.cache / "index", dense=job.config.use_dense
            )
            prediction = factory().localize(issue, index, job.config)
            if prediction.instance_id != issue.instance_id:
                raise ValueError("localizer returned the wrong instance ID")
    except Exception as exc:
        prediction = failure(
            issue.instance_id, f"{type(exc).__name__}: {exc}", perf_counter() - started
        )
    (job.out / "prediction.json").write_text(prediction.model_dump_json())


if __name__ == "__main__":
    evaluate(Job.model_validate(json.load(sys.stdin)))
