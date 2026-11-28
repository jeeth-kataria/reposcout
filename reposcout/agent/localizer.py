import os
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph

from reposcout.contracts.index import RepoIndex
from reposcout.contracts.llm import ChatClient
from reposcout.contracts.localizer import LocalizerConfig
from reposcout.contracts.models import Instance, Localization, RunStats
from reposcout.contracts.registry import register

from .nodes.explore import explore
from .nodes.rank import rank
from .nodes.rewrite import rewrite
from .nodes.route import after_explore, after_seed, after_verify
from .nodes.seed import seed
from .nodes.verify import verify
from .runtime import Session, save_trace, trace_file
from .state import AgentConfig, AgentState


@register("agent")
class AgentLocalizer:
    def __init__(
        self,
        client: ChatClient | None = None,
        model: str = "",
        role: Literal["teacher", "student"] = "teacher",
        trace_root: Path | None = None,
        disable_graph: bool = False,
        disable_verifier: bool = False,
    ):
        self.client, self.model, self.role = client, model, role
        self.trace_root = trace_root
        self.disable_graph, self.disable_verifier = disable_graph, disable_verifier

    def localize(
        self, instance: Instance, index: RepoIndex, config: LocalizerConfig
    ) -> Localization:
        cfg = AgentConfig.model_validate(config.model_dump())
        if self.disable_graph:
            cfg.use_graph_tools = False
        if self.disable_verifier:
            cfg.use_verifier = False
        state: AgentState = {
            "instance": instance.model_copy(update={"gold_files": [], "gold_functions": []}),
            "config": cfg,
            "queries": [instance.problem_statement],
            "identifiers": [],
            "suspected_paths": [],
            "seed_hits": [],
            "pool": [],
            "messages": [],
            "verdicts": {},
            "hint": None,
            "need_more": False,
            "seen": set(),
            "rounds": 0,
            "stats": RunStats(status="complete"),
            "trace": [],
            "localization": None,
        }
        stream, dest = trace_file(
            self.trace_root or Path(os.getenv("REPOSCOUT_TRACE_ROOT", "traces")),
            os.getenv("REPOSCOUT_RUN_NAME", "interactive"),
            instance.instance_id,
        )
        session = Session(index, self.client, self.model, self.role, stream)
        try:
            session.event(
                state,
                "start",
                "instance",
                repo=instance.repo,
                base_commit=instance.base_commit,
                instance_id=instance.instance_id,
                config=cfg.model_dump(),
                schema_version=1,
            )
            graph = StateGraph(AgentState)
            for name, node in (
                ("rewrite", rewrite),
                ("seed", seed),
                ("explore", explore),
                ("verify", verify),
                ("rank", rank),
            ):
                graph.add_node(name, session.timed(name, node))
            graph.add_edge(START, "rewrite")
            graph.add_edge("rewrite", "seed")
            graph.add_conditional_edges("seed", after_seed)
            graph.add_conditional_edges("explore", after_explore)
            graph.add_conditional_edges("verify", after_verify)
            graph.add_edge("rank", END)
            finished = graph.compile().invoke(
                state, {"recursion_limit": 2 * cfg.max_explore_rounds + 8}
            )
            result = finished["localization"]
            if result is None:
                raise RuntimeError(finished["stats"].error or "rank failed")
            session.event(finished, "rank", "localization", localization=result.model_dump())
            return result
        finally:
            save_trace(stream, dest)
            if self.client is None and session.client is not None:
                session.client.close()


@register("agent_no_graph")
def no_graph() -> AgentLocalizer:
    return AgentLocalizer(disable_graph=True)


@register("agent_no_verifier")
def no_verifier() -> AgentLocalizer:
    return AgentLocalizer(disable_verifier=True)


@register("agent_student")
def student() -> AgentLocalizer:
    return AgentLocalizer(role="student")
