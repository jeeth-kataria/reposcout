from typing import Literal

from ..state import AgentState


def after_seed(s: AgentState) -> Literal["explore", "rank"]:
    return "explore" if s["stats"].status == "complete" else "rank"


def after_explore(s: AgentState) -> Literal["verify", "rank"]:
    return "verify" if s["stats"].status == "complete" and s["config"].use_verifier else "rank"


def after_verify(s: AgentState) -> Literal["explore", "rank"]:
    if s["stats"].status == "complete" and s["need_more"]:
        if (
            s["rounds"] < s["config"].max_explore_rounds
            and s["stats"].llm_calls < s["config"].max_llm_calls
        ):
            return "explore"
        s["stats"].status = "budget_exhausted"
        s["stats"].error = "Verifier requested more evidence after the exploration budget was spent"
    return "rank"
