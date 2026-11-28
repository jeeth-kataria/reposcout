import json

from reposcout.contracts.llm import ChatMessage
from reposcout.contracts.models import Candidate
from reposcout.tools.registry import get_tool_specs

from ..runtime import OutOfBudget, Session, clipped, decode, prompt, repairable_json
from ..state import AgentState, Selection


def _validate(selection: Selection, s: AgentState, session: Session) -> list[Candidate]:
    if len(selection.candidates) > s["config"].explore_limit:
        raise ValueError("too many candidates")
    keys = set()
    pool = []
    for candidate in selection.candidates:
        key = candidate.symbol_id or candidate.path
        if key in keys:
            raise ValueError("duplicate candidate")
        keys.add(key)
        if candidate.path not in s["seen"]:
            raise ValueError(f"unseen candidate path: {candidate.path}")
        if candidate.symbol_id:
            symbol = session.index.get_symbol(candidate.symbol_id)
            if symbol is None or symbol.span.path != candidate.path:
                raise ValueError(f"unknown or mismatched candidate symbol: {candidate.symbol_id}")
        if len(candidate.citations) > s["config"].citation_limit:
            raise ValueError("too many citations")
        if not any(c.path == candidate.path for c in candidate.citations):
            raise ValueError("candidate needs a citation in its own file")
        for citation in candidate.citations:
            # Testing the inclusive endpoint checks bounds without loading an arbitrary-sized span.
            end = citation.model_copy(update={"start_line": citation.end_line})
            if session.index.read(end, max_lines=1).span != end:
                raise ValueError(f"invalid citation endpoint: {end}")
        pool.append(Candidate(score=0.0, **candidate.model_dump()))
    return pool


def explore(s: AgentState, session: Session) -> None:
    cfg = s["config"]
    if s["rounds"] >= cfg.max_explore_rounds:
        raise OutOfBudget("Exploration round budget reached")
    s["rounds"] += 1
    messages = s["messages"]
    if not messages:
        evidence = [
            {"symbol_id": h.symbol_id, "span": h.span.model_dump(), "snippet": h.snippet}
            for h in s["seed_hits"]
        ]
        messages.extend(
            [
                ChatMessage(role="system", content=prompt("explore")),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "issue": clipped(s["instance"].problem_statement, cfg.issue_chars),
                            "seed_evidence": evidence,
                            "max_candidates": cfg.explore_limit,
                            "max_citations": cfg.citation_limit,
                        }
                    ),
                ),
            ]
        )
    else:
        messages.append(
            ChatMessage(
                role="user",
                content=prompt("continue")
                + "\n"
                + json.dumps(
                    {
                        "hint": s["hint"],
                        "rejected": [v.model_dump() for v in s["verdicts"].values() if not v.keep],
                    }
                ),
            )
        )
    specs = get_tool_specs(cfg)
    while s["stats"].llm_calls < cfg.max_llm_calls:
        choice = "none" if s["stats"].tool_calls >= cfg.max_tool_calls else "auto"
        reply = session.chat(s, "explore", messages, specs, choice)
        messages.append(reply.message)
        if reply.tool_calls:
            if choice == "none":
                raise OutOfBudget("Model requested tools after the tool budget was exhausted")
            for call in reply.tool_calls:
                if s["stats"].tool_calls >= cfg.max_tool_calls:
                    text = "Tool budget exhausted. Produce your final JSON answer."
                    session.event(
                        s,
                        "explore",
                        "tool_denied",
                        tool_call_id=call.id,
                        name=call.function.name,
                        error=text,
                    )
                else:
                    result = session.tool(
                        s, "explore", call.function.name, call.function.arguments, call.id
                    )
                    text = result.text
                messages.append(ChatMessage(role="tool", tool_call_id=call.id, content=text))
            continue
        raw = reply.message.content or ""
        try:
            selection = decode(raw, Selection)
        except ValueError:
            if s["stats"].format_repairs:
                raise
            if s["stats"].llm_calls >= cfg.max_llm_calls:
                raise OutOfBudget("No LLM call remaining for JSON format repair") from None
            s["stats"].format_repairs += 1
            messages.append(ChatMessage(role="user", content=prompt("repair")))
            repaired = session.chat(s, "explore", messages, specs, "none")
            messages.append(repaired.message)
            corrected = repaired.message.content or ""
            selection = decode(corrected, Selection)
            try:
                original = decode(repairable_json(raw), Selection)
            except ValueError:
                raise ValueError("format repair could not establish original content") from None
            if original != selection:
                raise ValueError("format repair changed content") from None
        new_pool = _validate(selection, s, session)
        previous = {c.symbol_id or c.path: c for c in s["pool"]}
        for candidate in new_pool:
            key = candidate.symbol_id or candidate.path
            if previous.pop(key, None) != candidate:
                s["verdicts"].pop(key, None)
        s["pool"] = [*new_pool, *previous.values()]
        s["need_more"] = False
        return
    raise OutOfBudget("LLM call budget reached before a final exploration answer")
