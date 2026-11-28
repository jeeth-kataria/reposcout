import json
import re

from reposcout.baselines.retrieval import traceback_frames
from reposcout.contracts.llm import ChatMessage

from ..runtime import Session, clipped, decode, prompt
from ..state import AgentState, Rewrite

_PATH = re.compile(r"[\w./-]+\.py\b")
_IDENT = re.compile(r"\b(?:[a-zA-Z]\w*_\w+|[A-Z][a-z]+[A-Z]\w*|\w+\.\w+)\b")


def rewrite(s: AgentState, session: Session) -> None:
    config = s["config"]
    issue = clipped(s["instance"].problem_statement, config.issue_chars)
    frames = traceback_frames(issue)
    identifiers = list(
        dict.fromkeys([name for _, _, name in frames if name] + _IDENT.findall(issue))
    )
    paths = list(dict.fromkeys([p for p, _, _ in frames] + _PATH.findall(issue)))
    s["queries"] = [issue]
    s["identifiers"] = identifiers[: config.identifier_limit]
    s["suspected_paths"] = paths[: config.path_limit]
    if len(identifiers) > config.identifier_limit or len(paths) > config.path_limit:
        session.event(
            s,
            "rewrite",
            "limit",
            identifiers=len(identifiers),
            paths=len(paths),
            identifier_limit=config.identifier_limit,
            path_limit=config.path_limit,
        )
    inp = {"issue": issue, "identifiers": s["identifiers"], "paths": s["suspected_paths"]}
    reply = session.chat(
        s,
        "rewrite",
        [
            ChatMessage(role="system", content=prompt("rewrite")),
            ChatMessage(role="user", content=json.dumps(inp)),
        ],
        [],
    )
    result = decode(reply.message.content or "", Rewrite)
    for field, limit in (
        ("queries", config.query_limit),
        ("identifiers", config.identifier_limit),
        ("suspected_paths", config.path_limit),
    ):
        values = list(
            dict.fromkeys([*s[field], *(v.strip() for v in getattr(result, field) if v.strip())])
        )
        s[field] = values[:limit]
        if len(values) > limit:
            session.event(s, "rewrite", "limit", field=field, available=len(values), retained=limit)
