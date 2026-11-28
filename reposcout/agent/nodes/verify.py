import json

from reposcout.contracts.llm import ChatMessage

from ..runtime import Session, clipped, decode, prompt
from ..state import AgentState, Review


def verify(s: AgentState, session: Session) -> None:
    cfg = s["config"]
    if s["stats"].llm_calls >= cfg.max_llm_calls:
        from ..runtime import OutOfBudget

        raise OutOfBudget("No LLM call remaining for verification")
    pool = s["pool"][: cfg.verifier_limit]
    inputs = []
    for candidate in pool:
        evidence = []
        for citation in candidate.citations:
            end = min(citation.end_line, citation.start_line + cfg.verifier_lines - 1)
            args = {"path": citation.path, "start_line": citation.start_line, "end_line": end}
            result = session.tool(s, "verify", "read_range", json.dumps(args))
            evidence.append(
                {
                    "text": result.text,
                    "citations": [c.model_dump() for c in result.citations],
                    "truncated": result.truncated or end < citation.end_line,
                    "error": result.error,
                }
            )
        inputs.append({"candidate": candidate.model_dump(exclude={"score"}), "evidence": evidence})
    reply = session.chat(
        s,
        "verify",
        [
            ChatMessage(role="system", content=prompt("verify")),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "issue": clipped(s["instance"].problem_statement, cfg.issue_chars),
                        "candidates": inputs,
                    }
                ),
            ),
        ],
        [],
    )
    review = decode(reply.message.content or "", Review)
    expected = {c.symbol_id or c.path for c in pool}
    verdicts = {}
    for v in review.verdicts:
        if bool(v.symbol_id) == bool(v.path):
            raise ValueError("a verdict needs one symbol_id or path")
        key = v.symbol_id or v.path
        if key not in expected or key in verdicts:
            raise ValueError("unknown or duplicate verifier candidate")
        verdicts[key] = v
    if verdicts.keys() != expected:
        raise ValueError("verifier omitted a candidate")
    s["verdicts"].update(verdicts)
    s["need_more"] = review.need_more
    s["hint"] = review.next_hint
