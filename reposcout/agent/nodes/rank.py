from time import monotonic

from reposcout.contracts.models import Candidate, Localization

from ..runtime import Session
from ..state import AgentState


def rank(s: AgentState, session: Session) -> None:
    cfg = s["config"]
    pool: dict[str, Candidate] = {}
    seed_scores = {}
    best = max((hit.score for hit in s["seed_hits"]), default=1) or 1
    for hit in s["seed_hits"]:
        key = hit.symbol_id or hit.span.path
        if key in pool:
            continue
        seed_scores[key] = hit.score / best
        pool[key] = Candidate(
            path=hit.span.path,
            symbol_id=hit.symbol_id,
            score=0.0,
            rationale="Retrieved by reciprocal rank fusion.",
            citations=[hit.span],
        )
    explore_scores = {}
    for pos, candidate in enumerate(s["pool"], 1):
        key = candidate.symbol_id or candidate.path
        pool[key] = candidate
        explore_scores[key] = 1 / pos
    scored = []
    for key, candidate in pool.items():
        verdict = s["verdicts"].get(key)
        keep = (1 if verdict.keep else -1) if verdict is not None else 0
        score = (
            cfg.seed_weight * seed_scores.get(key, 0)
            + cfg.explore_weight * explore_scores.get(key, 0)
            + cfg.verifier_weight * keep
        )
        scored.append(candidate.model_copy(update={"score": score}))
    scored.sort(key=lambda c: (-c.score, c.path, c.citations[0].start_line, c.symbol_id or ""))
    paths = list(dict.fromkeys(c.path for c in scored))[: cfg.top_k_files]
    functions = []
    for c in scored:
        symbol = session.index.get_symbol(c.symbol_id) if c.symbol_id else None
        if symbol and symbol.kind in {"function", "method"}:
            functions.append(symbol.id)
            if len(functions) == cfg.top_k_functions:
                break
    s["stats"].latency_s = monotonic() - session.started
    s["localization"] = Localization(
        instance_id=s["instance"].instance_id,
        files=paths,
        functions=functions,
        candidates=scored,
        stats=s["stats"],
    )
