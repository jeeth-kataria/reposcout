from reposcout.baselines.retrieval import reciprocal_rank_fusion

from ..runtime import Session
from ..state import AgentState


def seed(s: AgentState, session: Session) -> None:
    cfg = s["config"]
    queries = list(dict.fromkeys([*s["queries"], *s["identifiers"], *s["suspected_paths"]]))
    rankings = []
    for query in queries:
        rankings.append(session.index.bm25(query, cfg.retrieval_depth))
        if cfg.use_dense:
            rankings.append(session.index.dense(query, cfg.retrieval_depth))
    hits = reciprocal_rank_fusion(rankings, rank_constant=cfg.rrf_k)
    s["seed_hits"] = hits[: cfg.seed_limit]
    s["seen"].update(h.span.path for h in s["seed_hits"])
    session.event(
        s,
        "seed",
        "retrieval",
        queries=queries,
        available=len(hits),
        retained=len(s["seed_hits"]),
        depth=cfg.retrieval_depth,
        seed_limit=cfg.seed_limit,
        hits=[h.model_dump() for h in s["seed_hits"]],
    )
