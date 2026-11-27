# Retrieval baselines

Importing `reposcout.baselines` registers `bm25`, `dense`, `hybrid_rrf`, and `hybrid_stacktrace` in the localizer registry. Each takes the issue text and a commit-pinned index, returns a ranked localization with code citations, and consumes zero LLM tokens.

Lexical and dense searches request 50 units per query. The hybrid score adds `1 / (60 + rank)` across the two lists. The traceback variant boosts retrieved units in the cited file and a matching function named in a Python frame. File score equals the maximum score of its retrieval units. Classes and modules contribute to file ranking; the functions list contains only functions and methods.

```sh
uv run python -m reposcout.eval run --localizer hybrid_rrf --split dev --limit 5
```
