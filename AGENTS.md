# RepoScout

Given a GitHub issue and a Python repository at a specific commit, RepoScout returns a
ranked list of files and functions that must change to resolve the issue, with cited
line ranges. Evaluated on SWE-bench Lite (localization only; no test execution).
A teacher LLM agent is distilled into a 3–7B student model.

## Ground rules
- Python 3.12, `uv` for env/deps, `ruff` (lint+format), `pytest`, type hints everywhere,
  Pydantic v2 for all data crossing module boundaries.
- Stay inside the directories your workstream owns. Never edit `reposcout/contracts/`.
  If a contract must change, STOP and write `docs/proposals/<short-name>.md`
  (problem, proposed diff, who is affected).
- Do not guess third-party APIs. Check the installed version (`uv pip show <pkg>`) and
  read its source or docs before using it. Pin every dependency in `pyproject.toml`.
- Unit tests make no network calls and use `tests/fixtures/minirepo/`. Mark network or
  slow tests `@pytest.mark.integration`.
- Never truncate silently. Every limit (chars, lines, tokens, calls) is an explicit config
  value, and when it is hit, the output says so.
- LLM calls: no silent retries. Failures are recorded in `RunStats.status/error`; one bad
  instance never crashes a batch run.
- Deterministic where possible: fixed seeds, sorted outputs, stable tie-breaks
  (score desc, then path asc, then start_line asc).
- Small commits with conventional messages (`feat(index): ...`, `test(eval): ...`).
- Definition of done for any task: tests pass, `ruff check` clean, a section in
  your module's `README.md`, and `NOTES.md` in your module listing decisions,
  known limitations and anything you were unsure about.

## Layout and ownership
reposcout/contracts/   models.py, index.py, tools.py, localizer.py, llm.py   (Prompt 0 / human)
reposcout/index/       Workstream B
reposcout/tools/       Workstream C
reposcout/agent/       Workstream D
reposcout/llm/         Workstream D
reposcout/eval/        Workstream A
reposcout/baselines/   Workstream A
reposcout/distill/     Workstream E
reposcout/cli.py, demo/, README.md, Makefile   Workstream F
tests/<module>/        owned by the module's workstream
results/<run_name>/    produced by eval runner; committed (small JSON/MD only)

## Contracts (summary; source of truth is reposcout/contracts/)
- Span(path, start_line, end_line): repo-relative POSIX path, 1-indexed, inclusive.
- Symbol(id="<path>::<qualname>", kind, name, qualname, span, parent_id, is_test).
  qualname: dotted nesting, e.g. `QuerySet.filter`, `outer.inner`.
- SearchHit(symbol_id|None, span, score, source, snippet)
- Candidate(path, symbol_id|None, score, rationale, citations: list[Span])
- Localization(instance_id, files: ranked list[str], functions: ranked list[symbol_id],
  candidates, stats: RunStats)
- RunStats(llm_calls, tool_calls, prompt_tokens, completion_tokens, latency_s, model,
  status: complete|budget_exhausted|error, error)
- Instance(instance_id, repo, base_commit, problem_statement, gold_files, gold_functions)
- RepoIndex protocol: symbols(), get_symbol(), callers(), callees(), bm25(), dense(),
  read(span), list_dir(); built by build_index(repo_root, commit, cache_dir, dense=True).
- Localizer protocol: localize(instance, index, config) -> Localization.
  Every baseline and the agent implement it, so the eval runner treats them identically.
- ChatClient protocol (OpenAI-compatible): chat(messages, tools, model, max_tokens,
  timeout) -> ChatResult(message, tool_calls, usage, finish_reason).

## Metrics (source of truth: reposcout/eval/metrics.py)
- file_hit@k: 1 if ANY gold file is in the top-k files.
- file_all@k: 1 if ALL gold files are in the top-k files.
- func_recall@k: |gold_functions ∩ top-k functions| / |gold_functions|; instances with no
  gold functions (module-level-only edits) are excluded and the count is reported.
- MRR: reciprocal rank of the first gold file (0 if absent).
- Cost/latency: mean and median of tokens, llm_calls, tool_calls, latency_s; error rate;
  budget-exhausted rate; JSON-format-failure rate.
k ∈ {1, 3, 5, 10}. Report both test (n=300) and dev (n=23); tune only on dev.
