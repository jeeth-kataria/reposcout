# Evaluation

From the project root, run a retrieval baseline on a subset or compare saved runs.

```sh
uv run python -m reposcout.eval run --localizer bm25 --split dev --limit 5
uv run python -m reposcout.eval report
```

The loader caches the official 23 dev and 300 test rows in `data/{split}.jsonl`. Rows contain `instance_id`, `repo`, `base_commit`, `problem_statement` and `patch`. This is also the offline import format for the official Parquet files. The localizer receives only the issue, repository and commit. [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) · [Dataset Viewer rows API](https://huggingface.co/docs/dataset-viewer/rows)

Each repository has one bare Git clone under `cache/repos`. Detached instance worktrees under `cache/worktrees` persist unless `--clean-worktrees` is supplied. The report computes file hits, all-file coverage, function recall and MRR from pre-image labels extracted from the implementation patch. Functions are scored only when gold functions exist.

## Running and resuming

Choose `--run-name` for a durable comparison. Repeat the same split, instances and localizer settings to resume. Use `--instance-ids` to select specific issues, `--workers` for parallelism, and `--timeout` for the per-instance wall-clock limit in seconds. Setup and indexing count toward that limit. Benchmarked repository tests are never executed.

Each result directory includes `config.json`, `predictions.jsonl`, `gold.jsonl`, `errors.jsonl` and `metrics.json`. Regenerate `results/LEADERBOARD.md` with the report command. Counters identify cases that failed before the gold labels could be read. Every attempted case contributes to the error rate.

## Model budgets

Set `--request-interval` in seconds between provider request starts. An LLM adapter calls `reposcout.eval.budget.before_request()` immediately before each provider invocation. `REPOSCOUT_RUN_NAME` identifies its trace directory.

`--max-total-tokens` stops the scheduler once completed predictions meet or exceed the cap. Already running instances finish and can overshoot. Increase the cap to resume. `--estimate` processes five selected instances (or the available selection), writes `estimate.json`, and prints projected full-split token use, localization time and USD cost. Supply rates with `--input-usd-per-million` and `--output-usd-per-million`; both default to zero for free plans. Interactive continuation requires confirmation; use `--yes` for an unattended continuation.
