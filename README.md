# RepoScout

RepoScout is a Python issue-localization project. Its output contract ranks repository files and functions and cites inclusive line ranges at a specific Git commit. The Python package is named `reposcout`.

## Contract review

Start with the [contract guide](reposcout/contracts/README.md) and [decisions](reposcout/contracts/NOTES.md). The current scaffold includes the data models, protocol definitions, offline fixture and CI. The [architecture](docs/ARCHITECTURE.md) describes the planned localization pipeline.

## Run locally

Use Python 3.12 and uv.

```sh
uv sync --locked
uv run pytest
uv run ruff check
uv run ruff format --check
```

## Benchmark replay

The [production explorer](https://resumeproject-woad.vercel.app) shows saved rankings and links each citation to the source at the evaluated commit. The [leaderboard](results/LEADERBOARD.md) records the split and sample size for each published run.

To build the replay locally, run `node web/build.mjs`. Serve `web/public` with any static HTTP server.
