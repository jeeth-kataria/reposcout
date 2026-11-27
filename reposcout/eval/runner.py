import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from time import monotonic

from reposcout.contracts.localizer import LocalizerConfig
from reposcout.contracts.models import Instance, Localization

from .dataset import DATASET, Split, load_split
from .metrics import Metric, summarize
from .repos import locked
from .worker import Job, failure, resolve_localizer


def _atomic(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
    Path(handle.name).replace(path)


def _job(
    job: Job, request_interval: float, run_dir: Path, run_name: str
) -> tuple[Instance | None, Localization]:
    start = monotonic()
    with tempfile.TemporaryDirectory(dir=job.out) as tmp:
        job.out = Path(tmp)
        proc = subprocess.Popen(
            [sys.executable, "-m", "reposcout.eval.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
            env={
                **os.environ,
                "REPOSCOUT_RUN_NAME": run_name,
                "REPOSCOUT_REQUEST_CLOCK": str((run_dir / ".request-clock").resolve()),
                "REPOSCOUT_REQUEST_INTERVAL": str(request_interval),
            },
        )
        try:
            proc.communicate(job.model_dump_json(), timeout=job.config.timeout_s)
            if proc.returncode:
                raise RuntimeError(f"worker exited with status {proc.returncode}")
            prediction = Localization.model_validate_json((job.out / "prediction.json").read_text())
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            prediction = failure(
                job.row.instance_id,
                f"instance timeout after {job.config.timeout_s}s",
                monotonic() - start,
            )
        gold_path = job.out / "gold.json"
        return (
            Instance.model_validate_json(gold_path.read_text()) if gold_path.exists() else None,
            prediction,
        )


def run(
    localizer_name: str,
    split: Split,
    run_name: str,
    limit: int | None = None,
    workers: int = 2,
    instance_ids: list[str] | None = None,
    *,
    data_dir: Path = Path("data"),
    cache_dir: Path = Path("cache"),
    results_dir: Path = Path("results"),
    request_interval: float = 0,
    max_total_tokens: int | None = None,
    estimate: bool = False,
    yes: bool = False,
    config: LocalizerConfig | None = None,
    clean_worktrees: bool = False,
    input_usd_per_million: float = 0,
    output_usd_per_million: float = 0,
) -> dict[str, Metric]:
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", run_name):
        raise ValueError("invalid run name")
    if (
        workers < 1
        or (limit is not None and limit < 0)
        or (max_total_tokens is not None and max_total_tokens < 0)
    ):
        raise ValueError("workers must be positive and limits nonnegative")
    if any(
        not math.isfinite(x) or x < 0
        for x in (request_interval, input_usd_per_million, output_usd_per_million)
    ):
        raise ValueError("rate and price settings must be finite and nonnegative")
    config = config or LocalizerConfig(use_dense=localizer_name != "bm25")
    if localizer_name == "bm25":
        config = config.model_copy(update={"use_dense": False})
    resolve_localizer(localizer_name)
    all_rows = load_split(split, data_dir)
    by_id = {r.instance_id: r for r in all_rows}
    ids = list(dict.fromkeys(instance_ids)) if instance_ids is not None else list(by_id)
    if missing := set(ids) - by_id.keys():
        raise ValueError(f"unknown instance IDs: {sorted(missing)}")
    selected = ids[:limit]
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Altering the localizer, scored population or dataset invalidates comparisons.
    identity = {
        "dataset": DATASET,
        "split": split,
        "localizer": localizer_name,
        "config": config.model_dump(),
        "instance_ids": selected,
        "rows_sha256": hashlib.sha256(
            "\n".join(by_id[i].model_dump_json() for i in selected).encode()
        ).hexdigest(),
    }
    identity["embedding_model"] = (
        os.getenv("REPOSCOUT_EMBEDDING_MODEL") if config.use_dense else None
    )
    if localizer_name.startswith("agent"):
        identity["model"] = os.environ.get(
            "STUDENT_MODEL" if localizer_name == "agent_student" else "TEACHER_MODEL"
        )
    with locked(
        cache_dir
        / "locks"
        / f"run-{hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()}.lock"
    ):
        cfgfile = run_dir / "config.json"
        if cfgfile.exists() and any(
            json.loads(cfgfile.read_text()).get(k) != v for k, v in identity.items()
        ):
            raise ValueError("resume config differs from existing run; use a new run name")
        _atomic(
            cfgfile,
            json.dumps(
                {
                    **identity,
                    "workers": workers,
                    "max_total_tokens": max_total_tokens,
                    "request_interval": request_interval,
                    "input_usd_per_million": input_usd_per_million,
                    "output_usd_per_million": output_usd_per_million,
                },
                indent=2,
            )
            + "\n",
        )
        predictions_path, gold_path = run_dir / "predictions.jsonl", run_dir / "gold.jsonl"
        predictions = {
            p.instance_id: p
            for line in (
                predictions_path.read_text().splitlines() if predictions_path.exists() else []
            )
            if (p := Localization.model_validate_json(line))
        }
        gold = {
            i.instance_id: i
            for line in (gold_path.read_text().splitlines() if gold_path.exists() else [])
            if (i := Instance.model_validate_json(line))
        }

        def flush() -> dict[str, Metric]:
            ordered = [predictions[i] for i in selected if i in predictions]
            _atomic(
                gold_path, "".join(gold[i].model_dump_json() + "\n" for i in selected if i in gold)
            )
            _atomic(predictions_path, "".join(p.model_dump_json() + "\n" for p in ordered))
            _atomic(
                run_dir / "errors.jsonl",
                "".join(p.model_dump_json() + "\n" for p in ordered if p.stats.status == "error"),
            )
            metrics = summarize([(gold.get(p.instance_id), p) for p in ordered])
            metrics["target_instances"] = len(selected)
            _atomic(run_dir / "metrics.json", json.dumps(metrics, indent=2) + "\n")
            return metrics

        def spent() -> int:
            return sum(
                v.stats.prompt_tokens + v.stats.completion_tokens for v in predictions.values()
            )

        def batch(todo: Iterator[str]) -> None:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                active = {}
                exhausted = False
                while active or not exhausted:
                    while len(active) < workers and not exhausted:
                        if max_total_tokens is not None and spent() >= max_total_tokens:
                            exhausted = True
                            break
                        key = next(todo, None)
                        if key is None:
                            exhausted = True
                            break
                        job = Job(
                            row=by_id[key],
                            localizer=localizer_name,
                            cache=cache_dir.resolve(),
                            out=cache_dir.resolve(),
                            config=config,
                            clean_worktrees=clean_worktrees,
                        )
                        active[pool.submit(_job, job, request_interval, run_dir, run_name)] = key
                    if not active:
                        break
                    for future in wait(active, return_when=FIRST_COMPLETED).done:
                        key = active.pop(future)
                        try:
                            labels, pred = future.result()
                        except Exception as exc:
                            labels, pred = None, failure(key, f"{type(exc).__name__}: {exc}", 0)
                        if labels:
                            gold[key] = labels
                        predictions[key] = pred
                        flush()

        started = monotonic()
        if estimate:
            pilot = selected[:5]
            batch(iter(i for i in pilot if i not in predictions))
            if pilot and all(i in predictions for i in pilot):
                stats = [
                    predictions[i].stats for i in pilot if predictions[i].stats.status != "error"
                ]
                if not stats:
                    estimate_data = {
                        "pilot_instances": len(pilot),
                        "pilot_errors": len(pilot),
                        "split_instances": len(all_rows),
                        "status": "unavailable",
                        "reason": "every pilot prediction failed",
                    }
                    _atomic(run_dir / "estimate.json", json.dumps(estimate_data, indent=2) + "\n")
                    print(json.dumps(estimate_data, indent=2), flush=True)
                    return flush()
                scale = len(all_rows) / len(stats)
                input_tokens = sum(s.prompt_tokens for s in stats) * scale
                output_tokens = sum(s.completion_tokens for s in stats) * scale
                usd = (
                    input_tokens * input_usd_per_million + output_tokens * output_usd_per_million
                ) / 1_000_000
                estimate_data = {
                    "pilot_instances": len(pilot),
                    "split_instances": len(all_rows),
                    "projected_tokens": round(input_tokens + output_tokens),
                    "projected_localization_seconds": sum(s.latency_s for s in stats)
                    * scale
                    / workers,
                    "projected_usd": usd,
                    "pilot_wall_seconds_this_invocation": monotonic() - started,
                    "pilot_errors": len(pilot) - len(stats),
                }
                _atomic(run_dir / "estimate.json", json.dumps(estimate_data, indent=2) + "\n")
                print(json.dumps(estimate_data, indent=2), flush=True)
                if any(i not in predictions for i in selected) and not yes:
                    if not sys.stdin.isatty() or input("Continue full run? [y/N] ").lower() not in {
                        "y",
                        "yes",
                    }:
                        return flush()
        batch(iter(i for i in selected if i not in predictions))
        if (
            max_total_tokens is not None
            and spent() >= max_total_tokens
            and len(predictions) < len(selected)
        ):
            print(
                f"Run stopped at {spent()} tokens (limit {max_total_tokens}). "
                "Increase the limit to resume."
            )
        return flush()
