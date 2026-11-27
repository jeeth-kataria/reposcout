"""Pure, instance-weighted SWE-bench localization metrics."""

from collections.abc import Sequence
from statistics import fmean, median

from reposcout.contracts.models import Instance, Localization

CUTOFFS = (1, 3, 5, 10)
type Metric = int | float | None


def score(instance: Instance, prediction: Localization) -> dict[str, Metric]:
    if instance.instance_id != prediction.instance_id:
        raise ValueError("prediction and gold instance IDs differ")
    files = list(dict.fromkeys(prediction.files))
    functions = list(dict.fromkeys(prediction.functions))
    gold_files, gold_functions = set(instance.gold_files), set(instance.gold_functions)
    metrics: dict[str, Metric] = {}
    for k in CUTOFFS:
        top = set(files[:k])
        metrics[f"file_hit@{k}"] = int(bool(gold_files & top))
        metrics[f"file_all@{k}"] = int(gold_files <= top)
        metrics[f"func_recall@{k}"] = (
            len(gold_functions.intersection(functions[:k])) / len(gold_functions)
            if gold_functions
            else None
        )
    metrics["MRR"] = next((1 / i for i, path in enumerate(files, 1) if path in gold_files), 0.0)
    return metrics


def summarize(pairs: Sequence[tuple[Instance | None, Localization]]) -> dict[str, Metric]:
    labelled = [(i, p) for i, p in pairs if i is not None]
    rows = [score(instance, prediction) for instance, prediction in labelled]
    stats = [prediction.stats for _, prediction in pairs]
    result: dict[str, Metric] = {
        "instances": len(pairs),
        "scored_instances": len(labelled),
        "unavailable_gold": len(pairs) - len(labelled),
        "no_gold_functions": sum(not case.gold_functions for case, _ in labelled),
        "mean_gold_files": fmean(len(set(i.gold_files)) for i, _ in labelled) if rows else None,
        "mean_gold_functions": fmean(len(set(i.gold_functions)) for i, _ in labelled)
        if rows
        else None,
    }
    for key in [
        f"{metric}@{k}" for metric in ("file_hit", "file_all", "func_recall") for k in CUTOFFS
    ] + ["MRR"]:
        values = [v for row in rows if (v := row[key]) is not None]
        result[key] = fmean(values) if values else None
    quantities = {
        "tokens": [s.prompt_tokens + s.completion_tokens for s in stats],
        "prompt_tokens": [s.prompt_tokens for s in stats],
        "completion_tokens": [s.completion_tokens for s in stats],
        "llm_calls": [s.llm_calls for s in stats],
        "tool_calls": [s.tool_calls for s in stats],
        "latency_s": [s.latency_s for s in stats],
    }
    for key, values in quantities.items():
        result[f"mean_{key}"] = fmean(values) if values else None
        result[f"median_{key}"] = median(values) if values else None
    result["error_rate"] = fmean(s.status == "error" for s in stats) if stats else None
    result["budget_exhausted_rate"] = (
        fmean(s.status == "budget_exhausted" for s in stats) if stats else None
    )
    result["format_failure_rate"] = fmean(s.format_repairs > 0 for s in stats) if stats else None
    return result
