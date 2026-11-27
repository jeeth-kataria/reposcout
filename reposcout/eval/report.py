import json
from pathlib import Path

COLUMNS = (
    "file_hit@1",
    "file_hit@3",
    "file_hit@5",
    "file_all@5",
    "func_recall@5",
    "func_recall@10",
    "MRR",
    "mean_tokens",
    "median_latency_s",
    "error_rate",
)


def report(results: Path = Path("results")) -> str:
    rows = [
        "| Run | Split | n | Scored | " + " | ".join((*COLUMNS[:-1], "error %")) + " |",
        "|" + "---|" * (4 + len(COLUMNS)),
    ]
    for path in sorted(results.glob("*/metrics.json")):
        config = json.loads(path.with_name("config.json").read_text())
        metrics = json.loads(path.read_text())
        cells = []
        for k in COLUMNS:
            value = metrics[k]
            if value is None:
                cells.append("—")
            else:
                cells.append(f"{value * 100:.1f}" if k == "error_rate" else f"{value:.3f}")
        rows.append(
            "| "
            + " | ".join(
                [
                    path.parent.name,
                    config["split"],
                    str(metrics["instances"]),
                    str(
                        metrics.get(
                            "scored_instances",
                            metrics["instances"] - metrics.get("unavailable_gold", 0),
                        )
                    ),
                    *cells,
                ]
            )
            + " |"
        )
    markdown = "\n".join(rows) + "\n"
    results.mkdir(parents=True, exist_ok=True)
    (results / "LEADERBOARD.md").write_text(markdown)
    return markdown
