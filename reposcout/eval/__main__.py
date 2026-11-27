import argparse
import logging
from pathlib import Path

from reposcout.contracts.localizer import LocalizerConfig

from .report import report
from .runner import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reposcout eval")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("--localizer", required=True)
    evaluate.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--instance-ids", nargs="+")
    evaluate.add_argument("--workers", type=int, default=2)
    evaluate.add_argument("--timeout", type=float, default=180)
    evaluate.add_argument("--data-dir", type=Path, default=Path("data"))
    evaluate.add_argument("--cache-dir", type=Path, default=Path("cache"))
    evaluate.add_argument("--results-dir", type=Path, default=Path("results"))
    evaluate.add_argument("--request-interval", type=float, default=0)
    evaluate.add_argument("--max-total-tokens", type=int)
    evaluate.add_argument("--estimate", action="store_true")
    evaluate.add_argument("--yes", action="store_true")
    evaluate.add_argument("--clean-worktrees", action="store_true")
    evaluate.add_argument("--input-usd-per-million", type=float, default=0)
    evaluate.add_argument("--output-usd-per-million", type=float, default=0)
    table = commands.add_parser("report")
    table.add_argument("--results-dir", type=Path, default=Path("results"))
    args = vars(parser.parse_args(argv))
    logging.basicConfig(level=logging.INFO)
    if args.pop("command") == "report":
        print(report(args["results_dir"]), end="")
        return
    localizer = args.pop("localizer")
    if args["run_name"] is None:
        args["run_name"] = f"{localizer}-{args['split']}" + (
            f"-{args['limit']}" if args["limit"] is not None else ""
        )
    config = LocalizerConfig(timeout_s=args.pop("timeout"), use_dense=localizer != "bm25")
    metrics = run(localizer, config=config, **args)
    print(f"Scored {metrics['instances']}/{metrics['target_instances']} instances.")


if __name__ == "__main__":
    main()
