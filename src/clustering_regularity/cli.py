import argparse
import json
from pathlib import Path

import networkx as nx

from .api import run_graph
from .observability.observers import (
    CompositeObserver,
    ConsoleObserver,
    JsonlObserver,
)
from .verification.models import VerificationConfig
from .verification.runner import verify_workspace


def _read_graph(path: Path, node_type: str) -> nx.Graph:
    nodetype = int if node_type == "int" else str
    graph = nx.read_edgelist(path, nodetype=nodetype, data=False)
    return nx.Graph(graph)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clustering-regularity",
        description="Run the file-backed clustering regularity algorithm.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser("run", help="refine an edge-list graph")
    run_parser.add_argument("graph", type=Path)
    run_parser.add_argument("--workspace", type=Path, required=True)
    run_parser.add_argument("--epsilon", type=float, default=0.05)
    run_parser.add_argument("--max-rounds", type=int)
    run_parser.add_argument(
        "--node-type",
        choices=("int", "str"),
        default="int",
    )
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument("--events-jsonl", action="store_true")
    verify_parser = subcommands.add_parser(
        "verify",
        help="verify a completed workspace",
    )
    verify_parser.add_argument("graph", type=Path)
    verify_parser.add_argument("--workspace", type=Path, required=True)
    verify_parser.add_argument("--epsilon", type=float, default=0.05)
    verify_parser.add_argument(
        "--node-type",
        choices=("int", "str"),
        default="int",
    )
    verify_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "run":
        graph = _read_graph(args.graph, args.node_type)
        observers = []
        if not args.quiet:
            observers.append(ConsoleObserver())
        if args.events_jsonl:
            observers.append(
                JsonlObserver(Path(args.workspace) / "logs" / "events.jsonl")
            )
        result = run_graph(
            graph,
            args.workspace,
            epsilon=args.epsilon,
            max_rounds=args.max_rounds,
            observer=CompositeObserver(observers),
        )
        if args.quiet:
            print(Path(result.workspace) / "result.json")
    elif args.command == "verify":
        graph = _read_graph(args.graph, args.node_type)
        report = verify_workspace(
            graph,
            args.workspace,
            VerificationConfig(eps=args.epsilon),
        )
        output = args.output or Path(args.workspace) / "verification.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        temporary.replace(output)
        print(output)


if __name__ == "__main__":
    main()
