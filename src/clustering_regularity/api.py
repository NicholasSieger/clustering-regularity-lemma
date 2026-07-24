from pathlib import Path
from typing import Optional

import networkx as nx

from .config import PaperParameters, RunConfig
from .core.models import RunResult
from .engine import RefinementEngine
from .graph.index import GraphIndex, graph_fingerprint, validate_input_graph
from .observability.observers import NullObserver, Observer
from .storage.local import LocalPartitionStore, Workspace


def run_graph(
    graph: nx.Graph,
    workspace: Path,
    *,
    epsilon: float = 0.05,
    max_rounds: Optional[int] = None,
    observer: Observer = None,
) -> RunResult:
    """Run the paper algorithm in a resumable local workspace."""
    validate_input_graph(graph)
    workspace = Workspace.create(workspace)
    manifest = workspace.index_dir / GraphIndex.MANIFEST_NAME
    if manifest.exists():
        graph_index = GraphIndex(workspace.index_dir)
        if (
            graph_index.node_count != graph.number_of_nodes()
            or int(graph_index.manifest["source_edge_count"])
            != graph.number_of_edges()
            or graph_index.manifest["graph_fingerprint"]
            != graph_fingerprint(graph)
        ):
            raise ValueError("workspace graph index does not match the input graph")
    else:
        graph_index = GraphIndex.build(graph, workspace.index_dir)

    partitions = LocalPartitionStore(
        workspace.generations_dir,
        graph_index.edge_count,
    )
    config = RunConfig(
        parameters=PaperParameters(epsilon),
        max_rounds=max_rounds,
    )
    engine = RefinementEngine(
        graph_index,
        partitions,
        config,
        observer or NullObserver(),
    )
    return engine.run(workspace.root)
