import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import networkx as nx
import numpy as np
from scipy import sparse


def _encode_node(node: Any) -> Any:
    if isinstance(node, np.integer):
        return int(node)
    if isinstance(node, np.floating):
        return float(node)
    if isinstance(node, tuple):
        return {"tuple": [_encode_node(item) for item in node]}
    if isinstance(node, list):
        return {"list": [_encode_node(item) for item in node]}
    if isinstance(node, (str, int, float, bool)) or node is None:
        return node
    raise TypeError(f"unsupported graph node type: {type(node).__name__}")


def _decode_node(value: Any) -> Any:
    if isinstance(value, dict) and "tuple" in value:
        return tuple(_decode_node(item) for item in value["tuple"])
    if isinstance(value, dict) and "list" in value:
        return [_decode_node(item) for item in value["list"]]
    return value


def _node_sort_key(node: Any) -> Tuple[str, str]:
    return type(node).__name__, repr(node)


def validate_input_graph(graph: nx.Graph) -> None:
    if graph.is_directed():
        raise TypeError(
            "the clustering regularity algorithm requires an undirected graph"
        )
    if graph.is_multigraph():
        raise TypeError("the clustering regularity algorithm requires a simple graph")
    if nx.number_of_selfloops(graph):
        raise ValueError("the clustering regularity algorithm does not allow self-loops")


def graph_fingerprint(graph: nx.Graph) -> str:
    """Hash the deterministic oriented adjacency stream without copying it."""
    digest = hashlib.sha256()
    for middle in sorted(graph.nodes(), key=_node_sort_key):
        encoded_middle = json.dumps(
            _encode_node(middle),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(encoded_middle.encode("utf-8"))
        digest.update(b"\0")
        for neighbor in sorted(graph.neighbors(middle), key=_node_sort_key):
            encoded_neighbor = json.dumps(
                _encode_node(neighbor),
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update(encoded_neighbor.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class LinkShard:
    shard_id: int
    middle: Any
    neighbors: Tuple[Any, ...]
    edge_indices: np.ndarray
    closure: sparse.csr_matrix


class GraphIndex:
    """Disk-backed implicit tripartite index grouped by middle vertex."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, root: Path):
        self.root = Path(root)
        with open(self.root / self.MANIFEST_NAME, encoding="utf-8") as handle:
            self.manifest = json.load(handle)

    @classmethod
    def build(cls, graph: nx.Graph, root: Path) -> "GraphIndex":
        validate_input_graph(graph)
        root = Path(root)
        if (root / cls.MANIFEST_NAME).exists():
            raise FileExistsError(f"graph index already exists: {root}")
        root.mkdir(parents=True, exist_ok=True)
        shard_dir = root / "shards"
        shard_dir.mkdir()

        edge_offset = 0
        shard_entries = []
        nodes = sorted(graph.nodes(), key=_node_sort_key)
        for shard_id, middle in enumerate(nodes):
            neighbors = sorted(graph.neighbors(middle), key=_node_sort_key)
            neighbor_index = {
                neighbor: position for position, neighbor in enumerate(neighbors)
            }
            rows: List[int] = []
            columns: List[int] = []
            for left, right in graph.subgraph(neighbors).edges():
                left_index = neighbor_index[left]
                right_index = neighbor_index[right]
                rows.append(left_index)
                columns.append(right_index)
                if left_index != right_index:
                    rows.append(right_index)
                    columns.append(left_index)
            closure = sparse.csr_matrix(
                (
                    np.ones(len(rows), dtype=np.uint8),
                    (np.asarray(rows, dtype=int), np.asarray(columns, dtype=int)),
                ),
                shape=(len(neighbors), len(neighbors)),
            )

            stem = f"{shard_id:08d}"
            sparse.save_npz(shard_dir / f"{stem}.npz", closure, compressed=True)
            metadata = {
                "shard_id": shard_id,
                "middle": _encode_node(middle),
                "neighbors": [_encode_node(node) for node in neighbors],
                "edge_start": edge_offset,
                "edge_stop": edge_offset + len(neighbors),
            }
            with open(shard_dir / f"{stem}.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, separators=(",", ":"))
            shard_entries.append(
                {
                    "shard_id": shard_id,
                    "middle": metadata["middle"],
                    "degree": len(neighbors),
                }
            )
            edge_offset += len(neighbors)

        manifest = {
            "schema_version": 1,
            "node_count": graph.number_of_nodes(),
            "source_edge_count": graph.number_of_edges(),
            "oriented_edge_count": edge_offset,
            "graph_fingerprint": graph_fingerprint(graph),
            "shards": shard_entries,
        }
        temporary = root / f"{cls.MANIFEST_NAME}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        temporary.replace(root / cls.MANIFEST_NAME)
        return cls(root)

    @property
    def edge_count(self) -> int:
        return int(self.manifest["oriented_edge_count"])

    @property
    def node_count(self) -> int:
        return int(self.manifest["node_count"])

    def iter_shard_ids(self) -> Iterator[int]:
        for entry in self.manifest["shards"]:
            yield int(entry["shard_id"])

    def load_shard(self, shard_id: int) -> LinkShard:
        stem = f"{shard_id:08d}"
        shard_dir = self.root / "shards"
        with open(shard_dir / f"{stem}.json", encoding="utf-8") as handle:
            metadata = json.load(handle)
        start = int(metadata["edge_start"])
        stop = int(metadata["edge_stop"])
        return LinkShard(
            shard_id=shard_id,
            middle=_decode_node(metadata["middle"]),
            neighbors=tuple(_decode_node(node) for node in metadata["neighbors"]),
            edge_indices=np.arange(start, stop, dtype=np.int64),
            closure=sparse.load_npz(shard_dir / f"{stem}.npz").tocsr(),
        )

    def iter_edge_records(self, side: str) -> Iterator[Tuple[int, Tuple]]:
        normalized = side.upper()
        if normalized not in {"A", "B"}:
            raise ValueError("side must be 'A' or 'B'")
        for shard_id in self.iter_shard_ids():
            shard = self.load_shard(shard_id)
            for edge_index, neighbor in zip(shard.edge_indices, shard.neighbors):
                if normalized == "A":
                    edge = ((neighbor, 0), (shard.middle, 1))
                else:
                    edge = ((shard.middle, 1), (neighbor, 2))
                yield int(edge_index), edge
