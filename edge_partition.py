from typing import List, Tuple

import numpy as np


class EdgePartitionAssembler:
    """Maps local vertex partitions onto global E12 and E23 edge masks."""

    def __init__(self, graph_manager):
        self.graph_manager = graph_manager

    def get_edge_lists(self) -> Tuple[List[Tuple], List[Tuple]]:
        """Get all E12 and E23 edges from the tripartite graph, sorted consistently."""
        H = self.graph_manager.H
        E12_edges = []
        E23_edges = []

        for u, v in H.edges():
            u_part = H.nodes[u]["part"]
            v_part = H.nodes[v]["part"]

            if u_part > v_part:
                u, v, u_part, v_part = v, u, v_part, u_part

            if u_part == 0 and v_part == 1:
                E12_edges.append((u, v))
            elif u_part == 1 and v_part == 2:
                E23_edges.append((u, v))

        E12_edges.sort()
        E23_edges.sort()
        return E12_edges, E23_edges

    def map_vertex_partition_to_edges(
        self,
        v: int,
        mask_A: np.ndarray,
        mask_B: np.ndarray,
        neighbors_A: List,
        neighbors_B: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map one vertex's local partition to global E12 and E23 edge masks."""
        E12_edges, E23_edges = self.get_edge_lists()
        E12_set = {e: i for i, e in enumerate(E12_edges)}
        E23_set = {e: i for i, e in enumerate(E23_edges)}

        E12_contrib = np.zeros(len(E12_edges), dtype=bool)
        E23_contrib = np.zeros(len(E23_edges), dtype=bool)
        v_node = (v, 1)

        for idx, neighbor_A in enumerate(neighbors_A):
            if mask_A[idx]:
                edge = (neighbor_A, v_node)
                if edge in E12_set:
                    E12_contrib[E12_set[edge]] = True

        for idx, neighbor_B in enumerate(neighbors_B):
            if mask_B[idx]:
                edge = (v_node, neighbor_B)
                if edge in E23_set:
                    E23_contrib[E23_set[edge]] = True

        return E12_contrib, E23_contrib

    def assemble_partition(self, vertex_ids: List[int], partition_store, direction: str) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble one direction's global E12 and E23 edge masks."""
        E12_edges, E23_edges = self.get_edge_lists()
        E12_partition = np.zeros(len(E12_edges), dtype=bool)
        E23_partition = np.zeros(len(E23_edges), dtype=bool)

        for v in vertex_ids:
            success, record = partition_store.loadPartitionRecord(v, direction)
            if not success:
                continue

            E12_contrib, E23_contrib = self.map_vertex_partition_to_edges(
                v,
                record.mask_A,
                record.mask_B,
                record.neighbors_A,
                record.neighbors_B,
            )
            E12_partition |= E12_contrib
            E23_partition |= E23_contrib

        return E12_partition, E23_partition


def partition_labels(bitmasks: List[np.ndarray]) -> np.ndarray:
    """Convert a disjoint, exhaustive family of bitmasks into labels."""
    if not bitmasks:
        return np.array([], dtype=int)

    masks = [np.asarray(bitmask, dtype=bool) for bitmask in bitmasks]
    if any(mask.ndim != 1 for mask in masks):
        raise ValueError("partition masks must be one-dimensional")

    num_indices = masks[0].shape[0]
    if any(mask.shape[0] != num_indices for mask in masks):
        raise ValueError("partition masks must have the same length")

    stacked = np.stack(masks)
    membership = np.sum(stacked, axis=0)
    uncovered = int(np.sum(membership == 0))
    overlapping = int(np.sum(membership > 1))
    if uncovered or overlapping:
        raise ValueError(
            "partition masks must be disjoint and exhaustive "
            f"(uncovered={uncovered}, overlapping={overlapping})"
        )

    return np.argmax(stacked, axis=0).astype(int, copy=False)
