import math
from typing import Dict, Iterable, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .models import CellData, CellGroup, VerificationConfig


def _iter_nonzero_cells(
    graph: nx.Graph,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
):
    expected_edge_count = sum(dict(graph.degree()).values())
    if (
        len(labels_A) != expected_edge_count
        or len(labels_B) != expected_edge_count
    ):
        raise ValueError("partition label lengths do not match tripartite edge sets")
    labels_A = np.asarray(labels_A, dtype=int)
    labels_B = np.asarray(labels_B, dtype=int)
    node_order = sorted(
        graph.nodes(),
        key=lambda node: (type(node).__name__, repr(node)),
    )
    observed_cells = set()
    offset = 0
    for middle in node_order:
        degree = graph.degree(middle)
        edge_indices = np.arange(offset, offset + degree, dtype=int)
        for label_A in np.unique(labels_A[edge_indices]):
            for label_B in np.unique(labels_B[edge_indices]):
                observed_cells.add((int(label_A), int(label_B)))
        offset += degree

    for label_A, label_B in sorted(observed_cells):
        groups = []
        offset = 0
        for middle in node_order:
            neighbors = sorted(
                graph.neighbors(middle),
                key=lambda node: (type(node).__name__, repr(node)),
            )
            degree = len(neighbors)
            edge_indices = np.arange(offset, offset + degree, dtype=int)
            positions_A = np.flatnonzero(labels_A[edge_indices] == label_A)
            positions_B = np.flatnonzero(labels_B[edge_indices] == label_B)
            offset += degree
            if positions_A.size == 0 or positions_B.size == 0:
                continue
            a_nodes = np.asarray(
                [neighbors[int(position)] for position in positions_A],
                dtype=object,
            )
            b_nodes = np.asarray(
                [neighbors[int(position)] for position in positions_B],
                dtype=object,
            )
            closure = np.fromiter(
                (
                    1.0 if graph.has_edge(a_node, b_node) else 0.0
                    for a_node in a_nodes
                    for b_node in b_nodes
                ),
                dtype=float,
                count=len(a_nodes) * len(b_nodes),
            ).reshape((len(a_nodes), len(b_nodes)))
            groups.append(
                CellGroup(
                    middle=middle,
                    a_edge_indices=edge_indices[positions_A],
                    b_edge_indices=edge_indices[positions_B],
                    a_nodes=a_nodes,
                    b_nodes=b_nodes,
                    closure=closure,
                )
            )
        yield CellData(label_A, label_B, groups)


def make_cell_group(
    closure: np.ndarray,
    middle: int = 0,
    a_offset: int = 0,
    b_offset: int = 0,
) -> CellGroup:
    """Build a synthetic cell group for focused verifier tests."""
    closure = np.asarray(closure, dtype=float)
    if closure.ndim != 2:
        raise ValueError("closure must be a two-dimensional matrix")
    rows, columns = closure.shape
    return CellGroup(
        middle=middle,
        a_edge_indices=np.arange(a_offset, a_offset + rows, dtype=int),
        b_edge_indices=np.arange(b_offset, b_offset + columns, dtype=int),
        a_nodes=np.arange(rows, dtype=int),
        b_nodes=np.arange(columns, dtype=int),
        closure=closure,
    )


def _cell_counts(cell: CellData) -> Tuple[int, int, float]:
    pathweight = sum(
        group.closure.shape[0] * group.closure.shape[1]
        for group in cell.groups
    )
    triangle_count = int(sum(group.closure.sum() for group in cell.groups))
    gamma = 0.0 if pathweight == 0 else triangle_count / pathweight
    return pathweight, triangle_count, gamma


def finite_scale_threshold_report(
    graph: nx.Graph,
    config: VerificationConfig,
) -> Dict:
    """Report when paper thresholds exceed one count on the finite input."""
    thresholds = config.thresholds
    degrees = [int(degree) for _, degree in graph.degree()]
    max_degree = max(degrees, default=0)
    total_pathweight = sum(degree**2 for degree in degrees)
    scales = {
        "total_pathweight": total_pathweight,
        "max_degree": max_degree,
        "delta_2_total_pathweight": thresholds["delta_2"] * total_pathweight,
        "delta_5_total_pathweight": thresholds["delta_5"] * total_pathweight,
        "max_delta_1_local_deviation_threshold": (
            thresholds["delta_1"] * max_degree**3
        ),
        "max_delta_3_local_degree_tolerance": (
            thresholds["delta_3"] * max_degree
        ),
        "max_delta_4_irregular_edge_count_threshold": (
            thresholds["delta_4"] * max_degree
        ),
    }
    scales["count_scale_checks"] = {
        "global_deviation": scales["delta_2_total_pathweight"] >= 1,
        "global_triangle_irregularity": scales["delta_5_total_pathweight"] >= 1,
        "local_deviation": scales["max_delta_1_local_deviation_threshold"] >= 1,
        "local_triangle_degree_tolerance": (
            scales["max_delta_3_local_degree_tolerance"] >= 1
        ),
        "local_triangle_irregular_edge_count": (
            scales["max_delta_4_irregular_edge_count_threshold"] >= 1
        ),
    }
    return scales


def _selection_counts(
    cell: CellData,
    selected_A: Set[int],
    selected_B: Set[int],
) -> Tuple[int, int]:
    pathweight = 0
    triangle_count = 0
    for group in cell.groups:
        mask_A = np.asarray(
            [int(index) in selected_A for index in group.a_edge_indices],
            dtype=bool,
        )
        mask_B = np.asarray(
            [int(index) in selected_B for index in group.b_edge_indices],
            dtype=bool,
        )
        count_A = int(mask_A.sum())
        count_B = int(mask_B.sum())
        pathweight += count_A * count_B
        if count_A and count_B:
            triangle_count += int(group.closure[np.ix_(mask_A, mask_B)].sum())
    return pathweight, triangle_count


def _witness_report(
    cell: CellData,
    selected_A: Iterable[int],
    selected_B: Iterable[int],
    parent_pathweight: int,
    parent_gamma: float,
    eps: float,
    source: str,
) -> Optional[Dict]:
    selected_A = set(int(index) for index in selected_A)
    selected_B = set(int(index) for index in selected_B)
    pathweight, triangle_count = _selection_counts(cell, selected_A, selected_B)
    if pathweight == 0 or pathweight + 1e-12 < eps * parent_pathweight:
        return None

    gamma = triangle_count / pathweight
    deviation = abs(gamma - parent_gamma)
    if deviation <= eps:
        return None
    return {
        "source": source,
        "edge_indices_A": sorted(selected_A),
        "edge_indices_B": sorted(selected_B),
        "pathweight": pathweight,
        "minimum_pathweight": eps * parent_pathweight,
        "triangle_count": triangle_count,
        "gamma": gamma,
        "parent_gamma": parent_gamma,
        "absolute_gamma_difference": deviation,
        "epsilon": eps,
    }


def validate_witness(cell: CellData, witness: Dict, eps: float) -> bool:
    parent_pathweight, _, parent_gamma = _cell_counts(cell)
    replayed = _witness_report(
        cell,
        witness["edge_indices_A"],
        witness["edge_indices_B"],
        parent_pathweight,
        parent_gamma,
        eps,
        source=witness.get("source", "replay"),
    )
    if replayed is None:
        return False
    return (
        replayed["pathweight"] == witness["pathweight"]
        and replayed["triangle_count"] == witness["triangle_count"]
        and math.isclose(
            replayed["gamma"],
            witness["gamma"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )


