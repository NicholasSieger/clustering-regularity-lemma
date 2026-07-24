import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from algorithm import AlgorithmRunner
from edge_partition import EdgePartitionAssembler
from parameters import AlgorithmParameters


GRAPH_FAMILIES = ("sbm", "powerlaw_line", "watts_strogatz")
GRAPH_SIZES = (30, 100, 500)
DENSITIES = (0.05, 0.15, 0.35)
REPORT_PATH = os.path.join("docs", "debug", "clustering_verification_results.json")
REPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class VerificationCase:
    family: str
    size: int
    density: float
    seed: int

    @property
    def case_id(self) -> str:
        density_key = str(self.density).replace(".", "p")
        return f"{self.family}_n{self.size}_d{density_key}_s{self.seed}"


ADDITIONAL_SBM_CASE = VerificationCase("sbm", 48, 0.35, 20260801)
SCALED_ADDITIONAL_CASES = (
    VerificationCase("powerlaw_line", 36, 0.15, 20260901),
    VerificationCase("watts_strogatz", 42, 0.15, 20260902),
    VerificationCase("sbm", 60, 0.15, 20260903),
    VerificationCase("sbm_near_epsilon", 60, 0.06, 20260904),
)


@dataclass(frozen=True)
class VerificationConfig:
    eps: float = 0.05
    exhaustive_edge_limit: int = 18
    svd_dense_limit: int = 256
    density_tolerance: float = 0.02
    numerical_tolerance: float = 1e-12
    witness_search: bool = True
    search_restarts: int = 12
    coordinate_edge_limit: int = 64
    coordinate_passes: int = 8

    def __post_init__(self) -> None:
        if not 0 < self.eps < 1 / 16:
            raise ValueError("paper-faithful verification requires 0 < eps < 1/16")
        if self.exhaustive_edge_limit < 0:
            raise ValueError("exhaustive_edge_limit must be nonnegative")
        if self.svd_dense_limit < 1:
            raise ValueError("svd_dense_limit must be positive")
        if self.density_tolerance < 0:
            raise ValueError("density_tolerance must be nonnegative")

    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "epsilon": self.eps,
            "delta_1": self.eps**2 / 9,
            "delta_2": 2 * self.eps**2 / 5,
            "delta_3": self.eps**5 / 90,
            "delta_4": self.eps ** (5 / 2) / 9,
            "delta_5": 2 * self.eps ** (5 / 2) / 5,
            "delta_6": self.eps**5,
        }


@dataclass
class CellGroup:
    middle: int
    a_edge_indices: np.ndarray
    b_edge_indices: np.ndarray
    a_nodes: np.ndarray
    b_nodes: np.ndarray
    closure: np.ndarray


@dataclass
class CellData:
    label_A: int
    label_B: int
    groups: List[CellGroup]

    @property
    def active_a_indices(self) -> List[int]:
        return sorted(
            {
                int(index)
                for group in self.groups
                for index in group.a_edge_indices.tolist()
            }
        )

    @property
    def active_b_indices(self) -> List[int]:
        return sorted(
            {
                int(index)
                for group in self.groups
                for index in group.b_edge_indices.tolist()
            }
        )


def verification_cases() -> List[VerificationCase]:
    cases = []
    seed = 20260709
    for family in GRAPH_FAMILIES:
        for size in GRAPH_SIZES:
            for density in DENSITIES:
                cases.append(VerificationCase(family, size, density, seed))
                seed += 1
    return cases


def _balanced_block_sizes(size: int, block_count: int = 4) -> List[int]:
    sizes = [size // block_count] * block_count
    for index in range(size % block_count):
        sizes[index] += 1
    return sizes


def _sbm_probabilities(block_sizes: Sequence[int], target_density: float) -> Tuple[float, float]:
    total_pairs = sum(block_sizes) * (sum(block_sizes) - 1) / 2
    within_pairs = sum(block_size * (block_size - 1) / 2 for block_size in block_sizes)
    between_pairs = total_pairs - within_pairs
    target_edges = target_density * total_pairs

    low = 0.0
    high = 1.0
    for _ in range(80):
        inter = (low + high) / 2
        intra = min(1.0, 4 * inter)
        expected_edges = intra * within_pairs + inter * between_pairs
        if expected_edges < target_edges:
            low = inter
        else:
            high = inter
    inter = (low + high) / 2
    return min(1.0, 4 * inter), inter


def generate_sbm_graph(size: int, density: float, seed: int) -> Tuple[nx.Graph, Dict]:
    block_sizes = _balanced_block_sizes(size)
    intra, inter = _sbm_probabilities(block_sizes, density)
    probabilities = [
        [intra if i == j else inter for j in range(len(block_sizes))]
        for i in range(len(block_sizes))
    ]

    candidates = []
    for attempt in range(16):
        candidate_seed = seed + 1009 * attempt
        graph = nx.stochastic_block_model(
            block_sizes,
            probabilities,
            seed=candidate_seed,
        )
        graph = nx.convert_node_labels_to_integers(nx.Graph(graph))
        candidates.append((abs(nx.density(graph) - density), candidate_seed, graph))

    _, selected_seed, graph = min(candidates, key=lambda item: (item[0], item[1]))
    return graph, {
        "model": "stochastic_block_model",
        "block_sizes": block_sizes,
        "intra_probability": intra,
        "inter_probability": inter,
        "candidate_count": len(candidates),
        "selected_model_seed": selected_seed,
    }


def algorithm_clustering_coefficient(graph: nx.Graph) -> float:
    """Return the root gamma used by the tripartite clustering algorithm."""
    pathweight = sum(degree**2 for _, degree in graph.degree())
    if pathweight == 0:
        return 0.0
    closed_paths = 2 * sum(nx.triangles(graph).values())
    return closed_paths / pathweight


def graph_clustering_metrics(graph: nx.Graph, eps: float) -> Dict:
    """Measure the graph coefficients relevant to a verification run."""
    root_gamma = algorithm_clustering_coefficient(graph)
    return {
        "root_gamma": root_gamma,
        "root_gamma_minus_epsilon": root_gamma - eps,
        "average_clustering": nx.average_clustering(graph),
        "transitivity": nx.transitivity(graph),
        "triangle_count": sum(nx.triangles(graph).values()) // 3,
    }


def generate_sbm_near_epsilon_graph(
    size: int,
    density: float,
    seed: int,
    target_clustering: float = 0.05,
    candidate_count: int = 256,
) -> Tuple[nx.Graph, Dict]:
    """Select an SBM whose algorithm root gamma is closest to epsilon."""
    block_sizes = _balanced_block_sizes(size)
    intra, inter = _sbm_probabilities(block_sizes, density)
    probabilities = [
        [intra if i == j else inter for j in range(len(block_sizes))]
        for i in range(len(block_sizes))
    ]

    candidates = []
    for attempt in range(candidate_count):
        candidate_seed = seed + 1009 * attempt
        graph = nx.stochastic_block_model(
            block_sizes,
            probabilities,
            seed=candidate_seed,
        )
        graph = nx.convert_node_labels_to_integers(nx.Graph(graph))
        gamma = algorithm_clustering_coefficient(graph)
        candidates.append(
            (
                abs(gamma - target_clustering),
                abs(nx.density(graph) - density),
                candidate_seed,
                gamma,
                graph,
            )
        )

    (
        clustering_error,
        density_error,
        selected_seed,
        selected_gamma,
        graph,
    ) = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return graph, {
        "model": "stochastic_block_model_near_epsilon",
        "block_sizes": block_sizes,
        "intra_probability": intra,
        "inter_probability": inter,
        "target_root_gamma": target_clustering,
        "selected_root_gamma": selected_gamma,
        "root_gamma_error": clustering_error,
        "selected_density_error": density_error,
        "candidate_count": candidate_count,
        "selected_model_seed": selected_seed,
    }


def _even_degree_from_density(size: int, density: float) -> int:
    valid = list(range(2, size, 2))
    if not valid:
        raise ValueError("Watts-Strogatz generation requires at least three vertices")
    return min(valid, key=lambda value: (abs(value / (size - 1) - density), value))


def generate_watts_strogatz_graph(size: int, density: float, seed: int) -> Tuple[nx.Graph, Dict]:
    k = _even_degree_from_density(size, density)
    graph = nx.watts_strogatz_graph(size, k, 0.1, seed=seed)
    return graph, {
        "model": "watts_strogatz",
        "k": k,
        "rewiring_probability": 0.1,
        "attainable_density": k / (size - 1),
    }


def _powerlaw_degree_sequence(size: int, rank_exponent: float) -> List[int]:
    target_degree_sum = 2 * size
    sequence = np.ones(size, dtype=int)
    remaining = target_degree_sum - size
    weights = np.arange(1, size + 1, dtype=float) ** (-rank_exponent)
    allocation = remaining * weights / weights.sum()
    additions = np.floor(allocation).astype(int)
    sequence += additions

    remainder = target_degree_sum - int(sequence.sum())
    fractional = allocation - additions
    for index in np.argsort(-fractional, kind="stable")[:remainder]:
        sequence[int(index)] += 1

    while sequence[0] > size - 1:
        excess = int(sequence[0] - (size - 1))
        sequence[0] = size - 1
        for index in range(1, size):
            room = size - 1 - int(sequence[index])
            moved = min(room, excess)
            sequence[index] += moved
            excess -= moved
            if excess == 0:
                break
        if excess:
            raise RuntimeError("could not cap the power-law degree sequence")

    return sorted((int(value) for value in sequence), reverse=True)


def generate_powerlaw_line_graph(size: int, density: float, seed: int) -> Tuple[nx.Graph, Dict]:
    candidates = []
    for rank_exponent in np.linspace(0.05, 3.0, 600):
        sequence = _powerlaw_degree_sequence(size, float(rank_exponent))
        if not nx.is_graphical(sequence):
            continue
        line_edges = sum(degree * (degree - 1) // 2 for degree in sequence)
        line_density = line_edges / (size * (size - 1) / 2)
        candidates.append(
            (
                abs(line_density - density),
                float(rank_exponent),
                line_density,
                sequence,
            )
        )

    if not candidates:
        raise RuntimeError("could not construct a graphical power-law degree sequence")

    _, rank_exponent, expected_line_density, sequence = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    source = nx.havel_hakimi_graph(sequence)
    requested_swaps = 5 * source.number_of_edges()
    completed_swaps = requested_swaps
    try:
        nx.double_edge_swap(
            source,
            nswap=requested_swaps,
            max_tries=max(100, 100 * source.number_of_edges()),
            seed=seed,
        )
    except nx.NetworkXAlgorithmError:
        completed_swaps = 0

    line_graph = nx.line_graph(source)
    nodes = sorted(line_graph.nodes(), key=repr)
    sampled = line_graph
    sample_method = "not_needed"
    if len(nodes) > size:
        rng = np.random.default_rng(seed + 17)
        selected_positions = np.sort(rng.choice(len(nodes), size=size, replace=False))
        selected_nodes = [nodes[int(position)] for position in selected_positions]
        sampled = line_graph.subgraph(selected_nodes).copy()
        sample_method = "seeded_uniform"

    graph = nx.convert_node_labels_to_integers(nx.Graph(sampled))
    return graph, {
        "model": "line_graph_randomized_powerlaw_degree_sequence",
        "rank_exponent": rank_exponent,
        "implied_degree_distribution_exponent": 1 + 1 / rank_exponent,
        "degree_sequence": sequence,
        "source_nodes": source.number_of_nodes(),
        "source_edges": source.number_of_edges(),
        "source_density": nx.density(source),
        "source_degree_preserving_swaps_requested": requested_swaps,
        "source_degree_preserving_swaps_completed": completed_swaps,
        "line_graph_nodes_before_sample": line_graph.number_of_nodes(),
        "line_graph_edges_before_sample": line_graph.number_of_edges(),
        "line_graph_density_before_sample": nx.density(line_graph),
        "expected_line_graph_density_from_degrees": expected_line_density,
        "sample_method": sample_method,
        "selected_model_seed": seed,
        "candidate_count": len(candidates),
    }


def generate_graph(family: str, size: int, density: float, seed: int) -> Tuple[nx.Graph, Dict]:
    if family == "sbm":
        return generate_sbm_graph(size, density, seed)
    if family == "sbm_near_epsilon":
        return generate_sbm_near_epsilon_graph(size, density, seed)
    if family == "powerlaw_line":
        return generate_powerlaw_line_graph(size, density, seed)
    if family == "watts_strogatz":
        return generate_watts_strogatz_graph(size, density, seed)
    raise ValueError(f"unknown graph family: {family}")


def _build_cells(
    graph: nx.Graph,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
) -> Tuple[List[CellData], int]:
    from tripartite import GraphManager

    assembler = EdgePartitionAssembler(GraphManager(graph))
    edges_A, edges_B = assembler.get_edge_lists()
    if len(labels_A) != len(edges_A) or len(labels_B) != len(edges_B):
        raise ValueError("partition label lengths do not match tripartite edge sets")

    labels_A = np.asarray(labels_A, dtype=int)
    labels_B = np.asarray(labels_B, dtype=int)
    unique_A = sorted(set(labels_A.tolist()))
    unique_B = sorted(set(labels_B.tolist()))
    cells = {
        (label_A, label_B): CellData(label_A, label_B, [])
        for label_A in unique_A
        for label_B in unique_B
    }

    by_middle: Dict[int, Dict[str, Dict[int, List[Tuple[int, int]]]]] = {
        int(node): {"A": {}, "B": {}}
        for node in graph.nodes()
    }
    for index, ((a_node, middle_node), label) in enumerate(zip(edges_A, labels_A)):
        by_middle[int(middle_node[0])]["A"].setdefault(int(label), []).append(
            (index, int(a_node[0]))
        )
    for index, ((middle_node, b_node), label) in enumerate(zip(edges_B, labels_B)):
        by_middle[int(middle_node[0])]["B"].setdefault(int(label), []).append(
            (index, int(b_node[0]))
        )

    total_pathweight = 0
    for middle, middle_data in by_middle.items():
        for label_A, values_A in middle_data["A"].items():
            for label_B, values_B in middle_data["B"].items():
                a_indices = np.asarray([item[0] for item in values_A], dtype=int)
                b_indices = np.asarray([item[0] for item in values_B], dtype=int)
                a_nodes = np.asarray([item[1] for item in values_A], dtype=int)
                b_nodes = np.asarray([item[1] for item in values_B], dtype=int)
                closure = np.fromiter(
                    (
                        1.0 if graph.has_edge(int(a_node), int(b_node)) else 0.0
                        for a_node in a_nodes
                        for b_node in b_nodes
                    ),
                    dtype=float,
                    count=len(a_nodes) * len(b_nodes),
                ).reshape((len(a_nodes), len(b_nodes)))
                cells[(label_A, label_B)].groups.append(
                    CellGroup(
                        middle=middle,
                        a_edge_indices=a_indices,
                        b_edge_indices=b_indices,
                        a_nodes=a_nodes,
                        b_nodes=b_nodes,
                        closure=closure,
                    )
                )
                total_pathweight += len(a_nodes) * len(b_nodes)

    expected_total = sum(graph.degree(node) ** 2 for node in graph.nodes())
    if total_pathweight != expected_total:
        raise AssertionError(
            f"cell pathweights sum to {total_pathweight}, expected {expected_total}"
        )
    return list(cells.values()), expected_total


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


def _paper_diagnostics(
    cell: CellData,
    pathweight: int,
    gamma: float,
    config: VerificationConfig,
) -> Tuple[Dict, List[Tuple[Set[int], Set[int], str]]]:
    thresholds = config.thresholds
    irregular_weight = 0
    deviation_weight = 0
    irregular_vertex_count = 0
    deviation_vertex_count = 0
    vertex_summaries = []
    candidates: List[Tuple[Set[int], Set[int], str]] = []
    positive_A: Set[int] = set()
    negative_A: Set[int] = set()
    irregular_B: Set[int] = set()
    deviation_A: Set[int] = set()
    deviation_B: Set[int] = set()

    for group in cell.groups:
        count_A, count_B = group.closure.shape
        local_pathweight = count_A * count_B
        row_degrees = group.closure.sum(axis=1)
        signed_errors = row_degrees - gamma * count_B
        irregular_rows = np.abs(signed_errors) > thresholds["delta_3"] * count_B
        irregular_count = int(irregular_rows.sum())
        is_triangle_irregular = irregular_count >= thresholds["delta_4"] * count_A

        column_degrees = group.closure.sum(axis=0)
        sigma = float(
            np.dot(column_degrees, column_degrees)
            - gamma**2 * count_A**2 * count_B
        )
        sigma_threshold = thresholds["delta_1"] * count_A**2 * count_B
        is_deviation = sigma > sigma_threshold

        if is_triangle_irregular:
            irregular_vertex_count += 1
            irregular_weight += local_pathweight
            irregular_B.update(int(index) for index in group.b_edge_indices)
            positive_A.update(
                int(group.a_edge_indices[index])
                for index in np.flatnonzero(
                    signed_errors > thresholds["delta_3"] * count_B
                )
            )
            negative_A.update(
                int(group.a_edge_indices[index])
                for index in np.flatnonzero(
                    signed_errors < -thresholds["delta_3"] * count_B
                )
            )

        if is_deviation:
            deviation_vertex_count += 1
            deviation_weight += local_pathweight
            common = group.closure @ group.closure.T
            specific = common - gamma**2 * count_B
            regular_rows = np.flatnonzero(
                np.abs(signed_errors) <= thresholds["delta_3"] * count_B
            )
            if regular_rows.size == 0:
                regular_rows = np.arange(count_A)
            scores = specific.sum(axis=1)
            pivot = int(regular_rows[np.argmax(scores[regular_rows])])
            selected_rows = np.flatnonzero(
                specific[pivot, :] > thresholds["delta_6"] * count_B
            )
            selected_columns = np.flatnonzero(group.closure[pivot, :] > 0)
            deviation_A.update(
                int(group.a_edge_indices[index]) for index in selected_rows
            )
            deviation_B.update(
                int(group.b_edge_indices[index]) for index in selected_columns
            )

        vertex_summaries.append(
            {
                "middle": int(group.middle),
                "degree_A": count_A,
                "degree_B": count_B,
                "pathweight": local_pathweight,
                "triangle_irregular_edge_count": irregular_count,
                "triangle_irregular_edge_threshold": thresholds["delta_4"] * count_A,
                "triangle_irregular": bool(is_triangle_irregular),
                "sigma": sigma,
                "sigma_threshold": sigma_threshold,
                "deviation_vertex": bool(is_deviation),
            }
        )

    if positive_A and irregular_B:
        candidates.append((positive_A, irregular_B, "paper_triangle_positive"))
    if negative_A and irregular_B:
        candidates.append((negative_A, irregular_B, "paper_triangle_negative"))
    if deviation_A and deviation_B:
        candidates.append((deviation_A, deviation_B, "paper_deviation"))

    irregular_limit = thresholds["delta_5"] * pathweight
    deviation_limit = thresholds["delta_2"] * pathweight
    paper_status = "regular_paper_checks"
    paper_reason = "triangle_and_deviation_checks_passed"
    if irregular_weight >= irregular_limit:
        paper_status = "refinement_required"
        paper_reason = "triangle_irregularity_threshold"
    elif deviation_weight >= deviation_limit:
        paper_status = "refinement_required"
        paper_reason = "deviation_threshold"

    worst_vertices = sorted(
        vertex_summaries,
        key=lambda item: (
            item["triangle_irregular"],
            item["deviation_vertex"],
            item["pathweight"],
            item["sigma"] - item["sigma_threshold"],
        ),
        reverse=True,
    )[:10]
    return {
        "paper_status": paper_status,
        "paper_reason": paper_reason,
        "triangle_irregular_vertex_count": irregular_vertex_count,
        "triangle_irregular_pathweight": irregular_weight,
        "triangle_irregular_pathweight_limit": irregular_limit,
        "deviation_vertex_count": deviation_vertex_count,
        "deviation_pathweight": deviation_weight,
        "deviation_pathweight_limit": deviation_limit,
        "worst_vertices": worst_vertices,
    }, candidates


def _exhaustive_definition_check(
    cell: CellData,
    pathweight: int,
    gamma: float,
    config: VerificationConfig,
) -> Tuple[str, Optional[Dict], Dict]:
    indices_A = cell.active_a_indices
    indices_B = cell.active_b_indices
    best_witness = None

    for mask_A in range(1, 1 << len(indices_A)):
        selected_A = {
            indices_A[index]
            for index in range(len(indices_A))
            if mask_A & (1 << index)
        }
        for mask_B in range(1, 1 << len(indices_B)):
            selected_B = {
                indices_B[index]
                for index in range(len(indices_B))
                if mask_B & (1 << index)
            }
            witness = _witness_report(
                cell,
                selected_A,
                selected_B,
                pathweight,
                gamma,
                config.eps,
                source="exhaustive",
            )
            if witness is None:
                continue
            if (
                best_witness is None
                or witness["absolute_gamma_difference"]
                > best_witness["absolute_gamma_difference"]
            ):
                best_witness = witness

    if best_witness is not None:
        return "proven_irregular", best_witness, {
            "method": "exhaustive",
            "combined_active_edge_count": len(indices_A) + len(indices_B),
        }
    return "certified_regular", None, {
        "method": "exhaustive",
        "combined_active_edge_count": len(indices_A) + len(indices_B),
    }


def _matrix_norm_upper_bound(
    matrix: np.ndarray,
    dense_limit: int,
) -> Tuple[float, str, Optional[np.ndarray], Optional[np.ndarray]]:
    if matrix.size == 0 or not np.any(matrix):
        return 0.0, "zero", None, None

    rows, columns = matrix.shape
    scale = max(1.0, float(np.linalg.norm(matrix, ord="fro")))
    inflation = 16 * np.finfo(float).eps * max(rows, columns) * scale
    if max(rows, columns) <= dense_limit:
        left, singular_values, right_transpose = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        return (
            float(singular_values[0] + inflation),
            "full_svd_backward_error_bound",
            left[:, 0],
            right_transpose[0, :],
        )

    frobenius = float(np.linalg.norm(matrix, ord="fro"))
    induced = math.sqrt(
        float(np.linalg.norm(matrix, ord=1))
        * float(np.linalg.norm(matrix, ord=np.inf))
    )
    return min(frobenius, induced) + inflation, "induced_or_frobenius", None, None


def _water_filling_bound(
    norm_bounds: Sequence[float],
    capacities: Sequence[int],
    required_pathweight: float,
) -> Tuple[float, List[float]]:
    if required_pathweight <= 0:
        return 0.0, [0.0 for _ in capacities]
    if required_pathweight > sum(capacities) + 1e-12:
        raise ValueError("required pathweight exceeds cell capacity")

    positive_capacity = sum(
        capacity
        for norm, capacity in zip(norm_bounds, capacities)
        if norm > 0
    )
    if positive_capacity <= required_pathweight:
        allocation = [
            float(capacity) if norm > 0 else 0.0
            for norm, capacity in zip(norm_bounds, capacities)
        ]
        remaining = required_pathweight - sum(allocation)
        for index, (norm, capacity) in enumerate(zip(norm_bounds, capacities)):
            if remaining <= 0:
                break
            if norm > 0:
                continue
            added = min(float(capacity), remaining)
            allocation[index] = added
            remaining -= added
    else:
        low = 0.0
        high = max(norm_bounds)
        while sum(
            min(float(capacity), (norm / (2 * high)) ** 2)
            for norm, capacity in zip(norm_bounds, capacities)
            if norm > 0
        ) > required_pathweight:
            high *= 2

        for _ in range(100):
            multiplier = (low + high) / 2
            allocated = sum(
                min(float(capacity), (norm / (2 * multiplier)) ** 2)
                for norm, capacity in zip(norm_bounds, capacities)
                if norm > 0
            )
            if allocated > required_pathweight:
                low = multiplier
            else:
                high = multiplier
        allocation = [
            min(float(capacity), (norm / (2 * high)) ** 2)
            if norm > 0
            else 0.0
            for norm, capacity in zip(norm_bounds, capacities)
        ]
        difference = required_pathweight - sum(allocation)
        if difference > 1e-9:
            for index, capacity in enumerate(capacities):
                room = float(capacity) - allocation[index]
                added = min(room, difference)
                allocation[index] += added
                difference -= added
                if difference <= 1e-9:
                    break

    numerator = sum(
        norm * math.sqrt(max(0.0, allocated))
        for norm, allocated in zip(norm_bounds, allocation)
    )
    return numerator / required_pathweight, allocation


def _spectral_definition_check(
    cell: CellData,
    pathweight: int,
    gamma: float,
    config: VerificationConfig,
) -> Tuple[bool, Dict, List[Tuple[Set[int], Set[int], str]]]:
    norm_bounds = []
    capacities = []
    methods = []
    singular_data = []

    for group in cell.groups:
        centered = group.closure - gamma
        bound, method, left, right = _matrix_norm_upper_bound(
            centered,
            config.svd_dense_limit,
        )
        norm_bounds.append(bound)
        capacities.append(group.closure.shape[0] * group.closure.shape[1])
        methods.append(method)
        singular_data.append((group, left, right))

    required_pathweight = config.eps * pathweight
    bound_ratio, allocation = _water_filling_bound(
        norm_bounds,
        capacities,
        required_pathweight,
    )
    certified = bound_ratio <= config.eps - config.numerical_tolerance

    candidates: List[Tuple[Set[int], Set[int], str]] = []
    for orientation_A, orientation_B in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for fraction in (0.25, 0.5):
            selected_A: Set[int] = set()
            selected_B: Set[int] = set()
            for group, left, right in singular_data:
                if left is None or right is None:
                    continue
                count_A = max(1, int(math.ceil(fraction * len(left))))
                count_B = max(1, int(math.ceil(fraction * len(right))))
                order_A = np.argsort(-(orientation_A * left), kind="stable")[:count_A]
                order_B = np.argsort(-(orientation_B * right), kind="stable")[:count_B]
                selected_A.update(
                    int(group.a_edge_indices[index]) for index in order_A
                )
                selected_B.update(
                    int(group.b_edge_indices[index]) for index in order_B
                )
            if selected_A and selected_B:
                candidates.append(
                    (
                        selected_A,
                        selected_B,
                        f"spectral_{orientation_A}_{orientation_B}_{fraction}",
                    )
                )

    return certified, {
        "method": "spectral_water_filling",
        "required_pathweight": required_pathweight,
        "deviation_ratio_upper_bound": bound_ratio,
        "epsilon": config.eps,
        "certificate_margin": config.eps - bound_ratio,
        "norm_methods": methods,
        "norm_upper_bounds": norm_bounds,
        "capacities": capacities,
        "water_filling_allocation": allocation,
    }, candidates


def _coordinate_improve(
    cell: CellData,
    selected_A: Set[int],
    selected_B: Set[int],
    pathweight: int,
    gamma: float,
    config: VerificationConfig,
    sign: int,
) -> Tuple[Set[int], Set[int]]:
    selected_A = set(selected_A)
    selected_B = set(selected_B)
    active_A = cell.active_a_indices[: config.coordinate_edge_limit]
    active_B = cell.active_b_indices[: config.coordinate_edge_limit]
    minimum_pathweight = config.eps * pathweight

    def objective(candidate_A: Set[int], candidate_B: Set[int]) -> float:
        candidate_pathweight, candidate_triangles = _selection_counts(
            cell,
            candidate_A,
            candidate_B,
        )
        if candidate_pathweight + 1e-12 < minimum_pathweight:
            return float("-inf")
        signed_deviation = sign * (
            candidate_triangles - gamma * candidate_pathweight
        )
        return signed_deviation - config.eps * candidate_pathweight

    current_score = objective(selected_A, selected_B)
    for _ in range(config.coordinate_passes):
        best_score = current_score
        best_selection = None
        for index in active_A:
            candidate_A = set(selected_A)
            if index in candidate_A:
                candidate_A.remove(index)
            else:
                candidate_A.add(index)
            score = objective(candidate_A, selected_B)
            if score > best_score + config.numerical_tolerance:
                best_score = score
                best_selection = (candidate_A, set(selected_B))
        for index in active_B:
            candidate_B = set(selected_B)
            if index in candidate_B:
                candidate_B.remove(index)
            else:
                candidate_B.add(index)
            score = objective(selected_A, candidate_B)
            if score > best_score + config.numerical_tolerance:
                best_score = score
                best_selection = (set(selected_A), candidate_B)
        if best_selection is None:
            break
        selected_A, selected_B = best_selection
        current_score = best_score
    return selected_A, selected_B


def _adversarial_witness_search(
    cell: CellData,
    pathweight: int,
    gamma: float,
    config: VerificationConfig,
    candidates: Sequence[Tuple[Set[int], Set[int], str]],
) -> Tuple[Optional[Dict], Dict]:
    all_A = set(cell.active_a_indices)
    all_B = set(cell.active_b_indices)
    starts = [(all_A, all_B, "full_cell"), *candidates]
    starts = starts[: config.search_restarts]
    tested = 0
    best_witness = None

    for selected_A, selected_B, source in starts:
        for sign in (1, -1):
            improved_A, improved_B = _coordinate_improve(
                cell,
                selected_A,
                selected_B,
                pathweight,
                gamma,
                config,
                sign,
            )
            tested += 1
            witness = _witness_report(
                cell,
                improved_A,
                improved_B,
                pathweight,
                gamma,
                config.eps,
                source=f"{source}_coordinate_{sign}",
            )
            if witness is not None and (
                best_witness is None
                or witness["absolute_gamma_difference"]
                > best_witness["absolute_gamma_difference"]
            ):
                best_witness = witness

        tested += 1
        witness = _witness_report(
            cell,
            selected_A,
            selected_B,
            pathweight,
            gamma,
            config.eps,
            source=source,
        )
        if witness is not None and (
            best_witness is None
            or witness["absolute_gamma_difference"]
            > best_witness["absolute_gamma_difference"]
        ):
            best_witness = witness

    return best_witness, {
        "method": "deterministic_adversarial_search",
        "starts_considered": len(starts),
        "candidates_tested": tested,
        "search_is_certificate": False,
    }


def verify_cell(
    cell: CellData,
    config: VerificationConfig,
) -> Dict:
    pathweight, triangle_count, gamma = _cell_counts(cell)
    base = {
        "label_A": int(cell.label_A),
        "label_B": int(cell.label_B),
        "pathweight": pathweight,
        "triangle_count": triangle_count,
        "gamma": gamma,
        "active_edge_count_A": len(cell.active_a_indices),
        "active_edge_count_B": len(cell.active_b_indices),
    }

    if pathweight == 0:
        return {
            **base,
            "paper_status": "regular_zero_path",
            "paper_reason": "zero_pathweight",
            "definition_status": "certified_regular",
            "definition_method": {
                "method": "zero_pathweight",
            },
            "witness": None,
            "paper_diagnostics": None,
        }

    diagnostics, paper_candidates = _paper_diagnostics(
        cell,
        pathweight,
        gamma,
        config,
    )
    if gamma < config.eps:
        diagnostics["paper_status"] = "regular_low_gamma"
        diagnostics["paper_reason"] = "gamma_below_epsilon"

    combined_edges = len(cell.active_a_indices) + len(cell.active_b_indices)
    if combined_edges <= config.exhaustive_edge_limit:
        definition_status, witness, method = _exhaustive_definition_check(
            cell,
            pathweight,
            gamma,
            config,
        )
    else:
        certified, method, spectral_candidates = _spectral_definition_check(
            cell,
            pathweight,
            gamma,
            config,
        )
        witness = None
        if certified:
            definition_status = "certified_regular"
        else:
            definition_status = "inconclusive"
            if config.witness_search:
                witness, search_report = _adversarial_witness_search(
                    cell,
                    pathweight,
                    gamma,
                    config,
                    [*paper_candidates, *spectral_candidates],
                )
                method["witness_search"] = search_report
                if witness is not None:
                    definition_status = "proven_irregular"

    return {
        **base,
        "paper_status": diagnostics["paper_status"],
        "paper_reason": diagnostics["paper_reason"],
        "definition_status": definition_status,
        "definition_method": method,
        "witness": witness,
        "paper_diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if key not in {"paper_status", "paper_reason"}
        },
    }


def aggregate_definition_status(
    cell_reports: Sequence[Dict],
    total_pathweight: int,
    eps: float,
) -> Dict:
    proven_irregular_weight = sum(
        cell["pathweight"]
        for cell in cell_reports
        if cell["definition_status"] == "proven_irregular"
    )
    inconclusive_weight = sum(
        cell["pathweight"]
        for cell in cell_reports
        if cell["definition_status"] == "inconclusive"
    )
    unresolved_weight = proven_irregular_weight + inconclusive_weight
    limit = eps * total_pathweight

    if total_pathweight == 0 or unresolved_weight < limit:
        status = "certified_success"
    elif proven_irregular_weight >= limit:
        status = "proven_failure"
    else:
        status = "inconclusive"
    return {
        "verification_status": status,
        "proven_irregular_pathweight": proven_irregular_weight,
        "inconclusive_pathweight": inconclusive_weight,
        "unresolved_pathweight": unresolved_weight,
        "allowed_irregular_pathweight": limit,
        "proven_irregular_pathweight_ratio": (
            0.0
            if total_pathweight == 0
            else proven_irregular_weight / total_pathweight
        ),
        "inconclusive_pathweight_ratio": (
            0.0
            if total_pathweight == 0
            else inconclusive_weight / total_pathweight
        ),
    }


def _aggregate_paper_status(
    cell_reports: Sequence[Dict],
    total_pathweight: int,
    eps: float,
) -> Dict:
    refinement_weight = sum(
        cell["pathweight"]
        for cell in cell_reports
        if cell["paper_status"] == "refinement_required"
    )
    limit = eps * total_pathweight
    status = (
        "regular"
        if total_pathweight == 0 or refinement_weight < limit
        else "refinement_required"
    )
    return {
        "paper_partition_status": status,
        "paper_refinement_pathweight": refinement_weight,
        "paper_allowed_irregular_pathweight": limit,
        "paper_refinement_pathweight_ratio": (
            0.0 if total_pathweight == 0 else refinement_weight / total_pathweight
        ),
    }


def compute_partition_regularity_report(
    graph: nx.Graph,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    config: Optional[VerificationConfig] = None,
) -> Dict:
    config = config or VerificationConfig()
    cells, total_pathweight = _build_cells(graph, labels_A, labels_B)
    cell_reports = [verify_cell(cell, config) for cell in cells]
    definition = aggregate_definition_status(
        cell_reports,
        total_pathweight,
        config.eps,
    )
    paper = _aggregate_paper_status(
        cell_reports,
        total_pathweight,
        config.eps,
    )
    counts = {
        status: sum(
            1 for cell in cell_reports if cell["definition_status"] == status
        )
        for status in ("certified_regular", "proven_irregular", "inconclusive")
    }
    return {
        **definition,
        **paper,
        "passes": definition["verification_status"] == "certified_success",
        "total_pathweight": total_pathweight,
        "cell_count": len(cell_reports),
        "definition_status_counts": counts,
        "paper_thresholds": config.thresholds,
        "verification_config": asdict(config),
        "cells": cell_reports,
    }


def compute_partition_deviation_report(
    graph: nx.Graph,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    parameters: Optional[AlgorithmParameters] = None,
) -> Dict:
    """Compatibility wrapper for the former verifier entry point."""
    eps = 0.05 if parameters is None else parameters.eps
    return compute_partition_regularity_report(
        graph,
        labels_A,
        labels_B,
        VerificationConfig(eps=eps),
    )


def run_algorithm_case(
    case: VerificationCase,
    eps: float = 0.05,
    max_depth: int = 8,
    verification_config: Optional[VerificationConfig] = None,
    telemetry_callback=None,
) -> Dict:
    start = time.monotonic()
    config = verification_config or VerificationConfig(eps=eps)
    if not math.isclose(config.eps, eps, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("verification_config.eps must match the algorithm eps")

    def emit(event: str, **details) -> None:
        if telemetry_callback is not None:
            telemetry_callback({"event": event, **details})

    emit("phase_started", phase="generation")
    graph, generator_metadata = generate_graph(
        case.family,
        case.size,
        case.density,
        case.seed,
    )
    actual_density = nx.density(graph)
    density_error = abs(actual_density - case.density)
    common = {
        "case_id": case.case_id,
        "family": case.family,
        "requested_size": case.size,
        "actual_size": graph.number_of_nodes(),
        "requested_density": case.density,
        "actual_density": actual_density,
        "density_error": density_error,
        "density_tolerance": config.density_tolerance,
        "edge_count": graph.number_of_edges(),
        "seed": case.seed,
        "eps": eps,
        "max_direction_code_length": max_depth,
        "generator": generator_metadata,
        "graph_clustering": graph_clustering_metrics(graph, eps),
        "finite_scale_thresholds": finite_scale_threshold_report(graph, config),
    }
    emit("phase_completed", phase="generation", **common)
    if density_error > config.density_tolerance:
        return {
            **common,
            "status": "generation_mismatch",
            "execution_status": "generation_mismatch",
            "verification_status": None,
            "runtime_seconds": time.monotonic() - start,
        }

    parameters = AlgorithmParameters(eps=eps)
    emit("phase_started", phase="algorithm")
    with tempfile.TemporaryDirectory(prefix=f"verify_{case.case_id}_") as temp_dir:
        runner = AlgorithmRunner(
            graph,
            parameters=parameters,
            partition_dir=os.path.join(temp_dir, "partitions"),
            graph_dir=os.path.join(temp_dir, "graphs"),
            max_depth=max_depth,
            progress_callback=(
                (lambda event: emit("algorithm_progress", progress=event))
                if telemetry_callback is not None
                else None
            ),
        )
        labels_A, labels_B = runner.run()
    emit(
        "phase_completed",
        phase="algorithm",
        partition_count_A=(
            int(len(set(labels_A.tolist()))) if len(labels_A) else 0
        ),
        partition_count_B=(
            int(len(set(labels_B.tolist()))) if len(labels_B) else 0
        ),
        directions_considered=len(runner.directions_considered),
    )

    emit("phase_started", phase="verification")
    report = compute_partition_regularity_report(
        graph,
        labels_A,
        labels_B,
        config,
    )
    emit(
        "phase_completed",
        phase="verification",
        verification_status=report["verification_status"],
        paper_partition_status=report["paper_partition_status"],
        cell_count=report["cell_count"],
    )
    return {
        **common,
        "status": report["verification_status"],
        "execution_status": "completed",
        "verification_status": report["verification_status"],
        "paper_partition_status": report["paper_partition_status"],
        "runtime_seconds": time.monotonic() - start,
        "partition_count_A": int(len(set(labels_A.tolist()))) if len(labels_A) else 0,
        "partition_count_B": int(len(set(labels_B.tolist()))) if len(labels_B) else 0,
        "directions_considered": runner.directions_considered,
        "accepted_directions": runner.accepted_directions,
        "partition_assembly_diagnostics": runner.partition_assembly_diagnostics,
        "verification": report,
    }


def _write_worker_result(result_path: str, result: Dict) -> None:
    temporary_path = f"{result_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    os.replace(temporary_path, result_path)


def _worker(case_data, eps, max_depth, config_data, result_path, telemetry_queue):
    case = VerificationCase(**case_data)
    config = VerificationConfig(**config_data)
    worker_start = time.monotonic()

    def emit(event: Dict) -> None:
        payload = {
            **event,
            "elapsed_seconds": time.monotonic() - worker_start,
        }
        try:
            telemetry_queue.put_nowait(payload)
        except Exception:
            pass

    try:
        _write_worker_result(
            result_path,
            run_algorithm_case(
                case,
                eps=eps,
                max_depth=max_depth,
                verification_config=config,
                telemetry_callback=emit,
            )
        )
    except Exception as exc:
        execution_status = (
            "max_depth"
            if isinstance(exc, ValueError)
            and "exceeds maximum depth" in str(exc)
            else "algorithm_error"
        )
        emit(
            {
                "event": "worker_failed",
                "execution_status": execution_status,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_worker_result(
            result_path,
            {
                "case_id": case.case_id,
                "status": execution_status,
                "execution_status": execution_status,
                "verification_status": None,
                "family": case.family,
                "requested_size": case.size,
                "requested_density": case.density,
                "seed": case.seed,
                "eps": eps,
                "max_direction_code_length": max_depth,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _drain_telemetry(telemetry_queue, events: List[Dict]) -> None:
    while True:
        try:
            events.append(telemetry_queue.get_nowait())
        except queue_module.Empty:
            return


def _summarize_telemetry(events: Sequence[Dict], runtime_seconds: float) -> Dict:
    phase_started = {}
    phase_durations = {}
    active_phase = None
    directions_started = 0
    directions_completed = 0
    max_code_length = 0
    max_logical_depth = 0
    last_direction = None
    last_completed_direction = None
    branch_counts = {}
    generation = None
    accepted_direction_count = None
    assembly = None

    for event in events:
        event_name = event.get("event")
        elapsed = float(event.get("elapsed_seconds", 0.0))
        if event_name == "phase_started":
            active_phase = event["phase"]
            phase_started[event["phase"]] = elapsed
        elif event_name == "phase_completed":
            phase = event["phase"]
            start = phase_started.get(phase)
            if start is not None:
                phase_durations[phase] = elapsed - start
            if active_phase == phase:
                active_phase = None
            if phase == "generation":
                generation = {
                    key: value
                    for key, value in event.items()
                    if key not in {"event", "elapsed_seconds", "phase"}
                }
        elif event_name == "algorithm_progress":
            progress = event.get("progress", {})
            progress_name = progress.get("event")
            if progress_name == "direction_started":
                directions_started += 1
                last_direction = progress.get("direction")
                max_code_length = max(
                    max_code_length,
                    int(progress.get("direction_code_length", 0)),
                )
                max_logical_depth = max(
                    max_logical_depth,
                    int(progress.get("logical_refinement_depth", 0)),
                )
            elif progress_name == "direction_completed":
                directions_completed += 1
                last_completed_direction = progress.get("direction")
                reason = progress.get("failure_reason", "UNKNOWN")
                branch_counts[reason] = branch_counts.get(reason, 0) + 1
            elif progress_name == "refinement_completed":
                accepted_direction_count = progress.get("accepted_direction_count")
            elif progress_name == "assembly_completed":
                assembly = {
                    "partition_count_A": progress.get("partition_count_A"),
                    "partition_count_B": progress.get("partition_count_B"),
                    "accepted_directions": progress.get("accepted_directions", []),
                    "partition_assembly_diagnostics": progress.get(
                        "partition_assembly_diagnostics"
                    ),
                }

    return {
        "active_phase_at_exit": active_phase,
        "event_count": len(events),
        "runtime_seconds": runtime_seconds,
        "phase_durations_seconds": phase_durations,
        "directions_started": directions_started,
        "directions_completed": directions_completed,
        "max_direction_code_length_seen": max_code_length,
        "max_logical_refinement_depth_seen": max_logical_depth,
        "last_direction_started": last_direction,
        "last_direction_completed": last_completed_direction,
        "branch_counts": branch_counts,
        "generation": generation,
        "accepted_direction_count": accepted_direction_count,
        "assembly": assembly,
    }


def run_case_with_timeout(
    case: VerificationCase,
    timeout_seconds: float = 20.0,
    eps: float = 0.05,
    max_depth: int = 8,
    verification_config: Optional[VerificationConfig] = None,
) -> Dict:
    config = verification_config or VerificationConfig(eps=eps)
    context = mp.get_context("spawn")
    telemetry_queue = context.Queue()
    with tempfile.TemporaryDirectory(prefix=f"verify_result_{case.case_id}_") as result_dir:
        result_path = os.path.join(result_dir, "result.json")
        process = context.Process(
            target=_worker,
            args=(
                case.__dict__,
                eps,
                max_depth,
                asdict(config),
                result_path,
                telemetry_queue,
            ),
        )
        start = time.monotonic()
        telemetry_events: List[Dict] = []
        process.start()
        while process.is_alive():
            elapsed = time.monotonic() - start
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                break
            process.join(min(0.1, remaining))
            _drain_telemetry(telemetry_queue, telemetry_events)
        runtime = time.monotonic() - start
        _drain_telemetry(telemetry_queue, telemetry_events)
        telemetry = _summarize_telemetry(telemetry_events, runtime)

        if process.is_alive():
            process.terminate()
            process.join(5)
            return {
                "case_id": case.case_id,
                "status": "timeout",
                "execution_status": "timeout",
                "verification_status": None,
                "runtime_seconds": runtime,
                "family": case.family,
                "requested_size": case.size,
                "requested_density": case.density,
                "seed": case.seed,
                "eps": eps,
                "max_direction_code_length": max_depth,
                "timeout_seconds": timeout_seconds,
                "telemetry": telemetry,
            }

        if not os.path.exists(result_path):
            return {
                "case_id": case.case_id,
                "status": "algorithm_error",
                "execution_status": "algorithm_error",
                "verification_status": None,
                "runtime_seconds": runtime,
                "family": case.family,
                "requested_size": case.size,
                "requested_density": case.density,
                "seed": case.seed,
                "eps": eps,
                "max_direction_code_length": max_depth,
                "error": (
                    f"worker exited with code {process.exitcode} "
                    "without writing a result"
                ),
                "telemetry": telemetry,
            }
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)

    result.setdefault("runtime_seconds", runtime)
    result["telemetry"] = telemetry
    return result


def _report_document(
    results: Sequence[Dict],
    run_config: Optional[Dict] = None,
) -> Dict:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config or {},
        "cases": sorted(results, key=lambda item: item["case_id"]),
    }


def write_verification_report(
    results: Sequence[Dict],
    path: str = REPORT_PATH,
    run_config: Optional[Dict] = None,
) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    document = _report_document(results, run_config)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".clustering_verification_",
        suffix=".json",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _partial_report_path(size: int, max_depth: int, timeout_seconds: float) -> str:
    filename = (
        f"clustering_verification_results_n{size}"
        f"_code{max_depth}_timeout{int(timeout_seconds)}.json"
    )
    return os.path.join("docs", "debug", filename)


def rerun_size_cases(
    size: int,
    timeout_seconds: float,
    max_depth: int,
    eps: float = 0.05,
) -> List[Dict]:
    results = []
    config = VerificationConfig(eps=eps)
    for case in verification_cases():
        if case.size != size:
            continue
        print(f"RUN {case.case_id}", flush=True)
        result = run_case_with_timeout(
            case,
            timeout_seconds=timeout_seconds,
            eps=eps,
            max_depth=max_depth,
            verification_config=config,
        )
        result["timeout_seconds"] = timeout_seconds
        results.append(result)
        print(
            f"  -> {result['status']} "
            f"execution={result['execution_status']} "
            f"runtime={result.get('runtime_seconds', 0):.2f}s",
            flush=True,
        )

    path = _partial_report_path(size, max_depth, timeout_seconds)
    write_verification_report(
        results,
        path=path,
        run_config={
            "scope": "partial_size_run",
            "size": size,
            "timeout_seconds": timeout_seconds,
            "max_direction_code_length": max_depth,
            "verification_config": asdict(config),
        },
    )
    return results


def run_all_cases(
    timeout_seconds: float,
    max_depth: int,
    eps: float = 0.05,
) -> List[Dict]:
    """Run the complete matrix and atomically replace the canonical report."""
    results = []
    config = VerificationConfig(eps=eps)
    for case in verification_cases():
        print(f"RUN {case.case_id}", flush=True)
        result = run_case_with_timeout(
            case,
            timeout_seconds=timeout_seconds,
            eps=eps,
            max_depth=max_depth,
            verification_config=config,
        )
        result["timeout_seconds"] = timeout_seconds
        results.append(result)
        print(
            f"  -> {result['status']} "
            f"execution={result['execution_status']} "
            f"runtime={result.get('runtime_seconds', 0):.2f}s",
            flush=True,
        )

    write_verification_report(
        results,
        path=REPORT_PATH,
        run_config={
            "scope": "full_matrix",
            "timeout_seconds": timeout_seconds,
            "max_direction_code_length": max_depth,
            "verification_config": asdict(config),
        },
    )
    return results


def run_additional_sbm_case(
    timeout_seconds: float = 240.0,
    max_depth: int = 30,
    eps: float = 0.05,
) -> Dict:
    config = VerificationConfig(eps=eps)
    result = run_case_with_timeout(
        ADDITIONAL_SBM_CASE,
        timeout_seconds=timeout_seconds,
        eps=eps,
        max_depth=max_depth,
        verification_config=config,
    )
    result["timeout_seconds"] = timeout_seconds
    path = os.path.join(
        "docs",
        "debug",
        (
            f"clustering_verification_results_sbm_n{ADDITIONAL_SBM_CASE.size}"
            f"_code{max_depth}_timeout{int(timeout_seconds)}.json"
        ),
    )
    write_verification_report(
        [result],
        path=path,
        run_config={
            "scope": "additional_finite_scale_sbm",
            "timeout_seconds": timeout_seconds,
            "max_direction_code_length": max_depth,
            "verification_config": asdict(config),
        },
    )
    return result


def run_scaled_additional_cases(
    timeout_seconds: float = 180.0,
    max_depth: int = 120,
    eps: float = 0.05,
) -> List[Dict]:
    """Run the scaled nontrivial and near-epsilon diagnostic cases."""
    config = VerificationConfig(eps=eps)
    results = []
    for case in SCALED_ADDITIONAL_CASES:
        print(f"RUN {case.case_id}", flush=True)
        result = run_case_with_timeout(
            case,
            timeout_seconds=timeout_seconds,
            eps=eps,
            max_depth=max_depth,
            verification_config=config,
        )
        result["timeout_seconds"] = timeout_seconds
        results.append(result)
        print(
            f"  -> {result['status']} "
            f"execution={result['execution_status']} "
            f"runtime={result.get('runtime_seconds', 0):.2f}s",
            flush=True,
        )

    path = os.path.join(
        "docs",
        "debug",
        (
            "clustering_verification_results_scaled_additional"
            f"_code{max_depth}_timeout{int(timeout_seconds)}.json"
        ),
    )
    write_verification_report(
        results,
        path=path,
        run_config={
            "scope": "scaled_additional_cases",
            "timeout_seconds": timeout_seconds,
            "max_direction_code_length": max_depth,
            "verification_config": asdict(config),
        },
    )
    return results


def main() -> None:
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Run clustering verification cases.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--size", type=int)
    scope.add_argument("--all", action="store_true")
    scope.add_argument("--additional-sbm", action="store_true")
    scope.add_argument("--scaled-additional", action="store_true")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--eps", type=float, default=0.05)
    args = parser.parse_args()

    if args.scaled_additional:
        results = run_scaled_additional_cases(
            timeout_seconds=args.timeout,
            max_depth=args.max_depth,
            eps=args.eps,
        )
    elif args.additional_sbm:
        result = run_additional_sbm_case(
            timeout_seconds=args.timeout,
            max_depth=args.max_depth,
            eps=args.eps,
        )
        results = [result]
    elif args.all:
        results = run_all_cases(
            timeout_seconds=args.timeout,
            max_depth=args.max_depth,
            eps=args.eps,
        )
    else:
        results = rerun_size_cases(
            size=args.size,
            timeout_seconds=args.timeout,
            max_depth=args.max_depth,
            eps=args.eps,
        )
    print("status_counts", dict(Counter(item["status"] for item in results)))


if __name__ == "__main__":
    main()
