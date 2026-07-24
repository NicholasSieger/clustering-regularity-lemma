from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


def _balanced_block_sizes(size: int, block_count: int = 4) -> List[int]:
    sizes = [size // block_count] * block_count
    for index in range(size % block_count):
        sizes[index] += 1
    return sizes


def _sbm_probabilities(
    block_sizes: Sequence[int],
    target_density: float,
) -> Tuple[float, float]:
    total_pairs = sum(block_sizes) * (sum(block_sizes) - 1) / 2
    within_pairs = sum(
        block_size * (block_size - 1) / 2 for block_size in block_sizes
    )
    between_pairs = total_pairs - within_pairs
    target_edges = target_density * total_pairs
    low, high = 0.0, 1.0
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
    pathweight = sum(degree**2 for _, degree in graph.degree())
    if pathweight == 0:
        return 0.0
    return 2 * sum(nx.triangles(graph).values()) / pathweight


def graph_clustering_metrics(graph: nx.Graph, eps: float) -> Dict:
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
    error, density_error, selected_seed, selected_gamma, graph = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return graph, {
        "model": "stochastic_block_model_near_epsilon",
        "block_sizes": block_sizes,
        "intra_probability": intra,
        "inter_probability": inter,
        "target_root_gamma": target_clustering,
        "selected_root_gamma": selected_gamma,
        "root_gamma_error": error,
        "selected_density_error": density_error,
        "candidate_count": candidate_count,
        "selected_model_seed": selected_seed,
    }


def _even_degree_from_density(size: int, density: float) -> int:
    valid = list(range(2, size, 2))
    if not valid:
        raise ValueError("Watts-Strogatz generation requires at least three vertices")
    return min(valid, key=lambda value: (abs(value / (size - 1) - density), value))


def generate_watts_strogatz_graph(
    size: int,
    density: float,
    seed: int,
) -> Tuple[nx.Graph, Dict]:
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


def generate_powerlaw_line_graph(
    size: int,
    density: float,
    seed: int,
) -> Tuple[nx.Graph, Dict]:
    candidates = []
    for rank_exponent in np.linspace(0.05, 3.0, 600):
        sequence = _powerlaw_degree_sequence(size, float(rank_exponent))
        if not nx.is_graphical(sequence):
            continue
        line_edges = sum(degree * (degree - 1) // 2 for degree in sequence)
        line_density = line_edges / (size * (size - 1) / 2)
        candidates.append(
            (abs(line_density - density), float(rank_exponent), line_density, sequence)
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
        positions = np.sort(rng.choice(len(nodes), size=size, replace=False))
        sampled = line_graph.subgraph([nodes[int(position)] for position in positions]).copy()
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


def generate_graph(
    family: str,
    size: int,
    density: float,
    seed: int,
) -> Tuple[nx.Graph, Dict]:
    generators = {
        "sbm": generate_sbm_graph,
        "sbm_near_epsilon": generate_sbm_near_epsilon_graph,
        "powerlaw_line": generate_powerlaw_line_graph,
        "watts_strogatz": generate_watts_strogatz_graph,
    }
    try:
        generator = generators[family]
    except KeyError as exc:
        raise ValueError(f"unknown graph family: {family}") from exc
    return generator(size, density, seed)

