import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from algorithm import AlgorithmRunner
from edge_partition import EdgePartitionAssembler
from parameters import AlgorithmParameters


GRAPH_FAMILIES = ("sbm", "powerlaw_line", "watts_strogatz")
GRAPH_SIZES = (30, 100, 500)
DENSITIES = (0.05, 0.15, 0.35)
REPORT_PATH = os.path.join("docs", "debug", "clustering_verification_results.json")


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


def verification_cases() -> List[VerificationCase]:
    cases = []
    seed = 20260709
    for family in GRAPH_FAMILIES:
        for size in GRAPH_SIZES:
            for density in DENSITIES:
                cases.append(VerificationCase(family, size, density, seed))
                seed += 1
    return cases


def generate_sbm_graph(size: int, density: float, seed: int) -> Tuple[nx.Graph, Dict]:
    block_count = 4
    base = size // block_count
    block_sizes = [base] * block_count
    block_sizes[-1] += size - sum(block_sizes)
    intra = min(1.0, 2.0 * density)
    inter = density / 2.0
    probs = [
        [intra if i == j else inter for j in range(block_count)]
        for i in range(block_count)
    ]
    graph = nx.stochastic_block_model(block_sizes, probs, seed=seed)
    graph = nx.convert_node_labels_to_integers(nx.Graph(graph))
    return graph, {
        "model": "stochastic_block_model",
        "block_sizes": block_sizes,
        "intra_probability": intra,
        "inter_probability": inter,
    }


def _even_degree_from_density(size: int, density: float) -> int:
    k = max(2, int(round(density * (size - 1))))
    if k % 2 == 1:
        k += 1
    if k >= size:
        k = size - 1
        if k % 2 == 1:
            k -= 1
    return max(2, k)


def generate_watts_strogatz_graph(size: int, density: float, seed: int) -> Tuple[nx.Graph, Dict]:
    k = _even_degree_from_density(size, density)
    graph = nx.watts_strogatz_graph(size, k, 0.1, seed=seed)
    return graph, {
        "model": "watts_strogatz",
        "k": k,
        "rewiring_probability": 0.1,
    }


def _powerlaw_source_graph(size: int, density: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    source_size = max(size, 30)
    target_edges = max(size, int(density * source_size * (source_size - 1) / 2))
    for attempt in range(6):
        degrees = rng.zipf(2.5, source_size).astype(float)
        degrees = np.clip(degrees, 1.0, max(2.0, source_size ** 0.5))
        degrees *= (2.0 * target_edges) / degrees.sum()
        graph = nx.expected_degree_graph(degrees, seed=seed + attempt, selfloops=False)
        graph = nx.Graph(graph)
        if graph.number_of_edges() >= size:
            return graph
        source_size *= 2
        target_edges = max(size, int(density * source_size * (source_size - 1) / 2))
    return graph


def generate_powerlaw_line_graph(size: int, density: float, seed: int) -> Tuple[nx.Graph, Dict]:
    source = _powerlaw_source_graph(size, density, seed)
    line_graph = nx.line_graph(source)
    if line_graph.number_of_nodes() == 0:
        line_graph.add_node(0)

    if line_graph.number_of_nodes() > size:
        selected = sorted(line_graph.nodes(), key=repr)[:size]
        line_graph = line_graph.subgraph(selected).copy()

    graph = nx.convert_node_labels_to_integers(nx.Graph(line_graph))
    return graph, {
        "model": "line_graph_powerlaw_expected_degree",
        "source_nodes": source.number_of_nodes(),
        "source_edges": source.number_of_edges(),
        "source_density": nx.density(source),
        "line_graph_nodes_before_sample": nx.line_graph(source).number_of_nodes(),
        "line_graph_edges_before_sample": nx.line_graph(source).number_of_edges(),
    }


def generate_graph(family: str, size: int, density: float, seed: int) -> Tuple[nx.Graph, Dict]:
    if family == "sbm":
        return generate_sbm_graph(size, density, seed)
    if family == "powerlaw_line":
        return generate_powerlaw_line_graph(size, density, seed)
    if family == "watts_strogatz":
        return generate_watts_strogatz_graph(size, density, seed)
    raise ValueError(f"unknown graph family: {family}")


def _edge_label_maps(graph: nx.Graph, labels_A: np.ndarray, labels_B: np.ndarray):
    from tripartite import GraphManager

    graph_manager = GraphManager(graph)
    assembler = EdgePartitionAssembler(graph_manager)
    E12_edges, E23_edges = assembler.get_edge_lists()
    if len(labels_A) != len(E12_edges) or len(labels_B) != len(E23_edges):
        raise ValueError("partition label lengths do not match tripartite edge sets")

    by_middle = {
        node: {"A": {}, "B": {}}
        for node in graph.nodes()
    }
    for edge, label in zip(E12_edges, labels_A):
        a_node, middle_node = edge
        label = int(label)
        by_middle[middle_node[0]]["A"].setdefault(label, []).append(a_node[0])
    for edge, label in zip(E23_edges, labels_B):
        middle_node, b_node = edge
        label = int(label)
        by_middle[middle_node[0]]["B"].setdefault(label, []).append(b_node[0])
    return by_middle


def _cell_first_pass(graph: nx.Graph, by_middle: Dict) -> Dict[Tuple[int, int], Dict]:
    graph_adj = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    cells = {}
    for middle_data in by_middle.values():
        for label_A, A_nodes in middle_data["A"].items():
            if not A_nodes:
                continue
            for label_B, B_nodes in middle_data["B"].items():
                if not B_nodes:
                    continue
                key = (label_A, label_B)
                cell = cells.setdefault(
                    key,
                    {
                        "label_A": label_A,
                        "label_B": label_B,
                        "pathweight": 0,
                        "triangle_count": 0,
                        "_groups": [],
                    },
                )
                cell["pathweight"] += len(A_nodes) * len(B_nodes)
                triangle_count = 0
                for a in A_nodes:
                    neighbors = graph_adj.get(a, set())
                    triangle_count += sum(1 for b in B_nodes if b in neighbors)
                cell["triangle_count"] += triangle_count
                cell["_groups"].append((A_nodes, B_nodes))
    return cells


def _cell_deviation(graph: nx.Graph, groups: List[Tuple[List[int], List[int]]], gamma: float) -> float:
    graph_adj = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    deviation = 0.0
    for A_nodes, B_nodes in groups:
        if not A_nodes or not B_nodes:
            continue
        common_sum = 0
        for b in B_nodes:
            degree_into_A = sum(1 for a in A_nodes if b in graph_adj.get(a, set()))
            common_sum += degree_into_A * degree_into_A
        deviation += common_sum - (gamma**2) * len(B_nodes) * (len(A_nodes) ** 2)
    return float(deviation)


def compute_partition_deviation_report(
    graph: nx.Graph,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    parameters: AlgorithmParameters,
) -> Dict:
    by_middle = _edge_label_maps(graph, labels_A, labels_B)
    cells = _cell_first_pass(graph, by_middle)
    total_pathweight = sum(cell["pathweight"] for cell in cells.values())
    bad_pathweight = 0
    cell_reports = []

    for cell in cells.values():
        pathweight = int(cell["pathweight"])
        triangle_count = int(cell["triangle_count"])
        gamma = 0.0 if pathweight == 0 else triangle_count / pathweight
        deviation = _cell_deviation(graph, cell["_groups"], gamma)
        threshold = parameters.dev_threshold * pathweight
        low_gamma_success = pathweight > 0 and gamma < parameters.clustering_threshold
        large_deviation = (
            pathweight > 0
            and not low_gamma_success
            and deviation > threshold
        )
        if large_deviation:
            bad_pathweight += pathweight
        cell_reports.append(
            {
                "label_A": int(cell["label_A"]),
                "label_B": int(cell["label_B"]),
                "pathweight": pathweight,
                "triangle_count": triangle_count,
                "gamma": float(gamma),
                "global_deviation": float(deviation),
                "deviation_threshold": float(threshold),
                "clustering_threshold": float(parameters.clustering_threshold),
                "low_gamma_success": bool(low_gamma_success),
                "large_deviation": bool(large_deviation),
            }
        )

    allowed_bad_pathweight = parameters.dev_threshold * total_pathweight
    passes = bad_pathweight <= allowed_bad_pathweight
    worst_cells = sorted(
        cell_reports,
        key=lambda item: (item["large_deviation"], item["pathweight"], item["global_deviation"]),
        reverse=True,
    )[:10]
    return {
        "passes": bool(passes),
        "total_pathweight": int(total_pathweight),
        "bad_pathweight": int(bad_pathweight),
        "allowed_bad_pathweight": float(allowed_bad_pathweight),
        "bad_pathweight_ratio": 0.0 if total_pathweight == 0 else bad_pathweight / total_pathweight,
        "cell_count": len(cell_reports),
        "low_gamma_success_cell_count": sum(1 for cell in cell_reports if cell["low_gamma_success"]),
        "large_deviation_cell_count": sum(1 for cell in cell_reports if cell["large_deviation"]),
        "worst_cells": worst_cells,
        "cells": cell_reports,
    }


def run_algorithm_case(case: VerificationCase, eps: float = 0.1, max_depth: int = 8) -> Dict:
    start = time.monotonic()
    graph, generator_metadata = generate_graph(case.family, case.size, case.density, case.seed)
    parameters = AlgorithmParameters(eps=eps)

    with tempfile.TemporaryDirectory(prefix=f"verify_{case.case_id}_") as temp_dir:
        runner = AlgorithmRunner(
            graph,
            parameters=parameters,
            partition_dir=os.path.join(temp_dir, "partitions"),
            graph_dir=os.path.join(temp_dir, "graphs"),
            max_depth=max_depth,
        )
        labels_A, labels_B = runner.run()

    report = compute_partition_deviation_report(graph, labels_A, labels_B, runner.parameters)
    status = "success" if report["passes"] else "verification_failure"
    return {
        "case_id": case.case_id,
        "status": status,
        "runtime_seconds": time.monotonic() - start,
        "family": case.family,
        "requested_size": case.size,
        "actual_size": graph.number_of_nodes(),
        "requested_density": case.density,
        "actual_density": nx.density(graph),
        "edge_count": graph.number_of_edges(),
        "seed": case.seed,
        "eps": eps,
        "max_depth": max_depth,
        "generator": generator_metadata,
        "partition_count_A": int(len(set(labels_A.tolist()))) if len(labels_A) else 0,
        "partition_count_B": int(len(set(labels_B.tolist()))) if len(labels_B) else 0,
        "directions_considered": runner.directions_considered,
        "verification": report,
    }


def _worker(case_data, eps, max_depth, queue):
    case = VerificationCase(**case_data)
    try:
        queue.put(run_algorithm_case(case, eps=eps, max_depth=max_depth))
    except Exception as exc:
        status = "max_depth" if isinstance(exc, ValueError) and "exceeds maximum depth" in str(exc) else "algorithm_error"
        queue.put(
            {
                "case_id": case.case_id,
                "status": status,
                "family": case.family,
                "requested_size": case.size,
                "requested_density": case.density,
                "seed": case.seed,
                "eps": eps,
                "max_depth": max_depth,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_case_with_timeout(
    case: VerificationCase,
    timeout_seconds: float = 20.0,
    eps: float = 0.1,
    max_depth: int = 8,
) -> Dict:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(case.__dict__, eps, max_depth, queue),
    )
    start = time.monotonic()
    process.start()
    process.join(timeout_seconds)
    runtime = time.monotonic() - start

    if process.is_alive():
        process.terminate()
        process.join(5)
        return {
            "case_id": case.case_id,
            "status": "timeout",
            "runtime_seconds": runtime,
            "family": case.family,
            "requested_size": case.size,
            "requested_density": case.density,
            "seed": case.seed,
            "eps": eps,
            "max_depth": max_depth,
            "timeout_seconds": timeout_seconds,
        }

    if queue.empty():
        return {
            "case_id": case.case_id,
            "status": "algorithm_error",
            "runtime_seconds": runtime,
            "family": case.family,
            "requested_size": case.size,
            "requested_density": case.density,
            "seed": case.seed,
            "eps": eps,
            "max_depth": max_depth,
            "error": f"worker exited with code {process.exitcode} without returning a result",
        }

    result = queue.get()
    result.setdefault("runtime_seconds", runtime)
    return result


def write_verification_report(results: List[Dict], path: str = REPORT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ordered = sorted(results, key=lambda item: item["case_id"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=2)


def merge_verification_report(results: List[Dict], path: str = REPORT_PATH) -> None:
    existing = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)

    by_case_id = {item["case_id"]: item for item in existing}
    for result in results:
        by_case_id[result["case_id"]] = result
    write_verification_report(list(by_case_id.values()), path)


def rerun_size_cases(
    size: int,
    timeout_seconds: float,
    max_depth: int,
    eps: float = 0.1,
) -> List[Dict]:
    results = []
    for case in verification_cases():
        if case.size != size:
            continue
        print(f"RUN {case.case_id}", flush=True)
        result = run_case_with_timeout(
            case,
            timeout_seconds=timeout_seconds,
            eps=eps,
            max_depth=max_depth,
        )
        result["timeout_seconds"] = timeout_seconds
        result["rerun_label"] = f"n{size}_max_depth{max_depth}_timeout{int(timeout_seconds)}"
        results.append(result)
        print(
            f"  -> {result['status']} runtime={result.get('runtime_seconds', 0):.2f}s",
            flush=True,
        )
    merge_verification_report(results)
    return results


def main() -> None:
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Run clustering verification cases.")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--eps", type=float, default=0.1)
    args = parser.parse_args()

    results = rerun_size_cases(
        size=args.size,
        timeout_seconds=args.timeout,
        max_depth=args.max_depth,
        eps=args.eps,
    )
    print("status_counts", dict(Counter(item["status"] for item in results)))


if __name__ == "__main__":
    main()
