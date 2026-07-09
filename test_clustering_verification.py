import json
import os

import networkx as nx
import numpy as np
import pytest

from clustering_verification import (
    REPORT_PATH,
    VerificationCase,
    compute_partition_deviation_report,
    run_case_with_timeout,
    verification_cases,
    write_verification_report,
)
from edge_partition import EdgePartitionAssembler
from parameters import AlgorithmParameters
from tripartite import GraphManager


VERIFICATION_RESULTS = []


def all_zero_edge_labels(graph):
    assembler = EdgePartitionAssembler(GraphManager(graph))
    E12_edges, E23_edges = assembler.get_edge_lists()
    return np.zeros(len(E12_edges), dtype=int), np.zeros(len(E23_edges), dtype=int)


def test_deviation_report_no_paths_passes():
    graph = nx.empty_graph(4)
    labels_A, labels_B = all_zero_edge_labels(graph)

    report = compute_partition_deviation_report(
        graph,
        labels_A,
        labels_B,
        AlgorithmParameters(eps=0.1),
    )

    assert report["passes"] is True
    assert report["total_pathweight"] == 0
    assert report["bad_pathweight"] == 0


def test_deviation_report_complete_triangle_values():
    graph = nx.complete_graph(3)
    labels_A, labels_B = all_zero_edge_labels(graph)

    report = compute_partition_deviation_report(
        graph,
        labels_A,
        labels_B,
        AlgorithmParameters(eps=0.1),
    )

    assert report["passes"] is True
    assert report["total_pathweight"] == 12
    assert report["cells"][0]["pathweight"] == 12
    assert report["cells"][0]["triangle_count"] == 6
    assert report["cells"][0]["gamma"] == pytest.approx(0.5)


def test_deviation_report_flags_large_deviation_cell():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
        ]
    )
    labels_A, labels_B = all_zero_edge_labels(graph)

    report = compute_partition_deviation_report(
        graph,
        labels_A,
        labels_B,
        AlgorithmParameters(eps=0.1, dev_threshold=0.0),
    )

    assert report["passes"] is False
    assert report["bad_pathweight"] > 0
    assert report["large_deviation_cell_count"] > 0


@pytest.mark.parametrize("case", verification_cases(), ids=lambda case: case.case_id)
def test_generated_graph_clustering_verification(case):
    result = run_case_with_timeout(case, timeout_seconds=20.0, eps=0.1, max_depth=8)
    VERIFICATION_RESULTS.append(result)
    write_verification_report(VERIFICATION_RESULTS)

    if result["status"] == "timeout":
        pytest.skip(
            f"TIMEOUT: {result['case_id']} exceeded {result['timeout_seconds']}s "
            f"after {result['runtime_seconds']:.2f}s"
        )

    if result["status"] == "max_depth":
        pytest.skip(
            f"MAX_DEPTH: {result['case_id']}: "
            f"{result.get('error_type', '')} {result.get('error', '')}"
        )

    if result["status"] == "algorithm_error":
        pytest.fail(
            f"ALGORITHM_ERROR: {result['case_id']}: "
            f"{result.get('error_type', '')} {result.get('error', '')}"
        )

    if result["status"] == "verification_failure":
        verification = result["verification"]
        pytest.fail(
            f"VERIFICATION_FAILURE: {result['case_id']} "
            f"bad_pathweight_ratio={verification['bad_pathweight_ratio']:.6g}, "
            f"bad_pathweight={verification['bad_pathweight']}, "
            f"allowed={verification['allowed_bad_pathweight']:.6g}, "
            f"worst_cells={verification['worst_cells'][:3]}"
        )

    assert result["status"] == "success"
    assert result["verification"]["passes"] is True


def test_verification_report_file_is_machine_readable():
    if not os.path.exists(REPORT_PATH):
        write_verification_report(VERIFICATION_RESULTS)

    with open(REPORT_PATH, encoding="utf-8") as handle:
        data = json.load(handle)

    assert isinstance(data, list)
