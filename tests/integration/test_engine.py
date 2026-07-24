import json

import networkx as nx
import numpy as np
import pytest

from clustering_regularity import run_graph


@pytest.mark.integration
def test_zero_edge_graph(tmp_path):
    result = run_graph(nx.empty_graph(5), tmp_path / "run")
    labels_a = result.labels("A")
    labels_b = result.labels("B")
    try:
        assert labels_a.shape == labels_b.shape == (0,)
        assert result.part_count_a == result.part_count_b == 0
        assert result.total_pathweight == 0
        assert len(result.rounds) == 1
    finally:
        labels_a._mmap.close()
        labels_b._mmap.close()


@pytest.mark.integration
def test_sparse_random_characterization(tmp_path):
    graph = nx.gnp_random_graph(8, 0.3, seed=1)
    result = run_graph(graph, tmp_path / "run", max_rounds=40)
    labels_a = result.labels("A")
    labels_b = result.labels("B")
    try:
        assert labels_a.tolist() == [
            0, 0, 1, 2, 3, 0, 0, 2, 4, 2, 5,
            1, 3, 2, 2, 2, 1, 3, 2, 2, 2, 2,
        ]
        assert labels_b.tolist() == [
            0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 4,
            1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0,
        ]
    finally:
        labels_a._mmap.close()
        labels_b._mmap.close()
    assert result.part_count_a == 6
    assert result.part_count_b == 5
    assert [summary.failed_cell_count for summary in result.rounds] == [
        1, 2, 2, 2, 0
    ]
    assert [summary.failed_pathweight for summary in result.rounds] == [
        64, 64, 26, 9, 0
    ]


@pytest.mark.integration
def test_compact_result_contains_rounds_not_cells(tmp_path):
    result = run_graph(nx.path_graph(6), tmp_path / "run")
    report_path = result.workspace / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.stat().st_size < 10_000
    assert "rounds" in report
    assert "cells" not in report
    assert "directions_considered" not in report


@pytest.mark.integration
def test_max_rounds_is_reported_separately(tmp_path):
    graph = nx.gnp_random_graph(8, 0.3, seed=1)

    with pytest.raises(RuntimeError, match="max_rounds"):
        run_graph(graph, tmp_path / "run", max_rounds=0)

