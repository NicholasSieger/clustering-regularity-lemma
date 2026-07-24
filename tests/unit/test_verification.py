import networkx as nx
import numpy as np
import pytest

from clustering_regularity.verification import (
    aggregate_definition_status,
    compute_partition_regularity_report,
    make_cell_group,
    validate_witness,
    verify_cell,
)
from clustering_regularity.verification.models import (
    CellData,
    VerificationConfig,
)


def _cell(*groups):
    return CellData(label_A=0, label_B=0, groups=list(groups))


def test_zero_path_partition_is_vacuously_certified():
    graph = nx.empty_graph(4)
    labels = np.asarray([], dtype=int)

    report = compute_partition_regularity_report(graph, labels, labels)

    assert report["verification_status"] == "certified_success"
    assert report["paper_partition_status"] == "regular"
    assert report["total_pathweight"] == 0
    assert report["cell_count"] == 0


def test_tripartite_counts_are_exact():
    graph = nx.complete_graph(3)
    labels = np.zeros(6, dtype=int)

    report = compute_partition_regularity_report(graph, labels, labels)
    cell = report["reported_cells"][0]

    assert report["total_pathweight"] == 12
    assert cell["pathweight"] == 12
    assert cell["triangle_count"] == 6
    assert cell["gamma"] == pytest.approx(0.5)


def test_zero_path_label_pairs_are_counted_without_report_bloat():
    graph = nx.complete_graph(3)
    labels = np.asarray([0, 0, 1, 1, 1, 1], dtype=int)

    report = compute_partition_regularity_report(graph, labels, labels)

    assert report["cell_count"] == 4
    assert report["nonzero_cell_count"] == 2
    assert report["zero_path_cell_count"] == 2


def test_signed_deviations_do_not_cancel():
    cell = _cell(
        make_cell_group(np.ones((2, 2)), middle=0),
        make_cell_group(np.zeros((2, 2)), middle=1, a_offset=2, b_offset=2),
    )

    report = verify_cell(cell, VerificationConfig())
    diagnostics = report["paper_diagnostics"]

    assert diagnostics["deviation_vertex_count"] == 1
    sigmas = [item["sigma"] for item in diagnostics["worst_vertices"]]
    assert any(value > 0 for value in sigmas)
    assert any(value < 0 for value in sigmas)


def test_low_gamma_paper_pass_can_be_literal_failure():
    closure = np.zeros((9, 9))
    closure[0, :4] = 1
    cell = _cell(make_cell_group(closure))

    report = verify_cell(cell, VerificationConfig())

    assert report["gamma"] == pytest.approx(4 / 81)
    assert report["paper_status"] == "regular_low_gamma"
    assert report["definition_status"] == "proven_irregular"
    assert validate_witness(cell, report["witness"], 0.05)


def test_exact_and_spectral_checks_certify_complete_cell():
    cell = _cell(make_cell_group(np.ones((3, 3))))

    exact = verify_cell(cell, VerificationConfig(exhaustive_edge_limit=6))
    spectral = verify_cell(cell, VerificationConfig(exhaustive_edge_limit=5))

    assert exact["definition_status"] == "certified_regular"
    assert exact["definition_method"]["method"] == "exhaustive"
    assert spectral["definition_status"] == "certified_regular"
    assert spectral["definition_method"]["method"] == "spectral_water_filling"


def test_loose_spectral_bound_is_inconclusive():
    cell = _cell(make_cell_group(np.eye(10)))

    report = verify_cell(
        cell,
        VerificationConfig(exhaustive_edge_limit=18, witness_search=False),
    )

    assert report["definition_status"] == "inconclusive"
    assert report["witness"] is None


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        (
            [
                {"pathweight": 96, "definition_status": "certified_regular"},
                {"pathweight": 4, "definition_status": "inconclusive"},
            ],
            "certified_success",
        ),
        (
            [
                {"pathweight": 95, "definition_status": "certified_regular"},
                {"pathweight": 5, "definition_status": "proven_irregular"},
            ],
            "proven_failure",
        ),
        (
            [
                {"pathweight": 94, "definition_status": "certified_regular"},
                {"pathweight": 4, "definition_status": "proven_irregular"},
                {"pathweight": 2, "definition_status": "inconclusive"},
            ],
            "inconclusive",
        ),
    ],
)
def test_partition_aggregation_has_three_states(cells, expected):
    report = aggregate_definition_status(cells, 100, 0.05)
    assert report["verification_status"] == expected
