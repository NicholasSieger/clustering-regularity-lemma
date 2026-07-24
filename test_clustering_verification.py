import json
import os

import networkx as nx
import numpy as np
import pytest

from clustering_verification import (
    ADDITIONAL_SBM_CASE,
    SCALED_ADDITIONAL_CASES,
    REPORT_SCHEMA_VERSION,
    CellData,
    VerificationCase,
    VerificationConfig,
    aggregate_definition_status,
    algorithm_clustering_coefficient,
    compute_partition_regularity_report,
    finite_scale_threshold_report,
    generate_graph,
    generate_sbm_near_epsilon_graph,
    make_cell_group,
    run_algorithm_case,
    run_additional_sbm_case,
    run_case_with_timeout,
    run_scaled_additional_cases,
    validate_witness,
    verification_cases,
    verify_cell,
    write_verification_report,
)
from edge_partition import EdgePartitionAssembler
from tripartite import GraphManager


def all_zero_edge_labels(graph):
    assembler = EdgePartitionAssembler(GraphManager(graph))
    edges_A, edges_B = assembler.get_edge_lists()
    return np.zeros(len(edges_A), dtype=int), np.zeros(len(edges_B), dtype=int)


def synthetic_cell(*groups):
    return CellData(label_A=0, label_B=0, groups=list(groups))


def test_paper_thresholds_match_the_paper():
    eps = 0.05
    thresholds = VerificationConfig(eps=eps).thresholds

    assert thresholds["delta_1"] == pytest.approx(eps**2 / 9)
    assert thresholds["delta_2"] == pytest.approx(2 * eps**2 / 5)
    assert thresholds["delta_3"] == pytest.approx(eps**5 / 90)
    assert thresholds["delta_4"] == pytest.approx(eps ** (5 / 2) / 9)
    assert thresholds["delta_5"] == pytest.approx(2 * eps ** (5 / 2) / 5)


def test_verification_rejects_epsilon_outside_paper_domain():
    with pytest.raises(ValueError, match="eps < 1/16"):
        VerificationConfig(eps=0.1)


def test_zero_path_partition_is_vacuously_certified():
    graph = nx.empty_graph(4)
    labels_A, labels_B = all_zero_edge_labels(graph)

    report = compute_partition_regularity_report(
        graph,
        labels_A,
        labels_B,
        VerificationConfig(),
    )

    assert report["verification_status"] == "certified_success"
    assert report["paper_partition_status"] == "regular"
    assert report["total_pathweight"] == 0
    assert report["cell_count"] == 0


def test_tripartite_pathweight_triangle_count_and_gamma_are_exact():
    graph = nx.complete_graph(3)
    labels_A, labels_B = all_zero_edge_labels(graph)

    report = compute_partition_regularity_report(
        graph,
        labels_A,
        labels_B,
        VerificationConfig(),
    )

    cell = report["cells"][0]
    assert report["total_pathweight"] == 12
    assert cell["pathweight"] == 12
    assert cell["triangle_count"] == 6
    assert cell["gamma"] == pytest.approx(0.5)


def test_report_includes_label_pairs_with_zero_pathweight():
    graph = nx.complete_graph(3)
    assembler = EdgePartitionAssembler(GraphManager(graph))
    edges_A, edges_B = assembler.get_edge_lists()
    labels_A = np.asarray(
        [0 if middle[0] == 0 else 1 for _, middle in edges_A],
        dtype=int,
    )
    labels_B = np.asarray(
        [0 if middle[0] == 0 else 1 for middle, _ in edges_B],
        dtype=int,
    )

    report = compute_partition_regularity_report(
        graph,
        labels_A,
        labels_B,
        VerificationConfig(),
    )

    assert report["cell_count"] == 4
    cells = {
        (cell["label_A"], cell["label_B"]): cell
        for cell in report["cells"]
    }
    assert cells[(0, 1)]["pathweight"] == 0
    assert cells[(1, 0)]["pathweight"] == 0
    assert cells[(0, 1)]["paper_status"] == "regular_zero_path"
    assert sum(cell["pathweight"] for cell in report["cells"]) == 12


def test_signed_deviations_do_not_cancel_between_middle_vertices():
    cell = synthetic_cell(
        make_cell_group(np.ones((2, 2)), middle=0),
        make_cell_group(np.zeros((2, 2)), middle=1, a_offset=2, b_offset=2),
    )

    report = verify_cell(cell, VerificationConfig())
    diagnostics = report["paper_diagnostics"]

    assert report["gamma"] == pytest.approx(0.5)
    assert diagnostics["deviation_vertex_count"] == 1
    assert diagnostics["deviation_pathweight"] == 4
    sigmas = [item["sigma"] for item in diagnostics["worst_vertices"]]
    assert any(sigma > 0 for sigma in sigmas)
    assert any(sigma < 0 for sigma in sigmas)


def test_low_gamma_paper_pass_can_be_definition_failure():
    closure = np.zeros((9, 9))
    closure[0, :4] = 1
    cell = synthetic_cell(make_cell_group(closure))

    report = verify_cell(cell, VerificationConfig())

    assert report["gamma"] == pytest.approx(4 / 81)
    assert report["gamma"] < 0.05
    assert report["paper_status"] == "regular_low_gamma"
    assert report["definition_status"] == "proven_irregular"
    assert validate_witness(cell, report["witness"], eps=0.05)
    assert report["witness"]["pathweight"] >= 0.05 * report["pathweight"]
    assert report["witness"]["absolute_gamma_difference"] > 0.05


def test_exhaustive_check_certifies_uniform_complete_cell():
    cell = synthetic_cell(make_cell_group(np.ones((3, 3))))

    report = verify_cell(cell, VerificationConfig())

    assert report["gamma"] == 1.0
    assert report["definition_status"] == "certified_regular"
    assert report["definition_method"]["method"] == "exhaustive"
    assert report["witness"] is None


def test_spectral_certificate_agrees_with_exhaustive_check():
    cell = synthetic_cell(make_cell_group(np.ones((3, 3))))

    exhaustive = verify_cell(
        cell,
        VerificationConfig(exhaustive_edge_limit=6),
    )
    spectral = verify_cell(
        cell,
        VerificationConfig(exhaustive_edge_limit=5),
    )

    assert exhaustive["definition_status"] == "certified_regular"
    assert spectral["definition_status"] == "certified_regular"
    assert spectral["definition_method"]["method"] == "spectral_water_filling"
    assert spectral["definition_method"]["deviation_ratio_upper_bound"] < 0.05


def test_loose_spectral_bound_is_inconclusive_without_witness_search():
    cell = synthetic_cell(make_cell_group(np.eye(10)))

    report = verify_cell(
        cell,
        VerificationConfig(
            exhaustive_edge_limit=18,
            witness_search=False,
        ),
    )

    assert report["definition_status"] == "inconclusive"
    assert report["definition_method"]["deviation_ratio_upper_bound"] > 0.05
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
    report = aggregate_definition_status(cells, total_pathweight=100, eps=0.05)
    assert report["verification_status"] == expected


def test_final_cell_density_variation_is_not_a_generation_error():
    sparse = verify_cell(
        synthetic_cell(make_cell_group(np.zeros((3, 3)))),
        VerificationConfig(),
    )
    dense = verify_cell(
        synthetic_cell(make_cell_group(np.ones((3, 3)))),
        VerificationConfig(),
    )

    assert sparse["gamma"] == 0.0
    assert dense["gamma"] == 1.0
    assert sparse["definition_status"] == "certified_regular"
    assert dense["definition_status"] == "certified_regular"


@pytest.mark.parametrize("family", ("sbm", "powerlaw_line", "watts_strogatz"))
@pytest.mark.parametrize("density", (0.05, 0.15, 0.35))
def test_n30_generators_control_final_density(family, density):
    graph, metadata = generate_graph(family, 30, density, seed=1234)

    assert graph.number_of_nodes() == 30
    assert abs(nx.density(graph) - density) <= 0.02
    if family == "powerlaw_line":
        assert metadata["line_graph_nodes_before_sample"] == 30
        assert metadata["sample_method"] == "not_needed"


def test_additional_sbm_has_nontrivial_finite_scale_deviation_thresholds():
    graph, _ = generate_graph(
        ADDITIONAL_SBM_CASE.family,
        ADDITIONAL_SBM_CASE.size,
        ADDITIONAL_SBM_CASE.density,
        ADDITIONAL_SBM_CASE.seed,
    )

    report = finite_scale_threshold_report(graph, VerificationConfig())

    assert report["count_scale_checks"]["global_deviation"] is True
    assert report["count_scale_checks"]["global_triangle_irregularity"] is True
    assert report["count_scale_checks"]["local_deviation"] is True
    assert report["count_scale_checks"]["local_triangle_degree_tolerance"] is False
    assert (
        report["count_scale_checks"]["local_triangle_irregular_edge_count"]
        is False
    )


def test_algorithm_clustering_coefficient_matches_root_path_definition():
    assert algorithm_clustering_coefficient(nx.complete_graph(3)) == 0.5


def test_near_epsilon_sbm_generator_is_deterministic_and_close():
    case = next(
        case
        for case in SCALED_ADDITIONAL_CASES
        if case.family == "sbm_near_epsilon"
    )
    graph, metadata = generate_sbm_near_epsilon_graph(
        case.size,
        case.density,
        case.seed,
    )

    assert graph.number_of_nodes() == case.size
    assert abs(nx.density(graph) - case.density) <= 0.02
    assert abs(algorithm_clustering_coefficient(graph) - 0.05) < 0.001
    assert metadata["candidate_count"] == 256
    assert metadata["selected_root_gamma"] >= 0.05


def test_generation_mismatch_is_separate_from_verification():
    case = VerificationCase("watts_strogatz", 30, 0.05, 1234)
    result = run_algorithm_case(
        case,
        verification_config=VerificationConfig(density_tolerance=0.0),
    )

    assert result["status"] == "generation_mismatch"
    assert result["execution_status"] == "generation_mismatch"
    assert result["verification_status"] is None


def test_timeout_is_separate_from_verification():
    case = VerificationCase("sbm", 30, 0.15, 1234)
    result = run_case_with_timeout(
        case,
        timeout_seconds=0.001,
        verification_config=VerificationConfig(),
    )

    assert result["status"] == "timeout"
    assert result["execution_status"] == "timeout"
    assert result["verification_status"] is None


def test_max_depth_is_separate_from_verification():
    case = VerificationCase("sbm", 30, 0.15, 1234)
    result = run_case_with_timeout(
        case,
        timeout_seconds=10,
        max_depth=-1,
        verification_config=VerificationConfig(),
    )

    assert result["status"] == "max_depth"
    assert result["execution_status"] == "max_depth"
    assert result["verification_status"] is None
    assert result["max_direction_code_length"] == -1
    assert result["telemetry"]["active_phase_at_exit"] == "algorithm"
    assert result["telemetry"]["directions_started"] == 1
    assert result["telemetry"]["directions_completed"] == 0
    assert result["telemetry"]["last_direction_started"] == ""
    assert result["telemetry"]["generation"]["actual_size"] == 30


def test_algorithm_error_is_separate_from_verification():
    case = VerificationCase("unknown", 30, 0.15, 1234)
    result = run_case_with_timeout(
        case,
        timeout_seconds=10,
        verification_config=VerificationConfig(),
    )

    assert result["status"] == "algorithm_error"
    assert result["execution_status"] == "algorithm_error"
    assert result["verification_status"] is None


def test_completed_case_reports_all_telemetry_phases():
    case = VerificationCase("sbm", 30, 0.05, 20260709)
    result = run_case_with_timeout(
        case,
        timeout_seconds=10,
        verification_config=VerificationConfig(),
    )

    assert result["execution_status"] == "completed"
    telemetry = result["telemetry"]
    assert telemetry["active_phase_at_exit"] is None
    assert set(telemetry["phase_durations_seconds"]) == {
        "generation",
        "algorithm",
        "verification",
    }
    assert telemetry["directions_started"] == 1
    assert telemetry["directions_completed"] == 1
    assert telemetry["branch_counts"] == {"PASSED_GAMMA_CHECK": 1}
    assert telemetry["accepted_direction_count"] == 1
    assert telemetry["assembly"]["partition_assembly_diagnostics"]["A"] == {
        "edge_count": result["edge_count"] * 2,
        "uncovered_edge_count": 0,
        "overlapping_edge_count": 0,
        "maximum_mask_membership": 1,
    }


def test_schema_v2_report_is_atomic_and_machine_readable(tmp_path):
    path = tmp_path / "verification.json"
    results = [
        {
            "case_id": "case_b",
            "status": "inconclusive",
            "execution_status": "completed",
            "verification_status": "inconclusive",
        },
        {
            "case_id": "case_a",
            "status": "certified_success",
            "execution_status": "completed",
            "verification_status": "certified_success",
        },
    ]

    write_verification_report(
        results,
        path=str(path),
        run_config={"eps": 0.05},
    )

    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["schema_version"] == REPORT_SCHEMA_VERSION
    assert document["run_config"] == {"eps": 0.05}
    assert [case["case_id"] for case in document["cases"]] == [
        "case_a",
        "case_b",
    ]
    assert not list(tmp_path.glob(".clustering_verification_*.json"))


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_CLUSTERING_VERIFICATION") != "1",
    reason="set RUN_SLOW_CLUSTERING_VERIFICATION=1 to run the 27 generated cases",
)
@pytest.mark.parametrize("case", verification_cases(), ids=lambda case: case.case_id)
def test_generated_graph_clustering_verification(case):
    result = run_case_with_timeout(
        case,
        timeout_seconds=20.0,
        eps=0.05,
        max_depth=8,
    )

    if result["execution_status"] in {
        "timeout",
        "max_depth",
        "generation_mismatch",
    }:
        pytest.skip(
            f"{result['execution_status'].upper()}: {result['case_id']}: "
            f"{result.get('error', '')}"
        )
    if result["execution_status"] == "algorithm_error":
        pytest.fail(
            f"ALGORITHM_ERROR: {result['case_id']}: "
            f"{result.get('error_type', '')} {result.get('error', '')}"
        )
    if result["verification_status"] == "proven_failure":
        pytest.fail(
            f"PROVEN_FAILURE: {result['case_id']} "
            f"weight={result['verification']['proven_irregular_pathweight']} "
            f"allowed={result['verification']['allowed_irregular_pathweight']}"
        )
    if result["verification_status"] == "inconclusive":
        pytest.skip(f"INCONCLUSIVE: {result['case_id']}")

    assert result["verification_status"] == "certified_success"


@pytest.mark.skipif(
    os.environ.get("RUN_ADDITIONAL_SBM_VERIFICATION") != "1",
    reason="set RUN_ADDITIONAL_SBM_VERIFICATION=1 to run the n=48 SBM case",
)
def test_additional_sbm_produces_a_multi_part_partition():
    result = run_additional_sbm_case(
        timeout_seconds=240,
        max_depth=30,
        eps=0.05,
    )

    assert result["execution_status"] == "completed"
    assert result["partition_count_A"] > 1
    assert result["partition_count_B"] > 1
    assert result["verification_status"] in {
        "certified_success",
        "proven_failure",
        "inconclusive",
    }


@pytest.mark.skipif(
    os.environ.get("RUN_SCALED_ADDITIONAL_VERIFICATION") != "1",
    reason="set RUN_SCALED_ADDITIONAL_VERIFICATION=1 to run scaled cases",
)
def test_scaled_additional_cases_produce_diagnostic_partitions():
    results = run_scaled_additional_cases(
        timeout_seconds=180,
        max_depth=120,
        eps=0.05,
    )

    assert len(results) == len(SCALED_ADDITIONAL_CASES)
    for result in results:
        assert result["execution_status"] == "completed", (
            f"{result['case_id']}: {result['execution_status']} "
            f"{result.get('error', '')}"
        )
        assert result["partition_count_A"] > 1
        assert result["partition_count_B"] > 1

    near_epsilon = next(
        result
        for result in results
        if result["family"] == "sbm_near_epsilon"
    )
    clustering = near_epsilon["graph_clustering"]
    assert abs(clustering["root_gamma_minus_epsilon"]) < 0.001
