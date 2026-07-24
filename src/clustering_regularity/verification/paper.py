from typing import Dict, List, Set, Tuple

import numpy as np

from .models import CellData, VerificationConfig

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


