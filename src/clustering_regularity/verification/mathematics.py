from dataclasses import asdict
from typing import Dict, Optional, Sequence

import networkx as nx
import numpy as np

from .cells import _cell_counts, _iter_nonzero_cells
from .definition import (
    _adversarial_witness_search,
    _exhaustive_definition_check,
    _spectral_definition_check,
)
from .models import CellData, VerificationConfig
from .paper import _paper_diagnostics


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


def _compact_cell_report(cell: Dict) -> Dict:
    method = cell.get("definition_method") or {}
    diagnostics = cell.get("paper_diagnostics") or {}
    compact_method = {
        key: method[key]
        for key in (
            "method",
            "deviation_ratio_upper_bound",
            "certificate_margin",
            "required_pathweight",
        )
        if key in method
    }
    compact_diagnostics = {
        key: diagnostics[key]
        for key in (
            "triangle_irregular_vertex_count",
            "triangle_irregular_pathweight",
            "triangle_irregular_pathweight_limit",
            "deviation_vertex_count",
            "deviation_pathweight",
            "deviation_pathweight_limit",
        )
        if key in diagnostics
    }
    if diagnostics.get("worst_vertices"):
        compact_diagnostics["worst_vertices"] = diagnostics["worst_vertices"][:3]
    return {
        "label_A": cell["label_A"],
        "label_B": cell["label_B"],
        "pathweight": cell["pathweight"],
        "triangle_count": cell["triangle_count"],
        "gamma": cell["gamma"],
        "paper_status": cell["paper_status"],
        "paper_reason": cell["paper_reason"],
        "definition_status": cell["definition_status"],
        "definition_method": compact_method,
        "witness": cell.get("witness"),
        "paper_diagnostics": compact_diagnostics or None,
    }


def compute_partition_regularity_report(
    graph: nx.Graph,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    config: Optional[VerificationConfig] = None,
) -> Dict:
    config = config or VerificationConfig()
    labels_A = np.asarray(labels_A, dtype=int)
    labels_B = np.asarray(labels_B, dtype=int)
    unique_A = set(labels_A.tolist())
    unique_B = set(labels_B.tolist())
    total_cell_count = len(unique_A) * len(unique_B)
    total_pathweight = sum(graph.degree(node) ** 2 for node in graph.nodes())
    aggregate_cells = []
    prioritized_cells = []
    nonzero_cell_count = 0
    for cell in _iter_nonzero_cells(graph, labels_A, labels_B):
        cell_report = verify_cell(cell, config)
        nonzero_cell_count += 1
        aggregate_cells.append(
            {
                "pathweight": cell_report["pathweight"],
                "definition_status": cell_report["definition_status"],
                "paper_status": cell_report["paper_status"],
            }
        )
        prioritized_cells.append(cell_report)
        prioritized_cells.sort(
            key=lambda item: (
                {
                    "proven_irregular": 0,
                    "inconclusive": 1,
                    "certified_regular": 2,
                }[item["definition_status"]],
                -item["pathweight"],
                item["label_A"],
                item["label_B"],
            )
        )
        del prioritized_cells[config.max_report_cells :]

    definition = aggregate_definition_status(
        aggregate_cells,
        total_pathweight,
        config.eps,
    )
    paper = _aggregate_paper_status(
        aggregate_cells,
        total_pathweight,
        config.eps,
    )
    counts = {
        status: sum(
            1 for cell in aggregate_cells if cell["definition_status"] == status
        )
        for status in ("certified_regular", "proven_irregular", "inconclusive")
    }
    zero_cell_count = total_cell_count - nonzero_cell_count
    counts["certified_regular"] += zero_cell_count
    reported_cells = [_compact_cell_report(cell) for cell in prioritized_cells]
    return {
        **definition,
        **paper,
        "passes": definition["verification_status"] == "certified_success",
        "total_pathweight": total_pathweight,
        "cell_count": total_cell_count,
        "nonzero_cell_count": nonzero_cell_count,
        "zero_path_cell_count": zero_cell_count,
        "definition_status_counts": counts,
        "paper_thresholds": config.thresholds,
        "verification_config": asdict(config),
        "reported_cells": reported_cells,
        "omitted_cell_count": total_cell_count - len(reported_cells),
    }


