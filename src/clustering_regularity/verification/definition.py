import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .cells import _selection_counts, _witness_report
from .models import CellData, VerificationConfig

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


