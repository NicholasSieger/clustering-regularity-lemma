from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import sparse

from ..config import PaperParameters


@dataclass
class LocalCheck:
    """Paper checks for one middle vertex and one edge-cell pair."""

    closure: sparse.csr_matrix
    parameters: PaperParameters

    def __post_init__(self) -> None:
        self.closure = self.closure.tocsr()
        self.size_a, self.size_b = self.closure.shape
        self.triangle_count = int(self.closure.sum())
        self.row_degrees = np.asarray(self.closure.sum(axis=1)).ravel()
        self.column_degrees = np.asarray(self.closure.sum(axis=0)).ravel()

    @property
    def pathweight(self) -> int:
        return self.size_a * self.size_b

    def irregular_edges(
        self,
        gamma: float,
    ) -> Tuple[np.ndarray, int, np.ndarray, int]:
        expected = gamma * self.size_b
        tolerance = self.parameters.delta_3 * self.size_b
        positive = self.row_degrees - expected > tolerance
        negative = self.row_degrees - expected < -tolerance
        return (
            positive,
            int(np.sum(positive)),
            negative,
            int(np.sum(negative)),
        )

    def deviation(self, gamma: float) -> float:
        # sum(M M^T) = sum_b deg_B(b)^2
        observed = float(np.dot(self.column_degrees, self.column_degrees))
        expected = gamma**2 * self.size_a**2 * self.size_b
        return observed - expected

    def deviation_split(self, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.size_a == 0 or self.size_b == 0:
            return (
                np.zeros(self.size_a, dtype=bool),
                np.zeros(self.size_b, dtype=bool),
            )

        candidate_rows = (
            np.abs(self.row_degrees - gamma * self.size_b)
            < self.parameters.delta_1 * self.size_b
        )
        candidates = np.flatnonzero(candidate_rows)
        if candidates.size == 0:
            candidates = np.arange(self.size_a)

        # Row sums of M M^T can be computed as M times the column degrees.
        common_neighbor_scores = np.asarray(
            self.closure @ self.column_degrees
        ).ravel()
        centered_scores = (
            common_neighbor_scores
            - gamma**2 * self.size_a * self.size_b
        )
        pivot = int(candidates[np.argmax(centered_scores[candidates])])

        pivot_row = self.closure[pivot : pivot + 1, :]
        common_neighbor_column = self.closure @ pivot_row.T
        common_neighbors = np.asarray(
            common_neighbor_column.toarray()
        ).ravel()
        split_a = common_neighbors > self.parameters.delta_6 * self.size_b
        split_b = np.asarray(pivot_row.toarray()).ravel() > 0
        return split_a, split_b
