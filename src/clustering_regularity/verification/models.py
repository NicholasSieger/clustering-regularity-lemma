from dataclasses import dataclass
from typing import Dict, List

import numpy as np


GRAPH_FAMILIES = ("sbm", "powerlaw_line", "watts_strogatz")
GRAPH_SIZES = (30, 100, 500)
DENSITIES = (0.05, 0.15, 0.35)
REPORT_SCHEMA_VERSION = 3


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


ADDITIONAL_SBM_CASE = VerificationCase("sbm", 48, 0.35, 20260801)
SCALED_ADDITIONAL_CASES = (
    VerificationCase("powerlaw_line", 36, 0.15, 20260901),
    VerificationCase("watts_strogatz", 42, 0.15, 20260902),
    VerificationCase("sbm", 60, 0.15, 20260903),
    VerificationCase("sbm_near_epsilon", 60, 0.06, 20260904),
)


@dataclass(frozen=True)
class VerificationConfig:
    eps: float = 0.05
    exhaustive_edge_limit: int = 18
    svd_dense_limit: int = 256
    density_tolerance: float = 0.02
    numerical_tolerance: float = 1e-12
    witness_search: bool = True
    search_restarts: int = 12
    coordinate_edge_limit: int = 64
    coordinate_passes: int = 8
    max_report_cells: int = 20

    def __post_init__(self) -> None:
        if not 0 < self.eps < 1 / 16:
            raise ValueError("paper-faithful verification requires 0 < eps < 1/16")
        if self.exhaustive_edge_limit < 0:
            raise ValueError("exhaustive_edge_limit must be nonnegative")
        if self.svd_dense_limit < 1:
            raise ValueError("svd_dense_limit must be positive")
        if self.density_tolerance < 0:
            raise ValueError("density_tolerance must be nonnegative")
        if self.max_report_cells < 0:
            raise ValueError("max_report_cells must be nonnegative")

    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "epsilon": self.eps,
            "delta_1": self.eps**2 / 9,
            "delta_2": 2 * self.eps**2 / 5,
            "delta_3": self.eps**5 / 90,
            "delta_4": self.eps ** (5 / 2) / 9,
            "delta_5": 2 * self.eps ** (5 / 2) / 5,
            "delta_6": self.eps**5,
        }


@dataclass
class CellGroup:
    middle: int
    a_edge_indices: np.ndarray
    b_edge_indices: np.ndarray
    a_nodes: np.ndarray
    b_nodes: np.ndarray
    closure: np.ndarray


@dataclass
class CellData:
    label_A: int
    label_B: int
    groups: List[CellGroup]

    @property
    def active_a_indices(self) -> List[int]:
        return sorted(
            {
                int(index)
                for group in self.groups
                for index in group.a_edge_indices.tolist()
            }
        )

    @property
    def active_b_indices(self) -> List[int]:
        return sorted(
            {
                int(index)
                for group in self.groups
                for index in group.b_edge_indices.tolist()
            }
        )


def verification_cases() -> List[VerificationCase]:
    cases = []
    seed = 20260709
    for family in GRAPH_FAMILIES:
        for size in GRAPH_SIZES:
            for density in DENSITIES:
                cases.append(VerificationCase(family, size, density, seed))
                seed += 1
    return cases

