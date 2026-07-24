from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True, order=True)
class CellId:
    label_a: int
    label_b: int


class CellStatus(str, Enum):
    REGULAR_ZERO_PATH = "regular_zero_path"
    REGULAR_LOW_GAMMA = "regular_low_gamma"
    REGULAR = "regular"
    IRREGULAR_TRIANGLE_DEGREES = "irregular_triangle_degrees"
    IRREGULAR_DEVIATION = "irregular_deviation"


@dataclass
class CellMetrics:
    pathweight: int = 0
    triangle_count: int = 0

    @property
    def gamma(self) -> float:
        return (
            0.0
            if self.pathweight == 0
            else self.triangle_count / self.pathweight
        )


@dataclass(frozen=True)
class RefinementCut:
    parent_label: int
    edge_indices: Tuple[int, ...] = ()
    edge_indices_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.edge_indices and self.edge_indices_path is not None:
            raise ValueError(
                "a refinement cut must use inline indices or an index file"
            )


@dataclass
class CellEvaluation:
    cell: CellId
    metrics: CellMetrics
    status: CellStatus
    irregular_pathweight: int = 0
    deviation_pathweight: int = 0

    @property
    def requires_refinement(self) -> bool:
        return self.status in {
            CellStatus.IRREGULAR_TRIANGLE_DEGREES,
            CellStatus.IRREGULAR_DEVIATION,
        }


@dataclass
class RoundSummary:
    round_index: int
    generation: int
    part_count_a: int
    part_count_b: int
    evaluated_nonzero_cells: int
    total_cell_count: int
    failed_cell_count: int
    total_pathweight: int
    failed_pathweight: int
    branch_counts: Dict[str, int]
    elapsed_seconds: float

    @property
    def failed_pathweight_ratio(self) -> float:
        return (
            0.0
            if self.total_pathweight == 0
            else self.failed_pathweight / self.total_pathweight
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["failed_pathweight_ratio"] = self.failed_pathweight_ratio
        return result


@dataclass
class RunResult:
    workspace: Path
    generation: int
    labels_a_path: Path
    labels_b_path: Path
    part_count_a: int
    part_count_b: int
    total_pathweight: int
    rounds: Sequence[RoundSummary] = field(default_factory=tuple)

    def labels(self, side: str) -> np.ndarray:
        normalized = side.upper()
        if normalized == "A":
            return np.load(self.labels_a_path, mmap_mode="r")
        if normalized == "B":
            return np.load(self.labels_b_path, mmap_mode="r")
        raise ValueError("side must be 'A' or 'B'")

    def to_summary_dict(self) -> dict:
        return {
            "generation": self.generation,
            "part_count_A": self.part_count_a,
            "part_count_B": self.part_count_b,
            "total_pathweight": self.total_pathweight,
            "round_count": len(self.rounds),
            "labels_A": str(self.labels_a_path.relative_to(self.workspace)),
            "labels_B": str(self.labels_b_path.relative_to(self.workspace)),
            "rounds": [round_summary.to_dict() for round_summary in self.rounds],
        }
