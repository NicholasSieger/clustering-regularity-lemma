from dataclasses import dataclass
from typing import Dict, List

from ..config import PaperParameters
from .models import (
    CellEvaluation,
    CellId,
    CellMetrics,
    CellStatus,
)


@dataclass
class CellAccumulator:
    metrics: CellMetrics
    irregular_pathweight: int = 0
    deviation_pathweight: int = 0
    positive_pathweight: int = 0
    negative_pathweight: int = 0


def evaluate_cells(
    metrics: Dict[CellId, CellMetrics],
    accumulators: Dict[CellId, CellAccumulator],
    parameters: PaperParameters,
) -> List[CellEvaluation]:
    evaluations = []
    for cell, cell_metrics in metrics.items():
        if cell_metrics.gamma < parameters.epsilon:
            evaluations.append(
                CellEvaluation(
                    cell,
                    cell_metrics,
                    CellStatus.REGULAR_LOW_GAMMA,
                )
            )
            continue

        accumulator = accumulators[cell]
        if (
            accumulator.irregular_pathweight
            > parameters.delta_5 * cell_metrics.pathweight
        ):
            evaluations.append(
                CellEvaluation(
                    cell,
                    cell_metrics,
                    CellStatus.IRREGULAR_TRIANGLE_DEGREES,
                    irregular_pathweight=accumulator.irregular_pathweight,
                    deviation_pathweight=accumulator.deviation_pathweight,
                )
            )
            continue

        if (
            accumulator.deviation_pathweight
            > parameters.delta_2 * cell_metrics.pathweight
        ):
            evaluations.append(
                CellEvaluation(
                    cell,
                    cell_metrics,
                    CellStatus.IRREGULAR_DEVIATION,
                    irregular_pathweight=accumulator.irregular_pathweight,
                    deviation_pathweight=accumulator.deviation_pathweight,
                )
            )
            continue

        evaluations.append(
            CellEvaluation(
                cell,
                cell_metrics,
                CellStatus.REGULAR,
                irregular_pathweight=accumulator.irregular_pathweight,
                deviation_pathweight=accumulator.deviation_pathweight,
            )
        )
    return evaluations
