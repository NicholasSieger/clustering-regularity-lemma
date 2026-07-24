from clustering_regularity.config import PaperParameters
from clustering_regularity.core.models import CellId, CellMetrics, CellStatus
from clustering_regularity.core.refinement import (
    CellAccumulator,
    evaluate_cells,
)


def test_low_gamma_cell_does_not_require_an_accumulator():
    cell = CellId(2, 3)
    metrics = {cell: CellMetrics(pathweight=100, triangle_count=4)}

    evaluation = evaluate_cells(metrics, {}, PaperParameters())[0]

    assert evaluation.status is CellStatus.REGULAR_LOW_GAMMA
    assert not evaluation.requires_refinement


def test_triangle_irregularity_uses_aggregate_pathweight():
    cell = CellId(2, 3)
    metrics = {cell: CellMetrics(pathweight=100, triangle_count=5)}
    accumulator = CellAccumulator(
        metrics[cell],
        irregular_pathweight=1,
        positive_pathweight=1,
    )

    evaluation = evaluate_cells(
        metrics,
        {cell: accumulator},
        PaperParameters(),
    )[0]

    assert evaluation.status is CellStatus.IRREGULAR_TRIANGLE_DEGREES
