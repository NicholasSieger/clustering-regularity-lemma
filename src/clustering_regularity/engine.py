import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import RunConfig
from .core.local_checks import LocalCheck
from .core.models import (
    CellEvaluation,
    CellId,
    CellMetrics,
    CellStatus,
    RefinementCut,
    RoundSummary,
    RunResult,
)
from .core.refinement import (
    CellAccumulator,
    evaluate_cells,
)
from .graph.index import GraphIndex
from .observability.events import RoundCompleted, RunCompleted, RunStarted
from .observability.observers import NullObserver, Observer
from .storage.protocols import PartitionStore


def _local_groups(
    labels: np.ndarray,
    edge_indices: np.ndarray,
) -> Dict[int, np.ndarray]:
    local_labels = np.asarray(labels[edge_indices], dtype=np.int64)
    return {
        int(label): np.flatnonzero(local_labels == label)
        for label in np.unique(local_labels)
    }


def _close_memmap(array: np.ndarray) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _append_indices(path: Path, indices: np.ndarray) -> None:
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return
    with open(path, "ab") as handle:
        values.tofile(handle)


class RefinementEngine:
    """Coordinates file-backed rounds without owning logging or reports."""

    def __init__(
        self,
        graph_index: GraphIndex,
        partitions: PartitionStore,
        config: RunConfig,
        observer: Observer = None,
    ):
        self.graph_index = graph_index
        self.partitions = partitions
        self.config = config
        self.parameters = config.parameters
        self.observer = observer or NullObserver()

    def _total_pathweight(self) -> int:
        return sum(
            len(self.graph_index.load_shard(shard_id).neighbors) ** 2
            for shard_id in self.graph_index.iter_shard_ids()
        )

    def _measure_cells(
        self,
        generation: int,
    ) -> Dict[CellId, CellMetrics]:
        labels_a = self.partitions.labels(generation, "A")
        labels_b = self.partitions.labels(generation, "B")
        metrics: Dict[CellId, CellMetrics] = defaultdict(CellMetrics)
        try:
            for shard_id in self.graph_index.iter_shard_ids():
                shard = self.graph_index.load_shard(shard_id)
                groups_a = _local_groups(labels_a, shard.edge_indices)
                groups_b = _local_groups(labels_b, shard.edge_indices)
                for label_a, positions_a in groups_a.items():
                    for label_b, positions_b in groups_b.items():
                        cell = CellId(label_a, label_b)
                        cell_metrics = metrics[cell]
                        cell_metrics.pathweight += (
                            len(positions_a) * len(positions_b)
                        )
                        cell_metrics.triangle_count += int(
                            shard.closure[positions_a, :][:, positions_b].sum()
                        )
        finally:
            _close_memmap(labels_a)
            _close_memmap(labels_b)
        return dict(metrics)

    def _collect_local_checks(
        self,
        generation: int,
        metrics: Dict[CellId, CellMetrics],
    ) -> Dict[CellId, CellAccumulator]:
        labels_a = self.partitions.labels(generation, "A")
        labels_b = self.partitions.labels(generation, "B")
        accumulators = {
            cell: CellAccumulator(cell_metrics)
            for cell, cell_metrics in metrics.items()
            if cell_metrics.gamma >= self.parameters.epsilon
        }
        if not accumulators:
            _close_memmap(labels_a)
            _close_memmap(labels_b)
            return accumulators

        try:
            for shard_id in self.graph_index.iter_shard_ids():
                shard = self.graph_index.load_shard(shard_id)
                groups_a = _local_groups(labels_a, shard.edge_indices)
                groups_b = _local_groups(labels_b, shard.edge_indices)
                for label_a, positions_a in groups_a.items():
                    for label_b, positions_b in groups_b.items():
                        cell = CellId(label_a, label_b)
                        accumulator = accumulators.get(cell)
                        if accumulator is None:
                            continue
                        local = LocalCheck(
                            shard.closure[positions_a, :][:, positions_b],
                            self.parameters,
                        )
                        positive, positive_count, negative, negative_count = (
                            local.irregular_edges(accumulator.metrics.gamma)
                        )
                        irregular_count = positive_count + negative_count
                        if (
                            irregular_count
                            > self.parameters.delta_4 * local.size_a
                        ):
                            accumulator.irregular_pathweight += local.pathweight
                            use_positive = positive_count > negative_count
                            if use_positive:
                                accumulator.positive_pathweight += local.pathweight
                            else:
                                accumulator.negative_pathweight += local.pathweight
                        elif (
                            local.deviation(accumulator.metrics.gamma)
                            > self.parameters.delta_1
                            * local.size_a**2
                            * local.size_b
                        ):
                            accumulator.deviation_pathweight += local.pathweight
        finally:
            _close_memmap(labels_a)
            _close_memmap(labels_b)
        return accumulators

    def _write_refinement_cuts(
        self,
        generation: int,
        evaluations: Sequence[CellEvaluation],
        accumulators: Dict[CellId, CellAccumulator],
        directory: Path,
    ) -> Tuple[List[RefinementCut], List[RefinementCut]]:
        failed = {
            evaluation.cell: evaluation
            for evaluation in evaluations
            if evaluation.requires_refinement
        }
        paths: Dict[Tuple[str, CellId], Path] = {}
        for cell, evaluation in failed.items():
            paths["A", cell] = directory / (
                f"A-{cell.label_a}-{cell.label_b}.indices"
            )
            paths["A", cell].touch()
            if evaluation.status is CellStatus.IRREGULAR_DEVIATION:
                paths["B", cell] = directory / (
                    f"B-{cell.label_a}-{cell.label_b}.indices"
                )
                paths["B", cell].touch()

        labels_a = self.partitions.labels(generation, "A")
        labels_b = self.partitions.labels(generation, "B")
        try:
            for shard_id in self.graph_index.iter_shard_ids():
                shard = self.graph_index.load_shard(shard_id)
                groups_a = _local_groups(labels_a, shard.edge_indices)
                groups_b = _local_groups(labels_b, shard.edge_indices)
                for label_a, positions_a in groups_a.items():
                    for label_b, positions_b in groups_b.items():
                        cell = CellId(label_a, label_b)
                        evaluation = failed.get(cell)
                        if evaluation is None:
                            continue
                        accumulator = accumulators[cell]
                        local = LocalCheck(
                            shard.closure[positions_a, :][:, positions_b],
                            self.parameters,
                        )
                        positive, positive_count, negative, negative_count = (
                            local.irregular_edges(accumulator.metrics.gamma)
                        )
                        irregular_count = positive_count + negative_count
                        locally_irregular = (
                            irregular_count
                            > self.parameters.delta_4 * local.size_a
                        )
                        if (
                            evaluation.status
                            is CellStatus.IRREGULAR_TRIANGLE_DEGREES
                        ):
                            if not locally_irregular:
                                continue
                            use_positive = (
                                accumulator.positive_pathweight
                                > accumulator.negative_pathweight
                            )
                            local_uses_positive = (
                                positive_count > negative_count
                            )
                            if use_positive != local_uses_positive:
                                continue
                            mask = positive if use_positive else negative
                            _append_indices(
                                paths["A", cell],
                                shard.edge_indices[positions_a][mask],
                            )
                            continue

                        if locally_irregular:
                            continue
                        if (
                            local.deviation(accumulator.metrics.gamma)
                            <= self.parameters.delta_1
                            * local.size_a**2
                            * local.size_b
                        ):
                            continue
                        split_a, split_b = local.deviation_split(
                            accumulator.metrics.gamma
                        )
                        _append_indices(
                            paths["A", cell],
                            shard.edge_indices[positions_a][split_a],
                        )
                        _append_indices(
                            paths["B", cell],
                            shard.edge_indices[positions_b][split_b],
                        )
        finally:
            _close_memmap(labels_a)
            _close_memmap(labels_b)

        cuts_a = []
        cuts_b = []
        for (side, cell), path in paths.items():
            if path.stat().st_size == 0:
                continue
            cut = RefinementCut(
                parent_label=(
                    cell.label_a if side == "A" else cell.label_b
                ),
                edge_indices_path=path,
            )
            (cuts_a if side == "A" else cuts_b).append(cut)
        return cuts_a, cuts_b

    def run(self, workspace: Path) -> RunResult:
        generation = self.partitions.current_generation
        total_pathweight = self._total_pathweight()
        self.observer.publish(
            RunStarted(self.graph_index.edge_count, total_pathweight)
        )
        round_summaries = []
        round_index = 0

        while True:
            if (
                self.config.max_rounds is not None
                and round_index > self.config.max_rounds
            ):
                raise RuntimeError(
                    f"refinement round {round_index} exceeds max_rounds "
                    f"{self.config.max_rounds}"
                )
            started = time.monotonic()
            part_count_a = self.partitions.part_count(generation, "A")
            part_count_b = self.partitions.part_count(generation, "B")
            metrics = self._measure_cells(generation)
            measured_pathweight = sum(
                item.pathweight for item in metrics.values()
            )
            if measured_pathweight != total_pathweight:
                raise RuntimeError(
                    "cell pathweights do not preserve global pathweight "
                    f"({measured_pathweight} != {total_pathweight})"
                )
            accumulators = self._collect_local_checks(generation, metrics)
            evaluations = evaluate_cells(
                metrics,
                accumulators,
                self.parameters,
            )
            failed = [
                evaluation
                for evaluation in evaluations
                if evaluation.requires_refinement
            ]
            failed_pathweight = sum(
                evaluation.metrics.pathweight for evaluation in failed
            )
            zero_cells = part_count_a * part_count_b - len(metrics)
            branches = Counter(
                evaluation.status.value for evaluation in evaluations
            )
            if zero_cells:
                branches[CellStatus.REGULAR_ZERO_PATH.value] = zero_cells
            summary = RoundSummary(
                round_index=round_index,
                generation=generation,
                part_count_a=part_count_a,
                part_count_b=part_count_b,
                evaluated_nonzero_cells=len(metrics),
                total_cell_count=part_count_a * part_count_b,
                failed_cell_count=len(failed),
                total_pathweight=total_pathweight,
                failed_pathweight=failed_pathweight,
                branch_counts=dict(branches),
                elapsed_seconds=time.monotonic() - started,
            )
            round_summaries.append(summary)
            self.observer.publish(RoundCompleted(summary))

            if (
                total_pathweight == 0
                or failed_pathweight < self.parameters.epsilon * total_pathweight
            ):
                break

            previous_counts = part_count_a, part_count_b
            work_root = Path(workspace) / "work"
            work_root.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f"round-{round_index:08d}-",
                dir=work_root,
            ) as directory:
                cuts_a, cuts_b = self._write_refinement_cuts(
                    generation,
                    failed,
                    accumulators,
                    Path(directory),
                )
                generation = self.partitions.commit_refinement(
                    generation,
                    cuts_a,
                    cuts_b,
                )
            next_counts = (
                self.partitions.part_count(generation, "A"),
                self.partitions.part_count(generation, "B"),
            )
            if next_counts == previous_counts:
                raise RuntimeError(
                    "regularity checks require refinement, but witness cuts "
                    "do not split a current edge class"
                )
            round_index += 1

        result = RunResult(
            workspace=Path(workspace),
            generation=generation,
            labels_a_path=self.partitions.labels_path(generation, "A"),
            labels_b_path=self.partitions.labels_path(generation, "B"),
            part_count_a=self.partitions.part_count(generation, "A"),
            part_count_b=self.partitions.part_count(generation, "B"),
            total_pathweight=total_pathweight,
            rounds=tuple(round_summaries),
        )
        self.observer.publish(
            RunCompleted(
                generation,
                result.part_count_a,
                result.part_count_b,
            )
        )
        result_path = Path(workspace) / "result.json"
        temporary = result_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(result.to_summary_dict(), handle, indent=2)
            handle.write("\n")
        temporary.replace(result_path)
        return result
