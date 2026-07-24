from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
import json
import queue as q

import networkx as nx
import numpy as np

from clustering_task import Task
from edge_partition import EdgePartitionAssembler, partition_labels
from iteration_log import PartitionStats
from parameters import AlgorithmParameters
from partition import PartitionRecord, convert_json_to_nodes
from storage import FileManager
from tripartite import GraphManager


class AlgorithmRunner(object):
    """Coordinates graph storage, local partition refinement, and final labels."""

    @staticmethod
    def _convert_json_to_nodes(json_list: List) -> List:
        return convert_json_to_nodes(json_list)

    @staticmethod
    def _resolve_parameters(
        parameters: Optional[AlgorithmParameters],
        eps: float,
        irreg_vtx_threshold: float,
        dev_vtx_threshold: float,
        irreg_vtx_count_threshold: float,
        dev_threshold: float,
        dev_split_threshold: float,
        irreg_threshold: float,
        clustering_threshold: float,
        max_depth: float,
    ) -> AlgorithmParameters:
        """Build the effective parameter set from defaults and explicit overrides."""
        if parameters is None:
            if eps is None:
                raise ValueError("eps is required when parameters is not provided")
            return AlgorithmParameters(
                eps=eps,
                irreg_vtx_threshold=irreg_vtx_threshold,
                dev_vtx_threshold=dev_vtx_threshold,
                irreg_vtx_count_threshold=irreg_vtx_count_threshold,
                dev_threshold=dev_threshold,
                dev_split_threshold=dev_split_threshold,
                irreg_threshold=irreg_threshold,
                clustering_threshold=clustering_threshold,
                max_depth=max_depth,
            )

        if eps is not None and eps != parameters.eps:
            raise ValueError("conflicting eps values provided by eps and parameters")

        overrides = {}
        for name, value in (
            ("irreg_vtx_threshold", irreg_vtx_threshold),
            ("dev_vtx_threshold", dev_vtx_threshold),
            ("irreg_vtx_count_threshold", irreg_vtx_count_threshold),
            ("dev_threshold", dev_threshold),
            ("dev_split_threshold", dev_split_threshold),
            ("irreg_threshold", irreg_threshold),
            ("clustering_threshold", clustering_threshold),
        ):
            if value is not None:
                overrides[name] = value

        if max_depth != float("inf"):
            overrides["max_depth"] = max_depth

        if not overrides:
            return parameters
        return replace(parameters, **overrides)

    def __init__(
        self,
        G: nx.Graph,
        eps: float = None,
        irreg_vtx_threshold: float = None,
        dev_vtx_threshold: float = None,
        irreg_vtx_count_threshold: float = None,
        dev_threshold: float = None,
        dev_split_threshold: float = None,
        irreg_threshold: float = None,
        clustering_threshold: float = None,
        max_depth: int = float("inf"),
        partition_dir: str = "partitions",
        graph_dir: str = "graphs",
        partition_manager=None,
        graph_manager=None,
        parameters: AlgorithmParameters = None,
        file_manager_cls=FileManager,
        progress_callback=None,
    ) -> None:
        parameters = self._resolve_parameters(
            parameters,
            eps,
            irreg_vtx_threshold,
            dev_vtx_threshold,
            irreg_vtx_count_threshold,
            dev_threshold,
            dev_split_threshold,
            irreg_threshold,
            clustering_threshold,
            max_depth,
        )

        self.parameters = parameters
        self.eps = parameters.eps
        self.irreg_vtx_threshold = parameters.irreg_vtx_threshold
        self.dev_vtx_threshold = parameters.dev_vtx_threshold
        self.irreg_vtx_count_threshold = parameters.irreg_vtx_count_threshold
        self.dev_threshold = parameters.dev_threshold
        self.dev_split_threshold = parameters.dev_split_threshold
        self.irreg_threshold = parameters.irreg_threshold
        self.clustering_threshold = parameters.clustering_threshold
        self.max_depth = parameters.max_depth

        self.graph_manager = graph_manager or GraphManager(G)
        self.partition_manager = partition_manager or file_manager_cls(partition_dir, graph_dir)
        self.edge_assembler = EdgePartitionAssembler(self.graph_manager)
        self.V2 = self.graph_manager.getV2()
        self.q = q.Queue()
        self.directions_considered = []
        self.partition_logs = []
        self.progress_callback = progress_callback

    def _emit_progress(self, event: str, **details) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback({"event": event, **details})
        except Exception:
            # Diagnostics must never alter algorithm behavior.
            return

    def _get_edge_lists(self) -> Tuple[List[Tuple], List[Tuple]]:
        return self.edge_assembler.get_edge_lists()

    def _map_vertex_partition_to_edges(
        self,
        v: int,
        mask_A: np.ndarray,
        mask_B: np.ndarray,
        neighbors_A: List,
        neighbors_B: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.edge_assembler.map_vertex_partition_to_edges(
            v, mask_A, mask_B, neighbors_A, neighbors_B
        )

    def _load_partition_record(self, vertex: int, direction: str) -> Tuple[bool, PartitionRecord]:
        if hasattr(self.partition_manager, "loadPartitionRecord"):
            return self.partition_manager.loadPartitionRecord(vertex, direction)

        success, partition_str = self.partition_manager.loadPartition(vertex, direction)
        if not success:
            return False, None
        return True, PartitionRecord.from_json_dict(json.loads(partition_str))

    def _load_partition_with_mask(self, vertex: int, direction: str) -> Tuple[bool, Any]:
        """Load a partition and apply its masks to select neighbors."""
        try:
            success, record = self._load_partition_record(vertex, direction)
            if not success:
                return False, ([], [])
            return True, record.masked_neighbors()
        except Exception:
            return False, ([], [])

    def _initialize_link_data(self) -> None:
        """Create initial full partitions and link graphs for all V2 vertices."""
        self.V2 = self.graph_manager.getV2()
        for v in self.V2:
            A, B = self.graph_manager.makeLinkPartition(v)
            mask_A = np.ones(len(A), dtype=bool)
            mask_B = np.ones(len(B), dtype=bool)
            self.partition_manager.savePartition(v, "", mask_A, mask_B, A, B)
            self.partition_manager.saveLinkGraph(v, self.graph_manager.makeLinkGraph(v))

    @staticmethod
    def _mask_diagnostics(masks: List[np.ndarray], edge_count: int) -> Dict[str, int]:
        if not masks:
            membership = np.zeros(edge_count, dtype=int)
        else:
            membership = np.sum(np.asarray(masks, dtype=int), axis=0)
        return {
            "edge_count": edge_count,
            "uncovered_edge_count": int(np.sum(membership == 0)),
            "overlapping_edge_count": int(np.sum(membership > 1)),
            "maximum_mask_membership": (
                int(membership.max()) if membership.size else 0
            ),
        }

    def _edge_incidence(self):
        """Index each global edge by its unique middle-part vertex."""
        edges_A, edges_B = self._get_edge_lists()
        incidence_A = {v: [] for v in self.V2}
        incidence_B = {v: [] for v in self.V2}

        for index, (neighbor, middle) in enumerate(edges_A):
            incidence_A.setdefault(middle[0], []).append((index, neighbor))
        for index, (middle, neighbor) in enumerate(edges_B):
            incidence_B.setdefault(middle[0], []).append((index, neighbor))

        return edges_A, edges_B, incidence_A, incidence_B

    @staticmethod
    def _masked_incidence(entries, mask: np.ndarray):
        selected = [(index, node) for index, node in entries if mask[index]]
        return [index for index, _ in selected], [node for _, node in selected]

    def _load_global_task(self, vertex, neighbors_A, neighbors_B) -> Task:
        success, link = self.partition_manager.loadLinkGraph(vertex)
        if not success:
            raise ValueError(f"Failed to load link graph for vertex {vertex}")
        return Task(
            link,
            (neighbors_A, neighbors_B),
            self.eps,
            self.irreg_vtx_threshold,
            self.dev_vtx_threshold,
            self.dev_split_threshold,
        )

    def _evaluate_global_pair(
        self,
        direction: str,
        mask_A: np.ndarray,
        mask_B: np.ndarray,
        incidence_A,
        incidence_B,
    ) -> Dict[str, Any]:
        """Evaluate one Cartesian pair of current global edge classes."""
        pathweight = 0
        triangle_count = 0

        for vertex in self.V2:
            _, neighbors_A = self._masked_incidence(
                incidence_A.get(vertex, []), mask_A
            )
            _, neighbors_B = self._masked_incidence(
                incidence_B.get(vertex, []), mask_B
            )
            local_pathweight = len(neighbors_A) * len(neighbors_B)
            pathweight += local_pathweight
            if local_pathweight:
                triangle_count += self._load_global_task(
                    vertex, neighbors_A, neighbors_B
                ).edges

        gamma = 0.0 if pathweight == 0 else triangle_count / pathweight
        stats = self._base_stats(direction, pathweight, triangle_count, gamma)
        result = {
            "stats": stats,
            "cut_A": None,
            "cut_B": None,
            "failed": False,
        }

        if gamma < self.clustering_threshold:
            stats.failure_reason = "PASSED_GAMMA_CHECK"
            return result

        irregular_vertices = []
        deviation_vertices = []
        irreg_weight = 0
        dev_weight = 0
        positive_weight = 0
        negative_weight = 0

        for vertex in self.V2:
            indices_A, neighbors_A = self._masked_incidence(
                incidence_A.get(vertex, []), mask_A
            )
            indices_B, neighbors_B = self._masked_incidence(
                incidence_B.get(vertex, []), mask_B
            )
            local_pathweight = len(neighbors_A) * len(neighbors_B)
            if local_pathweight == 0:
                continue

            task = self._load_global_task(vertex, neighbors_A, neighbors_B)
            positive, positive_count, negative, negative_count = (
                task.compute_irregular_vertices(gamma)
            )
            irregular_count = int(positive_count + negative_count)

            if (
                irregular_count
                > self.irreg_vtx_count_threshold * len(neighbors_A)
            ):
                irreg_weight += local_pathweight
                use_positive = positive_count > negative_count
                if use_positive:
                    positive_weight += local_pathweight
                else:
                    negative_weight += local_pathweight
                irregular_vertices.append(
                    {
                        "indices_A": indices_A,
                        "positive": np.asarray(positive, dtype=bool),
                        "negative": np.asarray(negative, dtype=bool),
                        "use_positive": use_positive,
                    }
                )
                continue

            deviation = task.compute_local_deviation(gamma)
            if (
                deviation
                > self.dev_vtx_threshold
                * len(neighbors_A) ** 2
                * len(neighbors_B)
            ):
                dev_weight += local_pathweight
                split_A, split_B = task.produce_new_masks(gamma)
                deviation_vertices.append(
                    {
                        "indices_A": indices_A,
                        "indices_B": indices_B,
                        "split_A": np.asarray(split_A, dtype=bool),
                        "split_B": np.asarray(split_B, dtype=bool),
                    }
                )

        stats.irreg_weight = irreg_weight
        stats.dev_weight = dev_weight
        stats.irreg_threshold = self.irreg_threshold * pathweight
        stats.dev_threshold = self.dev_threshold * pathweight

        if irreg_weight > self.irreg_threshold * pathweight:
            use_positive = positive_weight > negative_weight
            cut_A = np.zeros(mask_A.shape[0], dtype=bool)
            for vertex_data in irregular_vertices:
                if vertex_data["use_positive"] != use_positive:
                    continue
                local_cut = (
                    vertex_data["positive"]
                    if use_positive
                    else vertex_data["negative"]
                )
                cut_A[np.asarray(vertex_data["indices_A"], dtype=int)] = local_cut

            stats.failure_reason = "FAILED_IRREGULARITY_CHECK"
            result.update(failed=True, cut_A=cut_A)
            return result

        if dev_weight > self.dev_threshold * pathweight:
            cut_A = np.zeros(mask_A.shape[0], dtype=bool)
            cut_B = np.zeros(mask_B.shape[0], dtype=bool)
            for vertex_data in deviation_vertices:
                cut_A[np.asarray(vertex_data["indices_A"], dtype=int)] = (
                    vertex_data["split_A"]
                )
                cut_B[np.asarray(vertex_data["indices_B"], dtype=int)] = (
                    vertex_data["split_B"]
                )

            stats.failure_reason = "FAILED_DEVIATION_CHECK"
            result.update(failed=True, cut_A=cut_A, cut_B=cut_B)
            return result

        stats.failure_reason = "PASSED_ALL_CHECKS"
        return result

    @staticmethod
    def _common_refinement(
        parts: List[np.ndarray],
        cuts_by_part: Dict[int, List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Apply all witness cuts while preserving a true edge partition."""
        refined = []
        for part_index, parent in enumerate(parts):
            children = [parent]
            for proposed_cut in cuts_by_part.get(part_index, []):
                cut = np.asarray(proposed_cut, dtype=bool) & parent
                if not np.any(cut) or np.array_equal(cut, parent):
                    continue

                next_children = []
                for child in children:
                    inside = child & cut
                    outside = child & ~cut
                    if np.any(inside) and np.any(outside):
                        next_children.extend((inside, outside))
                    else:
                        next_children.append(child)
                children = next_children
            refined.extend(children)
        return refined

    def _run_global_refinement(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        edges_A, edges_B, incidence_A, incidence_B = self._edge_incidence()
        parts_A = [np.ones(len(edges_A), dtype=bool)]
        parts_B = [np.ones(len(edges_B), dtype=bool)]
        total_pathweight = sum(
            len(incidence_A.get(vertex, []))
            * len(incidence_B.get(vertex, []))
            for vertex in self.V2
        )
        round_index = 0
        self.directions_considered = []
        self.partition_logs = []

        while True:
            cuts_A = {}
            cuts_B = {}
            failed_pathweight = 0
            round_pathweight = 0
            final_pair_ids = []

            for part_A_index, mask_A in enumerate(parts_A):
                for part_B_index, mask_B in enumerate(parts_B):
                    direction = (
                        ""
                        if round_index == 0
                        else f"r{round_index}:a{part_A_index}:b{part_B_index}"
                    )
                    code_length = 2 * round_index
                    self.directions_considered.append(direction)
                    self._emit_progress(
                        "direction_started",
                        direction=direction,
                        direction_code_length=code_length,
                        logical_refinement_depth=round_index,
                        directions_considered=len(self.directions_considered),
                        pending_direction_count=(
                            len(parts_A) * len(parts_B)
                            - (part_A_index * len(parts_B) + part_B_index + 1)
                        ),
                    )
                    if code_length > self.max_depth:
                        self._emit_progress(
                            "direction_limit_exceeded",
                            direction=direction,
                            direction_code_length=code_length,
                            logical_refinement_depth=round_index,
                            max_direction_code_length=self.max_depth,
                        )
                        raise ValueError(
                            f"Direction code for refinement round {round_index} "
                            f"(length {code_length}) exceeds maximum depth "
                            f"{self.max_depth}"
                        )

                    evaluation = self._evaluate_global_pair(
                        direction,
                        mask_A,
                        mask_B,
                        incidence_A,
                        incidence_B,
                    )
                    stats = evaluation["stats"]
                    round_pathweight += stats.pathweight
                    self.partition_logs.append(stats.to_dict())
                    self._emit_progress("direction_completed", **stats.to_dict())
                    final_pair_ids.append(direction)

                    if not evaluation["failed"]:
                        continue
                    failed_pathweight += stats.pathweight
                    if evaluation["cut_A"] is not None:
                        cuts_A.setdefault(part_A_index, []).append(
                            evaluation["cut_A"]
                        )
                    if evaluation["cut_B"] is not None:
                        cuts_B.setdefault(part_B_index, []).append(
                            evaluation["cut_B"]
                        )

            if round_pathweight != total_pathweight:
                raise RuntimeError(
                    "Cartesian cell pathweights do not preserve the global "
                    f"pathweight ({round_pathweight} != {total_pathweight})"
                )

            if total_pathweight == 0 or (
                failed_pathweight < self.eps * total_pathweight
            ):
                self.accepted_directions = final_pair_ids
                self._emit_progress(
                    "refinement_completed",
                    accepted_direction_count=len(final_pair_ids),
                    directions_considered=len(self.directions_considered),
                    refinement_rounds=round_index,
                    failed_pathweight=failed_pathweight,
                    allowed_failed_pathweight=self.eps * total_pathweight,
                )
                return parts_A, parts_B

            refined_A = self._common_refinement(parts_A, cuts_A)
            refined_B = self._common_refinement(parts_B, cuts_B)
            if len(refined_A) == len(parts_A) and len(refined_B) == len(parts_B):
                raise RuntimeError(
                    "Regularity checks require refinement, but their witness "
                    "cuts do not split any current edge class"
                )
            parts_A = refined_A
            parts_B = refined_B
            round_index += 1

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """Execute the main algorithm and return final E12 and E23 partition labels."""
        self._emit_progress("initialization_started", vertex_count=len(self.V2))
        try:
            self._initialize_link_data()
            self._emit_progress("initialization_completed", vertex_count=len(self.V2))
            masks_A, masks_B = self._run_global_refinement()

            edges_A, edges_B = self._get_edge_lists()
            self.partition_assembly_diagnostics = {
                "A": self._mask_diagnostics(masks_A, len(edges_A)),
                "B": self._mask_diagnostics(masks_B, len(edges_B)),
            }
            self.partition_labels_A = self.partitionLabels(masks_A)
            self.partition_labels_B = self.partitionLabels(masks_B)
            self._emit_progress(
                "assembly_completed",
                partition_count_A=(
                    len(set(self.partition_labels_A.tolist()))
                    if len(self.partition_labels_A)
                    else 0
                ),
                partition_count_B=(
                    len(set(self.partition_labels_B.tolist()))
                    if len(self.partition_labels_B)
                    else 0
                ),
                accepted_directions=self.accepted_directions,
                partition_assembly_diagnostics=self.partition_assembly_diagnostics,
            )
            return self.partition_labels_A, self.partition_labels_B
        finally:
            self.partition_manager.deleteAllPartitions()
            self.partition_manager.deleteAllLinkGraphs()

    def assemble_partition(self, dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Assembles global edge partitions for all vertices in V2 for one direction."""
        return self.edge_assembler.assemble_partition(self.V2, self.partition_manager, dir)

    def partitionLabels(self, bitmasks: List[np.ndarray]) -> np.ndarray:
        return partition_labels(bitmasks)

    def compute_direction_code_length(self, direction: str) -> int:
        """Computes the length of a direction code."""
        return len(direction)

    def compute_path_data(self, dir: str) -> Tuple[int, int, float]:
        """Computes pathweight, triangle count, and gamma for the current partition."""
        pathweight = 0
        triangle_count = 0
        for v in self.V2:
            success, link = self.partition_manager.loadLinkGraph(v)
            success_partition, (A, B) = self._load_partition_with_mask(v, dir)
            if not success or not success_partition:
                continue

            task = Task(link, (A, B), self.eps, self.irreg_vtx_threshold, self.dev_vtx_threshold, self.dev_split_threshold)
            pathweight += len(task.A) * len(task.B)
            triangle_count += task.edges

        gamma = 0.0 if pathweight == 0 else triangle_count / pathweight
        return pathweight, triangle_count, gamma

    def compute_graph_data(self, dir: str) -> Tuple[int, int, float]:
        """Compatibility alias for compute_path_data."""
        return self.compute_path_data(dir)

    def _base_stats(self, direction: str, pathweight: float, triangle_count: float, gamma: float):
        return PartitionStats(
            direction=direction,
            pathweight=pathweight,
            triangle_count=triangle_count,
            gamma=gamma,
            clustering_threshold=self.clustering_threshold,
        )

    def _save_irregular_splits(self, direction: str, gamma: float, bad_vertices: np.ndarray, pos: bool) -> None:
        for i, v in enumerate(self.V2):
            if not bad_vertices[i]:
                continue

            success_g, link = self.partition_manager.loadLinkGraph(v)
            success_p, (A, B) = self._load_partition_with_mask(v, direction)
            if not success_g or not success_p:
                raise ValueError(
                    f"Failed to load partition for vertex {v} and direction {direction} "
                    "when setting irregularity partition"
                )

            task = Task(link, (A, B), self.eps, self.irreg_vtx_threshold, self.dev_vtx_threshold, self.dev_split_threshold)
            pos_v, pos_score, neg_v, neg_score = task.compute_irregular_vertices(gamma)
            irreg_v = pos_v if pos else neg_v
            if any(np.array(irreg_v)):
                self.partition_manager.savePartition(
                    v, direction + "i0", np.array(irreg_v), np.ones(len(B), dtype=bool), A, B
                )
            if any(~np.array(irreg_v)):
                self.partition_manager.savePartition(
                    v, direction + "i1", ~np.array(irreg_v), np.ones(len(B), dtype=bool), A, B
                )
            self.partition_manager.deletePartition(v, direction)

    def _save_deviation_splits(self, direction: str, gamma: float, dev_vertices: np.ndarray) -> None:
        for i, v in enumerate(self.V2):
            if not dev_vertices[i]:
                continue

            success_g, link = self.partition_manager.loadLinkGraph(v)
            success_p, (A, B) = self._load_partition_with_mask(v, direction)
            if not success_g or not success_p:
                raise ValueError(
                    f"Failed to load partition for vertex {v} and direction {direction} "
                    "when setting deviation partition"
                )

            task = Task(link, (A, B), self.eps, self.irreg_vtx_threshold, self.dev_vtx_threshold, self.dev_split_threshold)
            L, R = task.produce_new_masks(gamma)
            if any(np.array(L)) and any(np.array(R)):
                self.partition_manager.savePartition(v, direction + "d0", np.array(L), np.array(R), A, B)
            if any(~np.array(L)) and any(np.array(R)):
                self.partition_manager.savePartition(v, direction + "d1", ~np.array(L), np.array(R), A, B)
            if any(np.array(L)) and any(~np.array(R)):
                self.partition_manager.savePartition(v, direction + "d2", np.array(L), ~np.array(R), A, B)
            if any(~np.array(L)) and any(~np.array(R)):
                self.partition_manager.savePartition(v, direction + "d3", ~np.array(L), ~np.array(R), A, B)

            self.partition_manager.deletePartition(v, direction)

    def iterate(self) -> List[str]:
        """Main refinement loop."""
        out = []
        self.directions_considered = []
        self.partition_logs = []

        while not self.q.empty():
            direction = self.q.get()
            self.directions_considered.append(direction)
            self._emit_progress(
                "direction_started",
                direction=direction,
                direction_code_length=self.compute_direction_code_length(direction),
                logical_refinement_depth=len(direction) // 2,
                directions_considered=len(self.directions_considered),
                pending_direction_count=self.q.qsize(),
            )

            if self.compute_direction_code_length(direction) > self.max_depth:
                self._emit_progress(
                    "direction_limit_exceeded",
                    direction=direction,
                    direction_code_length=self.compute_direction_code_length(direction),
                    logical_refinement_depth=len(direction) // 2,
                    max_direction_code_length=self.max_depth,
                )
                self.partition_manager.deleteAllPartitions()
                self.partition_manager.deleteAllLinkGraphs()
                raise ValueError(
                    f"Direction code '{direction}' "
                    f"(length {self.compute_direction_code_length(direction)}) "
                    f"exceeds maximum depth {self.max_depth}"
                )

            pathweight, triangle_count, gamma = self.compute_path_data(direction)
            stats = self._base_stats(direction, pathweight, triangle_count, gamma)

            if gamma < self.clustering_threshold:
                stats.failure_reason = "PASSED_GAMMA_CHECK"
                out.append(direction)
                self.partition_logs.append(stats.to_dict())
                self._emit_progress("direction_completed", **stats.to_dict())
                continue

            irreg_weight = 0.0
            dev_weight = 0.0
            dev_vertices = np.array([False] * len(self.V2))
            irreg_vertices = np.array([False] * len(self.V2))

            pos_weight = 0.0
            neg_weight = 0.0
            pos_vertices = np.array([False] * len(self.V2))
            neg_vertices = np.array([False] * len(self.V2))

            for i, v in enumerate(self.V2):
                success_g, link = self.partition_manager.loadLinkGraph(v)
                success_p, (A, B) = self._load_partition_with_mask(v, direction)
                if not success_g or not success_p:
                    continue

                task = Task(link, (A, B), self.eps, self.irreg_vtx_threshold, self.dev_vtx_threshold, self.dev_split_threshold)
                #irreg_v, irreg_count = task.compute_irregular_vertices(gamma)
                dev = task.compute_local_deviation(gamma)
                pos_v, pos_count, neg_v, neg_count = task.compute_irregular_vertices(gamma)
                irreg_count = pos_count + neg_count
                local_pathweight = len(task.A) * len(task.B)

                if irreg_count > self.irreg_vtx_count_threshold * len(task.A):
                    irreg_vertices[i] = True # keeping this arond just in case
                    irreg_weight += local_pathweight
                    if pos_count > neg_count:
                        pos_vertices[i] = True
                        pos_weight += local_pathweight
                    else:
                        neg_vertices[i] = True
                        neg_weight += local_pathweight
                    
                elif dev > self.dev_vtx_threshold * len(task.A)**2 * len(task.B):
                    dev_vertices[i] = True
                    dev_weight += local_pathweight

            stats.irreg_weight = irreg_weight
            stats.dev_weight = dev_weight
            stats.irreg_threshold = self.irreg_threshold * pathweight
            stats.dev_threshold = self.dev_threshold * pathweight

            bad_vertices = pos_vertices if pos_weight > neg_weight else neg_vertices

            if irreg_weight > self.irreg_threshold * pathweight:
                stats.failure_reason = "FAILED_IRREGULARITY_CHECK"
                self.partition_logs.append(stats.to_dict())
                self.q.put(direction + "i0")
                self.q.put(direction + "i1")
                self._save_irregular_splits(direction, gamma, bad_vertices, pos_weight > neg_weight)
            elif dev_weight > self.dev_threshold * pathweight:
                stats.failure_reason = "FAILED_DEVIATION_CHECK"
                self.partition_logs.append(stats.to_dict())
                self.q.put(direction + "d0")
                self.q.put(direction + "d1")
                self.q.put(direction + "d2")
                self.q.put(direction + "d3")
                self._save_deviation_splits(direction, gamma, dev_vertices)
            else:
                stats.failure_reason = "PASSED_ALL_CHECKS"
                self.partition_logs.append(stats.to_dict())
                out.append(direction)

            self._emit_progress("direction_completed", **stats.to_dict())

        return out
