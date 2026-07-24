import json
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np

from ..api import run_graph
from ..observability.events import RoundCompleted
from .generators import generate_graph, graph_clustering_metrics
from .cells import finite_scale_threshold_report
from .mathematics import compute_partition_regularity_report
from .models import VerificationCase, VerificationConfig


class _QueueObserver:
    def __init__(self, output_queue):
        self.output_queue = output_queue

    def publish(self, event) -> None:
        if isinstance(event, RoundCompleted):
            try:
                self.output_queue.put_nowait(event.summary.to_dict())
            except Exception:
                return


def run_algorithm_case(
    case: VerificationCase,
    eps: float = 0.05,
    max_rounds: Optional[int] = None,
    verification_config: Optional[VerificationConfig] = None,
    observer=None,
) -> Dict:
    started = time.monotonic()
    config = verification_config or VerificationConfig(eps=eps)
    if config.eps != eps:
        raise ValueError("verification_config.eps must match the algorithm epsilon")
    graph, generator_metadata = generate_graph(
        case.family,
        case.size,
        case.density,
        case.seed,
    )
    actual_density = nx.density(graph)
    common = {
        "case_id": case.case_id,
        "family": case.family,
        "requested_size": case.size,
        "actual_size": graph.number_of_nodes(),
        "requested_density": case.density,
        "actual_density": actual_density,
        "density_error": abs(actual_density - case.density),
        "density_tolerance": config.density_tolerance,
        "edge_count": graph.number_of_edges(),
        "seed": case.seed,
        "eps": eps,
        "max_rounds": max_rounds,
        "generator": generator_metadata,
        "graph_clustering": graph_clustering_metrics(graph, eps),
        "finite_scale_thresholds": finite_scale_threshold_report(graph, config),
    }
    if common["density_error"] > config.density_tolerance:
        return {
            **common,
            "status": "generation_mismatch",
            "execution_status": "generation_mismatch",
            "verification_status": None,
            "runtime_seconds": time.monotonic() - started,
        }

    with tempfile.TemporaryDirectory(prefix=f"verify_{case.case_id}_") as directory:
        result = run_graph(
            graph,
            directory,
            epsilon=eps,
            max_rounds=max_rounds,
            observer=observer,
        )
        labels_a_mmap = result.labels("A")
        labels_b_mmap = result.labels("B")
        try:
            labels_a = labels_a_mmap.copy()
            labels_b = labels_b_mmap.copy()
        finally:
            labels_a_mmap._mmap.close()
            labels_b_mmap._mmap.close()
        verification = compute_partition_regularity_report(
            graph,
            labels_a,
            labels_b,
            config,
        )
        round_summaries = [summary.to_dict() for summary in result.rounds]

    return {
        **common,
        "status": verification["verification_status"],
        "execution_status": "completed",
        "verification_status": verification["verification_status"],
        "paper_partition_status": verification["paper_partition_status"],
        "runtime_seconds": time.monotonic() - started,
        "partition_count_A": result.part_count_a,
        "partition_count_B": result.part_count_b,
        "algorithm": {
            "generation": result.generation,
            "round_count": len(round_summaries),
            "rounds": round_summaries,
        },
        "verification": verification,
    }


def verify_workspace(
    graph: nx.Graph,
    workspace: Path,
    config: Optional[VerificationConfig] = None,
) -> Dict:
    """Verify the final labels referenced by a completed workspace."""
    workspace = Path(workspace).resolve()
    with open(workspace / "result.json", encoding="utf-8") as handle:
        result = json.load(handle)
    labels_a = np.load(workspace / result["labels_A"], mmap_mode="r")
    labels_b = np.load(workspace / result["labels_B"], mmap_mode="r")
    try:
        return compute_partition_regularity_report(
            graph,
            labels_a,
            labels_b,
            config or VerificationConfig(),
        )
    finally:
        labels_a._mmap.close()
        labels_b._mmap.close()


def _write_result(path: Path, result: Dict) -> None:
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, separators=(",", ":"))
    temporary.replace(path)


def _worker(
    case_data: Dict,
    eps: float,
    max_rounds: Optional[int],
    config_data: Dict,
    result_path: str,
    progress_queue,
) -> None:
    case = VerificationCase(**case_data)
    config = VerificationConfig(**config_data)
    try:
        result = run_algorithm_case(
            case,
            eps=eps,
            max_rounds=max_rounds,
            verification_config=config,
            observer=_QueueObserver(progress_queue),
        )
    except Exception as exc:
        status = (
            "max_rounds"
            if isinstance(exc, RuntimeError) and "max_rounds" in str(exc)
            else "algorithm_error"
        )
        result = {
            "case_id": case.case_id,
            "family": case.family,
            "requested_size": case.size,
            "requested_density": case.density,
            "seed": case.seed,
            "eps": eps,
            "max_rounds": max_rounds,
            "status": status,
            "execution_status": status,
            "verification_status": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    _write_result(Path(result_path), result)


def _drain_latest(progress_queue, current: Dict) -> Dict:
    latest = current
    while True:
        try:
            latest = progress_queue.get_nowait()
        except queue.Empty:
            return latest


def run_case_with_timeout(
    case: VerificationCase,
    timeout_seconds: float = 20.0,
    eps: float = 0.05,
    max_rounds: Optional[int] = None,
    verification_config: Optional[VerificationConfig] = None,
) -> Dict:
    config = verification_config or VerificationConfig(eps=eps)
    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    with tempfile.TemporaryDirectory(prefix=f"result_{case.case_id}_") as directory:
        result_path = Path(directory) / "result.json"
        process = context.Process(
            target=_worker,
            args=(
                asdict(case),
                eps,
                max_rounds,
                asdict(config),
                str(result_path),
                progress_queue,
            ),
        )
        started = time.monotonic()
        latest_round = {}
        process.start()
        while process.is_alive():
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            process.join(min(0.1, remaining))
            latest_round = _drain_latest(progress_queue, latest_round)
        runtime = time.monotonic() - started
        latest_round = _drain_latest(progress_queue, latest_round)
        if process.is_alive():
            process.terminate()
            process.join(5)
            return {
                "case_id": case.case_id,
                "family": case.family,
                "requested_size": case.size,
                "requested_density": case.density,
                "seed": case.seed,
                "eps": eps,
                "max_rounds": max_rounds,
                "status": "timeout",
                "execution_status": "timeout",
                "verification_status": None,
                "timeout_seconds": timeout_seconds,
                "runtime_seconds": runtime,
                "last_completed_round": latest_round or None,
            }
        if not result_path.exists():
            return {
                "case_id": case.case_id,
                "status": "algorithm_error",
                "execution_status": "algorithm_error",
                "verification_status": None,
                "runtime_seconds": runtime,
                "error": (
                    f"worker exited with code {process.exitcode} "
                    "without writing a result"
                ),
                "last_completed_round": latest_round or None,
            }
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)
    result.setdefault("runtime_seconds", runtime)
    result["last_completed_round"] = latest_round or None
    return result
