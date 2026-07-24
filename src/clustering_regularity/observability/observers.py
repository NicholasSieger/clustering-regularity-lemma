import json
from pathlib import Path
from typing import Iterable, Protocol

from .events import RoundCompleted, RunCompleted, RunStarted


class Observer(Protocol):
    def publish(self, event) -> None: ...


class NullObserver:
    def publish(self, event) -> None:
        return


class ConsoleObserver:
    def publish(self, event) -> None:
        if isinstance(event, RunStarted):
            print(
                f"start edges={event.edge_count} "
                f"pathweight={event.total_pathweight}",
                flush=True,
            )
        elif isinstance(event, RoundCompleted):
            summary = event.summary
            print(
                f"round={summary.round_index} "
                f"parts={summary.part_count_a}x{summary.part_count_b} "
                f"failed={summary.failed_cell_count}/"
                f"{summary.total_cell_count} "
                f"failed_weight={summary.failed_pathweight_ratio:.6f} "
                f"seconds={summary.elapsed_seconds:.3f}",
                flush=True,
            )
        elif isinstance(event, RunCompleted):
            print(
                f"complete generation={event.generation} "
                f"parts={event.part_count_a}x{event.part_count_b}",
                flush=True,
            )


class JsonlObserver:
    """Streams bounded round-level events directly to disk."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, event) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            json.dump(event.to_dict(), handle, separators=(",", ":"))
            handle.write("\n")


class CompositeObserver:
    def __init__(self, observers: Iterable[Observer]):
        self.observers = tuple(observers)

    def publish(self, event) -> None:
        for observer in self.observers:
            observer.publish(event)

