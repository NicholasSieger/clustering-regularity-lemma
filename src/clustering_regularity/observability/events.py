from dataclasses import asdict, dataclass
from typing import Any, Dict

from ..core.models import RoundSummary


@dataclass(frozen=True)
class RunStarted:
    edge_count: int
    total_pathweight: int

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "run_started", **asdict(self)}


@dataclass(frozen=True)
class RoundCompleted:
    summary: RoundSummary

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "round_completed", **self.summary.to_dict()}


@dataclass(frozen=True)
class RunCompleted:
    generation: int
    part_count_a: int
    part_count_b: int

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "run_completed", **asdict(self)}

