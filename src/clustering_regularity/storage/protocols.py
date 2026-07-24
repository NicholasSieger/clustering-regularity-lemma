from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..core.models import RefinementCut


class PartitionStore(Protocol):
    @property
    def current_generation(self) -> int: ...

    def labels(self, generation: int, side: str) -> np.ndarray: ...

    def part_count(self, generation: int, side: str) -> int: ...

    def commit_refinement(
        self,
        generation: int,
        cuts_a: Sequence[RefinementCut],
        cuts_b: Sequence[RefinementCut],
    ) -> int: ...

    def labels_path(self, generation: int, side: str) -> Path: ...

