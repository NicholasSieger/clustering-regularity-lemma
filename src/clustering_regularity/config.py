from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PaperParameters:
    """Thresholds from the clustering regularity paper."""

    epsilon: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.epsilon < 1 / 16:
            raise ValueError("paper parameters require 0 < epsilon < 1/16")

    @property
    def delta_1(self) -> float:
        return self.epsilon**2 / 9

    @property
    def delta_2(self) -> float:
        return 2 * self.epsilon**2 / 5

    @property
    def delta_3(self) -> float:
        return self.epsilon**5 / 90

    @property
    def delta_4(self) -> float:
        return self.epsilon ** (5 / 2) / 9

    @property
    def delta_5(self) -> float:
        return 2 * self.epsilon ** (5 / 2) / 5

    @property
    def delta_6(self) -> float:
        return self.epsilon**5

    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "delta_3": self.delta_3,
            "delta_4": self.delta_4,
            "delta_5": self.delta_5,
            "delta_6": self.delta_6,
        }


@dataclass(frozen=True)
class RunConfig:
    """Execution limits that do not alter the paper thresholds."""

    parameters: PaperParameters = PaperParameters()
    max_rounds: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_rounds is not None and self.max_rounds < 0:
            raise ValueError("max_rounds must be nonnegative")
