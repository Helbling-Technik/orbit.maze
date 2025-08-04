from dataclasses import dataclass

import numpy as np
import torch

from .randomization_bound import RandomizationBound


@dataclass
class RandomizationParameter:
    """
    Dataclass describing a randomization associated with a specific environment.
    """

    name: str
    lower_bound: RandomizationBound
    upper_bound: RandomizationBound
    delta: float
    sampling_weight: float = 1.0

    def __post_init__(self):
        """
        Post-init for randomization parameters, adds validation for the values provided.

        Returns:
          None
        """
        assert self.lower_bound.value <= self.upper_bound.value
        assert self.lower_bound.max_value <= self.upper_bound.min_value

    @property
    def range(self) -> float:
        """
        Return the range of the parameter.

        Returns:
          float
        """
        return self.upper_bound.value - self.lower_bound.value

    def sample(self, mode: str = "other") -> float:  # TODO DRP Verify when used that it is behaving according to seed
        """
        Sample a value for the randomized parameter.

        Returns:
          float
        """
        if mode == "positive":
            return abs(np.random.uniform(self.lower_bound.value, self.upper_bound.value))
        return np.random.uniform(self.lower_bound.value, self.upper_bound.value)

    def sample_n(
        self, n: int, generator, mode: str = "other", device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Vectorized sampling using PyTorch (GPU-compatible).
        """
        low = self.lower_bound.value
        high = self.upper_bound.value

        if mode == "positive":
            return torch.abs(
                torch.empty(n, device=device).uniform_(
                    self.lower_bound.value, self.upper_bound.value, generator=generator
                )
            )
        return torch.empty(n, device=device).uniform_(low, high, generator=generator)

    def increase_upper_bound(self) -> None:
        """
        Increase the current upper bound for the randomized parameter.

        Returns:
          None
        """
        self.upper_bound.increase(self.delta)

    def decrease_upper_bound(self) -> None:
        """
        Decrease the current upper bound for the randomized parameter.

        Returns:
          None
        """
        self.upper_bound.decrease(self.delta)

    def decrease_lower_bound(self) -> None:
        """
        Decrease the lower bound by the delta value.

        Returns:
          None
        """
        self.lower_bound.decrease(self.delta)

    def increase_lower_bound(self) -> None:
        """
        Increase the lower bound by the delta value.

        Returns:
          None
        """
        self.lower_bound.increase(self.delta)
