from __future__ import annotations

import random
import numpy as np
import torch
import globals
from typing import TYPE_CHECKING, List, Tuple, Sequence

from .randomization_bound_type import RandomizationBoundType
from .randomization_bound import RandomizationBound
from .randomization_boundary import RandomizationBoundary
from .randomization_parameter import RandomizationParameter
from .randomization_performance_buffer import RandomizationPerformanceBuffer

if TYPE_CHECKING:
    from ..maze_env import MazeEnv, MazeEnvCfg
    from ..maze_env_cfg import DomainRandomizationCfg


class DomainRandomizer:

    def __init__(self, env: MazeEnv, cfg: MazeEnvCfg):
        self.env = env
        domain_cfg: DomainRandomizationCfg = cfg.domain_randomization
        randomizable_params: List[RandomizationParameter] = domain_cfg.RANDOMIZABLE_PARAMETERS
        self.randomized_parameters, self.sampling_weights = self._init_params(randomizable_params)
        self.PARAM_IDS = {p.name: idx for idx, p in enumerate(randomizable_params)}

        self.buffer = RandomizationPerformanceBuffer(randomizable_params, buffer_size=domain_cfg.buffer_size)
        self.buffer_size = domain_cfg.buffer_size
        self.evaluation_probability = domain_cfg.evaluation_probability

        self.sampled_boundaries = [None] * cfg.scene.num_envs
        self.evaluated_param_ids = torch.full((cfg.scene.num_envs,), -1, dtype=torch.int64, device="cuda:0")

        # performance
        self.lower_performance_threshold = domain_cfg.performance_threshold_lower
        self.upper_performance_threshold = domain_cfg.performance_threshold_upper

    @staticmethod
    def _init_params(params: List[RandomizationParameter]) -> tuple[dict, dict]:
        """
        Convert a list of parameters to dict.

        Args:
            params (List[RandomizationParameter]): A list of randomized parameters.

        Returns:
            dict
        """
        randomized = dict()
        weights = dict()

        for param in params:
            randomized[param.name] = param
            weights[param.name] = param.sampling_weight

        return randomized, weights

    def step(self, env_ids: Sequence[int]):

        if globals.path_accumulated is not None:
            print(f"Overall performance at ADR step: {torch.median(globals.path_accumulated[env_ids]).item()}")
            print("Maximum average path: ", self.env.maximum_average_path)
        for env_id in env_ids:
            boundary = self.sampled_boundaries[env_id]
            if boundary is not None:
                self.update_buffer(boundary, float(globals.path_accumulated[env_id]))
                self.update_boundary(boundary)

            # sample
            self._sample_boundary(env_id)

    def update_buffer(self, sampled_boundary: RandomizationBoundary, episode_return: float) -> None:
        """
        Update buffer with the sampled boundary and associated episode return.

        Args:
          sampled_boundary (RandomizationBoundary): Parameter boundary sampled for Auto DR.
          episode_return (float): Episode return for the sampled boundary.

        Returns:
          None
        """
        self.buffer.insert(sampled_boundary, episode_return)

    def update_boundary(self, sampled_boundary: RandomizationBoundary) -> None:
        """
        Update ADR bounds based on the performance for a given boundary.

        Args:
          sampled_boundary (RandomizationBoundary): Sampled boundary to evaluate.

        Returns:
          None
        """

        if not self.buffer.is_full(sampled_boundary):
            return

        performance = np.median(np.array(self.buffer.get(sampled_boundary)))
        print(
            f"Performance for parameter: {sampled_boundary.parameter.name} {sampled_boundary.bound.type.name} = {performance}"
        )
        self.buffer.truncate(sampled_boundary)

        param: RandomizationParameter = sampled_boundary.parameter
        bound: RandomizationBound = sampled_boundary.bound

        # increase entropy
        if performance >= self.upper_performance_threshold:
            if bound.type == RandomizationBoundType.UPPER_BOUND:
                self.randomized_parameters[param.name].increase_upper_bound()
            elif bound.type == RandomizationBoundType.LOWER_BOUND:
                self.randomized_parameters[param.name].decrease_lower_bound()
            else:
                raise ValueError

        # decrease entropy
        if performance < self.lower_performance_threshold:
            if bound.type == RandomizationBoundType.UPPER_BOUND:
                self.randomized_parameters[param.name].decrease_upper_bound()
            elif bound.type == RandomizationBoundType.LOWER_BOUND:
                self.randomized_parameters[param.name].increase_lower_bound()
            else:
                raise ValueError

    def _sample_boundary(self, env_id):
        """
        Get randomized parameter values.

        Returns:
          Tuple
        """
        if np.random.uniform(0, 1) <= self.evaluation_probability:
            params = list(self.randomized_parameters.values())
            weights = list(self.sampling_weights.values())
            sampled_param = random.choices(params, weights=weights, k=1)[0]
            sampled_bound = random.choice(list([sampled_param.lower_bound, sampled_param.upper_bound]))

            sampled_boundary = RandomizationBoundary(parameter=sampled_param, bound=sampled_bound)

            self.sampled_boundaries[env_id] = sampled_boundary
            self.evaluated_param_ids[env_id] = self.PARAM_IDS[sampled_boundary.parameter.name]
