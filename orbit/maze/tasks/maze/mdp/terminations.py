from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import globals

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from ..maze_env import MazeEnv


def inactivity(env: MazeEnv) -> torch.Tensor:
    """Terminate the episode when the inactivity time exceeds the maximum inactivity length."""
    return env.inactive_buf >= env.max_inactive_time


def full_completion(env: MazeEnv) -> torch.Tensor:
    return globals.path_accumulated >= env.full_completion_threshold
