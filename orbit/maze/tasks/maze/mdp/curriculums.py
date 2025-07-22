from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from ..maze_env import MazeEnv


def increase_penalty_on_hole(env: MazeEnv, env_ids: Sequence[int], new_weight: float, threshold: float):
    if env.maximum_average_path > threshold:
        penalty_term = env.reward_manager.get_term_cfg("on_hole")
        penalty_term.weight = new_weight
        env.reward_manager.set_term_cfg("on_hole", penalty_term)
