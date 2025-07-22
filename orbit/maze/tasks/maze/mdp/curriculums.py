from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
from collections.abc import Sequence

if TYPE_CHECKING:
    from ..maze_env import MazeEnv


def increase_penalty_on_hole(
    env: MazeEnv,
    env_ids: Sequence[int],
    thresholds_and_weights: Sequence[Tuple[float, float]],
):
    """
    Increases the penalty on 'on_hole' term based on max average path across training.

    Args:
        env: the environment
        env_ids: list of environment indices, not used but necessary for CurrTerm defintion
        thresholds_and_weights: list of (threshold, new_weight) tuples
    """
    # Sort by threshold to ensure proper ordering
    current_max = env.maximum_average_path
    applied_weight = None

    for threshold, weight in thresholds_and_weights:
        if current_max > threshold:
            applied_weight = weight
        else:
            break

    if applied_weight is not None:
        penalty_term = env.reward_manager.get_term_cfg("on_hole")
        if penalty_term.weight != applied_weight:
            penalty_term.weight = applied_weight
            env.reward_manager.set_term_cfg("on_hole", penalty_term)
