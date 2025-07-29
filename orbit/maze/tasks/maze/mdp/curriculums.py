from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
from collections.abc import Sequence

if TYPE_CHECKING:
    from ..maze_env import MazeEnv


def modify_reward_path(
    env: MazeEnv,
    env_ids: Sequence[int],
    term_name: str,
    thresholds_and_weights: Sequence[Tuple[float, float]],
):
    """
    Modifies a reward term based on max average path across training.

    Args:
        env: the environment
        env_ids: list of environment indices, not used but necessary for CurrTerm defintion
        term_name: The name of the reward term.
        thresholds_and_weights: list of (threshold, new_weight) tuples
    """
    current_max = env.maximum_average_path
    applied_weight = None

    for threshold, weight in thresholds_and_weights:
        if current_max > threshold:
            applied_weight = weight
        else:
            break

    if applied_weight is not None:
        penalty = env.reward_manager.get_term_cfg(term_name)
        if penalty.weight != applied_weight:
            penalty.weight = applied_weight
            env.reward_manager.set_term_cfg(term_name, penalty)


def modify_reward_steps(
    env: MazeEnv,
    env_ids: Sequence[int],
    term_name: str,
    steps_and_weights: Sequence[Tuple[float, float]],
):
    """
    Modifies a reward term based on number of steps in training.

    Args:
        env: the environment
        env_ids: list of environment indices, not used but necessary for CurrTerm defintion
        term_name: The name of the reward term.
        steps_and_weights: list of (steps, new_weight) tuples
    """
    # Sort by threshold to ensure proper ordering
    current_step = env.common_step_counter
    applied_weight = None

    for step, weight in steps_and_weights:
        if current_step > step:
            applied_weight = weight
        else:
            break

    if applied_weight is not None:
        penalty = env.reward_manager.get_term_cfg(term_name)
        if penalty.weight != applied_weight:
            penalty.weight = applied_weight
            env.reward_manager.set_term_cfg(term_name, penalty)


def increase_actuator_gains(env: MazeEnv, env_ids: Sequence[int]):
    current_step = env.common_step_counter
    event_term = env.event_manager.get_term_cfg("randomize_outer_actuator")
    stiffness = event_term.params["stiffness_distribution_params"]
    # damping = event_term.params["damping_distribution_params"]
    if current_step > 2000:
        stiffness = (0.9, 1.1)
        damping = (0.9, 1.1)
        event_term.params["stiffness_distribution_params"] = stiffness
        event_term.params["damping_distribution_params"] = damping
        env.event_manager.set_term_cfg("randomize_outer_actuator", event_term)


def increase_joint_friction(env: MazeEnv, env_ids: Sequence[int]):
    current_step = env.common_step_counter
    event_term = env.event_manager.get_term_cfg("randomize_outer_joint")
    if current_step > 2000:
        friction = (0.0, 0.1)
        event_term.params["friction_distribution_params"] = friction
        env.event_manager.set_term_cfg("randomize_outer_joint", event_term)


def update_domain_randomization(env: MazeEnv, env_ids: Sequence[int]):
    env.randomizer.step(env_ids)
    env.update_delay(env_ids)
