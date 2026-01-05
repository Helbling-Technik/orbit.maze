# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Maze-v0", help="Name of the task.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--maze_start_point", type=int, default=0, help="Negative = random, 0-len(path), will be clipped to max length"
)
parser.add_argument(
    "--frames_per_second", type=int, default=30, help="Update frames per second of observation and action"
)
parser.add_argument("--debug_images", action="store_true", default=False, help="Output debug images of camera")
parser.add_argument("--real_maze", action="store_true", default=False, help="For real maze usd")
parser.add_argument("--pos_ctrl", action="store_true", default=False, help="Position control, default is torque")
parser.add_argument("--use_pid", action="store_true", default=False, help="Use same PID as real hardware for actuation")
parser.add_argument(
    "--velocity_obs", action="store_true", default=False, help="Use velocity observation, default false"
)
parser.add_argument("--record_path_score", action="store_true", default=False, help="Log path score for each env")
parser.add_argument(
    "--multi_maze", action="store_true", default=False, help="Multi maze environment, has --real_maze inherently"
)
parser.add_argument(
    "--synced_obs_delay", action="store_true", default=False, help="Use synced delay for observations, no randomization"
)
# TODO ROV make these in levels
parser.add_argument(
    "--delay_level",
    type=int,
    choices=[-1, 0, 1],
    default=-1,
    help="Use delay for observation & motor commands: -1 no, 0 small, 1 large",
)
parser.add_argument(
    "--ext_force_level",
    type=int,
    choices=[-1, 0, 1],
    default=-1,
    help="Apply ext force on sphere: -1 no, 0 small, 1 large",
)
parser.add_argument(
    "--joint_friction_level",
    type=int,
    choices=[-1, 0, 1],
    default=-1,
    help="Apply joint friction: -1 no, 0 small, 1 large",
)
parser.add_argument(
    "--actuator_gain_level",
    type=int,
    choices=[-1, 0, 1],
    default=-1,
    help="Apply actuator gain: -1 no, 0 small, 1 large",
)
parser.add_argument(
    "--randomization_level",
    type=int,
    choices=[-1, 0, 1],
    default=None,
    help="Overwrites actuator gain and joint friction: -1 no, 0 small, 1 large",
)

parser.add_argument(
    "--set_params", action="store_true", default=False, help="Set domain parameters. Primary use for evaluation"
)
# specify model to use here, it is advised to use one which has not overfitted
parser.add_argument(
    "--checkpoint",
    type=str,
    # default="logs/sb3/Isaac-Maze-v0/2025-07-28_17-11-29/model_372736000_steps.zip"
    default="logs/sb3/Isaac-Maze-v0/2025-11-28_16-36-12/model_2211840000_steps.zip",
    # default="logs/sb3/Isaac-Maze-v0/2025-08-11_11-32-26/model_1437696000_steps.zip",
    help="Path to model checkpoint.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

import globals


def set_globals(args):
    # Need to initialize these for proper env config
    if args_cli.debug_images:
        globals.debug_images = True
    if args_cli.real_maze:
        globals.real_maze = True
    if args_cli.pos_ctrl:
        globals.position_control = True
    if args_cli.use_pid:
        globals.use_pid = True
    if args_cli.velocity_obs:
        globals.velocity_obs = True
    if args_cli.record_path_score:
        globals.record_path_score = True

    globals.actuator_gain_level = args_cli.actuator_gain_level
    globals.joint_friction_level = args_cli.joint_friction_level

    if args_cli.randomization_level is not None:
        globals.actuator_gain_level = args_cli.randomization_level
        globals.joint_friction_level = args_cli.randomization_level

    globals.ext_force_level = args_cli.ext_force_level
    globals.delay_level = args_cli.delay_level
    globals.synced_obs_delay = args_cli.synced_obs_delay
    globals.targeted_frequency = args_cli.frames_per_second

    # Init globals before everything else
    if args_cli.multi_maze:
        globals.use_multi_maze = True
        globals.init_multi_usd()
    else:
        globals.init_single_usd()

    globals.set_params = args_cli.set_params


set_globals(args_cli)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import VecEnv, Sb3VecEnvWrapper, process_sb3_cfg
import orbit.maze  # noqa: F401


def make_env(args) -> VecEnv:
    # parse configuration
    env_cfg = parse_env_cfg(
        args.task,
        device="cuda:0",
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )

    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device

    # override maze_start_point
    if args.maze_start_point is not None:
        globals.init_maze_start_point(args.maze_start_point)

    agent_cfg = load_cfg_from_registry(args.task, "sb3_cfg_entry_point")
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    # create isaac environment
    env = gym.make(args.task, cfg=env_cfg)
    # Bound the actions!
    env.unwrapped.single_action_space = gym.spaces.Box(low=-1, high=1, shape=env.unwrapped.single_action_space.shape)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=False,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    return env


def make_agent(env: VecEnv, args) -> PPO:
    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args.task)
    log_root_path = os.path.abspath(log_root_path)
    # check checkpoint is valid
    if args.checkpoint is None:
        if args.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args.checkpoint

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)
    total_params = sum(p.numel() for p in agent.policy.parameters())
    print(agent.policy)
    print(f"Total number of parameters in the model: {total_params}")

    return agent


def run_evaluation_loop(env: VecEnv, agent: PPO, num_envs: int, num_episodes: int | None):
    # reset environment
    obs = env.reset()

    if num_episodes is not None:
        scores = torch.zeros((num_envs, num_episodes))
        episodes_per_env = num_episodes
        episode_counts = [0] * num_envs

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            prev_scores = globals.path_accumulated.clone()
            actions, _ = agent.predict(obs, deterministic=True)
            # actions = np.zeros_like(actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)

        if num_episodes:
            for i in range(num_envs):
                if dones[i] and episode_counts[i] < episodes_per_env:
                    scores[i, episode_counts[i]] = prev_scores[i]
                    print(f"[Env {i}] Episode {episode_counts[i]} done")
                    print(f"Score of env {i}: {prev_scores[i]}")
                    # print(f"Scores : {scores}")
                    episode_counts[i] += 1
            if all(c >= episodes_per_env for c in episode_counts):
                break
    return scores


def evaluate_policy(args):
    env = make_env(args)
    agent = make_agent(env, args)

    num_envs = args.num_envs
    num_episodes = args.num_episodes
    scores = run_evaluation_loop(env, agent, num_envs, num_episodes)
    if scores is not None:
        mean_score = torch.mean(scores[:, :num_episodes])
        median_score = torch.median(scores[:, :num_episodes])
        max_score = torch.max(scores[:, :num_episodes])
        print(f"FINAL_MEAN_SCORE: {mean_score:.4f}", flush=True)
        print(f"FINAL_MEDIAN_SCORE: {median_score:.4f}", flush=True)
        print(f"FINAL_MAX_SCORE: {max_score:.4f}", flush=True)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    evaluate_policy(args_cli)
    # close sim app
    simulation_app.close()
