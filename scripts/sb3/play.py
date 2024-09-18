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
parser.add_argument("--task", type=str, default="Isaac-Maze-v0", help="Name of the task.")
# specify model to use here, it is advised to use one which has not overfitted
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/sb3/Isaac-Maze-v0/2024-09-18_07-54-25/model_122880000_steps.zip",
    help="Path to model checkpoint.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--maze_start_point", type=int, default=0, help="Negative = random, 0-len(path), will be clipped to max length"
)
parser.add_argument("--debug_images", action="store_true", default=False, help="Output debug images of camera")
parser.add_argument("--real_maze", action="store_true", default=False, help="For real maze usd")
parser.add_argument("--pos_ctrl", action="store_true", default=False, help="Position control, default is torque")
parser.add_argument(
    "--delay", action="store_true", default=False, help="Add delay to observation & randomized longer delay"
)
parser.add_argument("--ext_force", action="store_true", default=False, help="Add random external force to sphere")
parser.add_argument(
    "--multi_maze",
    action="store_true",
    default=False,
    help="Multi maze environment, has --real_maze inherently",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

import globals

# Need to initialize these for proper env config
if args_cli.debug_images:
    globals.debug_images = True
if args_cli.real_maze:
    globals.real_maze = True
if args_cli.pos_ctrl:
    globals.position_control = True
if args_cli.delay:
    globals.use_delay = True
if args_cli.ext_force:
    globals.use_force = True

# Init globals before everything else
if args_cli.multi_maze:
    globals.use_multi_maze = True
    globals.init_multi_usd()
else:
    globals.init_single_usd()


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
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import orbit.maze  # noqa: F401


def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        # TODO ROV device="cuda:0" instead of use_gpu
        args_cli.task,
        use_gpu=True,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # override maze_start_point
    if args_cli.maze_start_point is not None:
        globals.init_maze_start_point(args_cli.maze_start_point)

    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
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

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    # check checkpoint is valid
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)

    total_params = sum(p.numel() for p in agent.policy.parameters())
    print(agent.policy)
    print(f"Total number of parameters in the model: {total_params}")

    # reset environment
    obs = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
