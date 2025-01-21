# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Maze-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
# specify a starting model here, it is advised to use one which has not overfitted
# TODO ROV currently training with high delay, 20 Hz, no ext force on real maze. Might be good to use that one then for ext force
parser.add_argument(
    "--model_path",
    type=str,
    default=None,  # "logs/sb3/Isaac-Maze-v0/2024-11-21_11-11-23_25Hz_2x_img_length_4x_crop_length/model_40960000_steps.zip",
    # "logs/sb3/Isaac-Maze-v0/2024-10-25_14-46-22_25_Hz_increased_actuator_rand_longerTraining/model.zip",
    # logs/sb3/Isaac-Maze-v0/2024-10-11_08-45-31_friction_force_on_reset_delay_realmaze/model_98304000_steps.zip
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
from datetime import datetime

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import orbit.maze  # noqa: F401  TODO: import orbit.<your_extension_name>


def main():
    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        # TODO ROV device="cuda:0" instead of use_gpu
        args_cli.task,
        # use_gpu=True,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # override configuration with command line arguments
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    # override maze_start_point
    if args_cli.maze_start_point is not None:
        globals.init_maze_start_point(args_cli.maze_start_point)

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # set the seed
    env.seed(seed=agent_cfg["seed"])

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # Check if a model path is provided
    if args_cli.model_path:
        model_path = os.path.abspath(args_cli.model_path)
        if os.path.isfile(model_path):
            # Load the existing model
            agent = PPO.load(args_cli.model_path, env=env)
            print(f"[INFO] Loaded existing model from {args_cli.model_path}")
    else:
        # Create a new agent from scratch
        agent = PPO(policy_arch, env, verbose=1, **agent_cfg)

    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)
    print(agent.policy)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path=log_dir, name_prefix="model", verbose=2)
    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
