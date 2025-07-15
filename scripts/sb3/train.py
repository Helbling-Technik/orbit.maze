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
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Maze-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--maze_start_point", type=int, default=0, help="Negative = random, 0-len(path), will be clipped to max length")
parser.add_argument("--frames_per_second", type=int, default=30, help="Update frames per second of observation and action")
parser.add_argument("--overwrite_n_timesteps", type=float, default=None, help="If specified overwrite n_timesteps of training config")
parser.add_argument("--debug_images", action="store_true", default=False, help="Output debug images of camera")
parser.add_argument("--real_maze", action="store_true", default=False, help="For real maze usd")
parser.add_argument("--pos_ctrl", action="store_true", default=False, help="Position control, default is torque")
parser.add_argument("--use_pid", action="store_true", default=False, help="Use same PID as real hardware for actuation")
parser.add_argument("--velocity_obs", action="store_true", default=False, help="Use velocity observation, default false")
parser.add_argument("--multi_maze", action="store_true", default=False, help="Multi maze environment, has --real_maze inherently")
parser.add_argument("--synced_obs_delay", action="store_true", default=False, help="Use synced delay for observations, no randomization")

# TODO ROV make these in levels
parser.add_argument("--delay_level", type=int, choices=[-1, 0, 1], default=-1, help="Use delay for observation & motor commands: -1 no, 0 small, 1 large")
parser.add_argument("--ext_force_level", type=int, choices=[-1, 0, 1], default=-1, help="Apply ext force on sphere: -1 no, 0 small, 1 large")
parser.add_argument("--joint_friction_level", type=int, choices=[-1, 0, 1], default=-1, help="Apply joint friction: -1 no, 0 small, 1 large")
parser.add_argument("--actuator_gain_level", type=int, choices=[-1, 0, 1], default=-1, help="Apply actuator gain: -1 no, 0 small, 1 large")
parser.add_argument("--randomization_level", type=int, choices=[-1, 0, 1], default=None, help="Overwrites actuator gain and joint friction: -1 no, 0 small, 1 large")

# specify a starting model here, it is advised to use one which has not overfitted
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
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
if args_cli.use_pid:
    globals.use_pid = True
if args_cli.velocity_obs:
    globals.velocity_obs = True

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


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import orbit.maze  # noqa: F401  TODO: import orbit.<your_extension_name>
import json

from orbit.maze.tasks.maze.agents import helbling_combined_extractor


def serialize_config(config):
    """Recursively serialize a configuration object."""
    if isinstance(config, dict):
        # Handle nested dictionaries
        return {k: serialize_config(v) for k, v in config.items()}
    elif hasattr(config, "__dict__"):
        # Handle objects with __dict__ attribute
        return serialize_config(config.__dict__)
    elif isinstance(config, (list, tuple)):
        # Handle lists and tuples
        return [serialize_config(item) for item in config]
    elif isinstance(config, (str, int, float, bool, type(None))):
        # Handle basic types
        return config
    else:
        # Fallback for non-serializable objects
        return str(config)


def main():
    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
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

    # Save agent configuration
    agent_cfg_path = os.path.join(log_dir, "agent_config.json")
    with open(agent_cfg_path, "w") as f:
        json.dump(agent_cfg, f, indent=4)

    # Save environment configuration
    env_cfg_path = os.path.join(log_dir, "env_config.json")
    with open(env_cfg_path, "w") as f:
        json.dump(serialize_config(env_cfg), f, indent=4)

    # Convert Namespace to dictionary
    args_dict = vars(args_cli)

    # Define output file
    argparse_cmd_path = os.path.join(log_dir, "parsed_arguments.json")

    # Save to file in JSON format
    with open(argparse_cmd_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    if args_cli.overwrite_n_timesteps:
        n_timesteps = args_cli.overwrite_n_timesteps

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Bound the actions!
    env.unwrapped.single_action_space = gym.spaces.Box(low=-1, high=1, shape=env.unwrapped.single_action_space.shape)
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

    # create custom object
    policy_kwargs = dict(
        features_extractor_class=helbling_combined_extractor.CustomCombinedExtractor,
        features_extractor_kwargs=dict(normalized_image=True),
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
        # TODO ROV seems worse
        # share_features_extractor=False,  # False: 444517, True: 373429
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
        agent = PPO(policy_arch, env, verbose=1, policy_kwargs=policy_kwargs, **agent_cfg)

    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)
    print(agent.policy)
    total_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"Total number of parameters in the model: {total_params}")

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    if "normalize_input" in agent_cfg:
        env.save(os.path.join(log_dir, "vecnormalize.pkl"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
