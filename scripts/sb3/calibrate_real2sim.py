# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to tune real2sim of labyrinth actuation."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tune real2sim of labyrinth actuation. Run like this \"python scripts/sb3/calibrate_real2sim.py --headless --pos_ctrl --delay_level 0\"")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Maze-Calibration-v0", help="Name of the task.")
parser.add_argument(
    "--frames_per_second", type=int, default=30, help="Update frames per second of observation and action"
)
parser.add_argument("--pos_ctrl", action="store_true", default=False, help="Position control, default is torque")
parser.add_argument("--use_pid", action="store_true", default=False, help="Use same PID as real hardware for actuation")

# TODO ROV make these in levels
parser.add_argument(
    "--delay_level",
    type=int,
    choices=[-1, 0, 1],
    default=-1,
    help="Use delay for motor commands: -1 no, 0 small, 1 large",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

import globals

# Need to initialize these for proper env config
globals.real_maze = True
if args_cli.pos_ctrl:
    globals.position_control = True
if args_cli.use_pid:
    globals.use_pid = True

globals.delay_level = args_cli.delay_level
globals.targeted_frequency = args_cli.frames_per_second

# Init globals before everything else
globals.init_single_usd()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
import orbit.maze  # noqa: F401

import matplotlib.pyplot as plt
import math
import time


# TODO ROV I need to create own env cfg and kick out everything I do not need
class Calibration:
    def __init__(self, frequency_controller=30.0, duration=4.0, frequency_sinusoid=0.5, pad_time=1.0):
        """
        frequency_sinusoid: Hz for sinusoidal signals
        duration: seconds for each segment
        frequency_controller: frequency of the controller
        pad_time: zero-padding duration (in seconds) between sequences
        """
        self.dt = 1 / frequency_controller
        self.actions = self.generate_actions(frequency_sinusoid, duration, self.dt, pad_time)
        self.step_counter = 0
        self.logged_maze_angles = []
        self.logged_scaled_actions = []
        self.logged_times = []

        self._plot_actions()
        self._logs_saved = False

    def logging(self, observed_maze_angles_raw: np.ndarray, actions: np.ndarray):
        self.logged_maze_angles.append(observed_maze_angles_raw)
        self.logged_scaled_actions.append(actions)
        self.logged_times.append(time.time())

    def save_logged_data(self, data_file="logs/calibration/"):
        if not self.logged_maze_angles or not self.logged_scaled_actions:
            raise ValueError("No logged data to save.")
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        np.save(data_file + "observed_angles.npy", np.array(self.logged_maze_angles))
        np.save(data_file + "scaled_actions.npy", np.array(self.logged_scaled_actions))
        np.save(data_file + "timestamps.npy", np.array(self.logged_times))

    def get_next_action(self):
        if self.step_counter < self.actions.shape[0]:
            next_action = self.actions[self.step_counter]
            self.step_counter += 1
            return next_action
        else:
            print("Calibration actions finished")
            if self.logged_maze_angles and not self._logs_saved:
                self.save_logged_data()
                self._plot_logged_data()
                self._logs_saved = True
            return None

    def generate_actions(self, frequency, duration, dt, pad_time):
        t = np.arange(0, duration, dt)
        zeros_pad = np.zeros((int(pad_time / dt), 2))

        # Sinusoidal segments
        sin_x = np.column_stack((np.sin(2 * np.pi * frequency * t), np.zeros_like(t)))
        sin_y = np.column_stack((np.zeros_like(t), np.sin(2 * np.pi * frequency * t)))
        sin_xy = np.column_stack((np.sin(2 * np.pi * frequency * t), np.sin(2 * np.pi * frequency * t)))

        # Step helper
        def step_array(value):
            return np.full((int(duration / dt), 2), value, dtype=float)

        # Step sequences
        step_x_pos = step_array([1, 0])
        step_x_neg = step_array([-1, 0])
        step_y_pos = step_array([0, 1])
        step_y_neg = step_array([0, -1])
        step_xy_pos = step_array([1, 1])
        step_xy_neg = step_array([-1, -1])

        # Combine all with zero padding
        actions = np.vstack([
            zeros_pad,
            sin_x,
            zeros_pad,
            sin_y,
            zeros_pad,
            sin_xy,
            zeros_pad,
            step_x_pos,
            zeros_pad,
            step_x_neg,
            zeros_pad,
            step_y_pos,
            zeros_pad,
            step_y_neg,
            zeros_pad,
            step_xy_pos,
            zeros_pad,
            step_xy_neg,
            zeros_pad,
        ])

        return actions

    def _plot_actions(self, data_file="logs/calibration/"):
        if not hasattr(self, "actions") or self.actions is None:
            raise ValueError("No actions available to plot.")

        n_steps = self.actions.shape[0]
        time = np.arange(n_steps) * self.dt

        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # X actions
        ax1.plot(time, self.actions[:, 0], color="blue")
        ax1.set_ylabel("X Action")
        ax1.set_ylim([-1.1, 1.1])
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.set_title("Calibration Action Sequences")

        # Y actions
        ax2.plot(time, self.actions[:, 1], color="orange")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Y Action")
        ax2.set_ylim([-1.1, 1.1])
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(data_file + "action_sequences.png")
        plt.close()

        np.save(data_file + "actions.npy", np.array(self.actions))

    def _plot_logged_data(self, output_file="logs/calibration/observed_angles.png"):
        if not self.logged_maze_angles:
            raise ValueError("No logged data to plot.")

        data_angles = np.rad2deg(np.array(self.logged_maze_angles))
        data_scaled_actions = np.rad2deg(np.array(self.logged_scaled_actions))
        timestamps = np.array(self.logged_times)
        time = timestamps - timestamps[0]

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(time, data_angles[:, 0], color="blue")
        ax1.plot(time, data_scaled_actions[:, 0], color="green")
        ax1.set_ylabel("X Angle [deg]")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.set_title("Observed Maze Angles")

        ax2.plot(time, data_angles[:, 1], color="orange")
        ax2.plot(time, data_scaled_actions[:, 1], color="red")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Y Angle [deg]")
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close(fig)


class CalibrationController:
    def __init__(self, frequency: float = 30.0):
        self.calibration_actions = Calibration(frequency)

    def update(self, observation):
        print(observation['mlp_policy'][0])
        
        actions = self.calibration_actions.get_next_action()
        if actions is None:
            return np.zeros((1, 2)), False

        scaled_actions = np.zeros_like(actions)

        outer_joint_scaling = 3 * math.pi / 180
        inner_joint_scaling = 3 * math.pi / 180

        scaled_actions[0] = actions[0] * outer_joint_scaling
        scaled_actions[1] = actions[1] * inner_joint_scaling

        if observation:
            obs = observation['mlp_policy'][0]
            self.calibration_actions.logging([obs[0], obs[1]], scaled_actions)

        return actions.reshape(1, 2), True


def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Bound the actions!
    env.unwrapped.single_action_space = gym.spaces.Box(low=-1, high=1, shape=env.unwrapped.single_action_space.shape)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

    # create calibration controller
    calibration_controller = CalibrationController()

    # reset environment
    obs = env.reset()
    # simulate environment
    calibration_ongoing = True
    while simulation_app.is_running() and calibration_ongoing:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, calibration_ongoing = calibration_controller.update(obs)
            # actions = np.zeros_like(actions)
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
