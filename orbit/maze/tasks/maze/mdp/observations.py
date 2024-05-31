# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.sensors import Camera
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def camera_image(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera image from top camera."""
    # Extract the used quantities (to enable type-hinting)
    asset: Camera = env.scene[asset_cfg.name]

    # Get the RGBA image tensor
    rgba_tensor = asset.data.output["rgb"]

    # Check the shape of the input tensor
    assert rgba_tensor.dim() == 4 and rgba_tensor.size(-1) == 4, "Expected tensor of shape (n, 128, 128, 4)"

    # Ensure the tensor is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgba_tensor = rgba_tensor.to(device)

    # Convert the RGBA tensor to grayscale
    # Using the weights for R, G, and B, and ignoring the Alpha channel
    weights = torch.tensor([0.2989, 0.5870, 0.1140, 0.0], device=device).view(1, 1, 1, 4)
    grayscale_tensor = (rgba_tensor * weights).sum(dim=-1)

    # Flatten each image to a 1D tensor
    n_envs = grayscale_tensor.size(0)
    n_pixels = grayscale_tensor.size(1) * grayscale_tensor.size(2)
    grayscale_tensor_flattened = grayscale_tensor.view(n_envs, n_pixels)

    return grayscale_tensor_flattened


def get_target_pos(env: RLTaskEnv, target: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    # asset: RigidObject = env.scene[asset_cfg.name]
    # target_tensor = torch.tensor([target.get(key, 0.0) for key in ["x", "y"]], device=asset.data.root_pos_w.device)
    # root_pos = asset.data.root_pos_w - env.scene.env_origins
    zeros_tensor = torch.zeros_like(env.scene.env_origins)
    # return (zeros_tensor - root_pos)[:, :2].to(dtype=torch.float16)
    return zeros_tensor[:, :2]


def get_env_pos_of_command(env: RLTaskEnv, object_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    """The env frame target position can not fully be recovered as one of the terms is updated less frequently"""
    object: RigidObject = env.scene[object_cfg.name]
    object_pos = object.data.root_pos_w - env.scene.env_origins
    commanded = env.command_manager.get_command(command_name)
    target_pos_env = commanded[:, :2] + object_pos[:, :2]
    print("target_pos_env_observation: ", target_pos_env[:, :2])
    return target_pos_env


def get_generated_commands_xy(env: RLTaskEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commanded = env.command_manager.get_command(command_name)
    return commanded[:, :2]
