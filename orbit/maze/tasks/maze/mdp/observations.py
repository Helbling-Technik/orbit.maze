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

from PIL import Image
from torchvision import transforms
from datetime import datetime


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


class VelocityExtractor:
    def __init__(self):
        self.previous_root_pos = None
        self.previous_joint_pos = None

    def extract_root_velocity(self, env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Extract the velocity of the object."""
        asset: RigidObject = env.scene[asset_cfg.name]
        current_root_pos = asset.data.root_pos_w - env.scene.env_origins

        if self.previous_root_pos is None:
            self.previous_root_pos = current_root_pos
            return torch.zeros_like(current_root_pos)

        current_vel = (current_root_pos - self.previous_root_pos) / env.step_dt

        self.previous_root_pos = current_root_pos

        return current_vel

    def extract_joint_velocity(self, env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Extract the velocity of the object."""
        asset: Articulation = env.scene[asset_cfg.name]
        current_joint_pos = torch.clone(asset.data.joint_pos[:, asset_cfg.joint_ids])

        if self.previous_joint_pos is None:
            self.previous_joint_pos = current_joint_pos
            print("previous_joint_pos is None")
            return torch.zeros_like(current_joint_pos)

        current_joint_vel = (current_joint_pos - self.previous_joint_pos) / env.step_dt

        self.previous_joint_pos = current_joint_pos

        return current_joint_vel


def joint_pos_with_noise(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.0
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    noise_tensor = torch.normal(mean=0, std=std, size=joint_pos.shape).to(joint_pos.device)
    return joint_pos + noise_tensor


def root_pos_w_with_noise(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.0
) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w - env.scene.env_origins
    noise_tensor = torch.normal(mean=0, std=std, size=asset.data.root_pos_w.shape).to(robot_pos.device)
    return robot_pos + noise_tensor


def camera_image(env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera image from top camera."""
    # extract the used quantities (to enable type-hinting)
    # asset: Articulation = env.scene[asset_cfg.name]
    asset: Camera = env.scene[asset_cfg.name]

    print("saving image[0] to logs/sb3/Isaac-Maze-v0/test-images/output_image.png")
    now = datetime.now()
    image = Image.fromarray(asset.data.output["rgb"][0, :, :, :3].cpu().numpy())
    date_string = now.strftime("%Y%m%d-%H%M%S")
    image.save("logs/sb3/Isaac-Maze-v0/test-images/output_image_" + date_string + ".png")

    # Assuming asset and its data are properly defined and initialized
    n_envs = asset.data.output["rgb"].size(0)
    n = int(asset.data.output["rgb"].numel() / n_envs)
    return asset.data.output["rgb"].view(n_envs, n)


def root_pos_w_xy(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w - env.scene.env_origins
    return position[:, :2]
