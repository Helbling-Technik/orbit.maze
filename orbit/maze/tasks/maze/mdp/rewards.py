# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def joint_pos_target_l2(env: RLTaskEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    # print("joint pos reward: ", torch.sum(torch.square(joint_pos - target), dim=1))
    return torch.sum(torch.square(joint_pos - target), dim=1)


def root_pos_target_l2(env: RLTaskEnv, target: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    target_list = torch.tensor([target.get(key, 0.0) for key in ["x", "y", "z"]], device=asset.data.root_pos_w.device)
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    # compute the reward
    return torch.sum(torch.square(root_pos - target_list), dim=1)


def root_xypos_target_l2(env: RLTaskEnv, target: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    target_tensor = torch.tensor([target.get(key, 0.0) for key in ["x", "y"]], device=asset.data.root_pos_w.device)
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    # compute the reward
    # xy_reward_l2 = (torch.sum(torch.square(root_pos[:,:2] - target_tensor), dim=1) <= 0.0025).float()*2 - 1
    xy_reward_l2 = torch.sum(torch.square(root_pos[:, :2] - target_tensor), dim=1)
    # print("sphere_xypos_rewards: ", xy_reward_l2.tolist())
    return xy_reward_l2


def object_goal_distance_l2(
    env: RLTaskEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # command_pos is difference between target in env frame and object in env frame
    command_pos = command[:, :2]
    object_pos = object.data.root_pos_w - env.scene.env_origins
    # print("target_pos: ", command_pos)
    # print("object_pos: ", object_pos[:, :2])
    # distance of the target to the object: (num_envs,)
    distance = torch.norm(command_pos, dim=1)
    # print("distance: ", distance)
    # rewarded if the object is closest to the target
    return distance
