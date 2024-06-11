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
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


@configclass
class PathTracker:
    def __init__(self):
        self.path_idx = None
        self.maze_path = None

    def path_point_target(
        self,
        env: RLTaskEnv,
        target1_cfg: SceneEntityCfg,
        target2_cfg: SceneEntityCfg,
        target3_cfg: SceneEntityCfg,
        sphere_cfg: SceneEntityCfg,
        distance_from_target: float = 0.005,
    ) -> torch.Tensor:
        """Asset root position in the environment frame."""
        # extract the used quantities (to enable type-hinting)
        sphere: RigidObject = env.scene[sphere_cfg.name]
        target1: RigidObject = env.scene[target1_cfg.name]
        target2: RigidObject = env.scene[target2_cfg.name]
        target3: RigidObject = env.scene[target3_cfg.name]
        sphere_pos = sphere.data.root_pos_w - env.scene.env_origins
        target1_pos = target1.data.root_pos_w - env.scene.env_origins

        if self.path_idx is None:
            self.path_idx = 2 * torch.ones(env.num_envs, device=sphere.device, dtype=int)
            self.maze_path = self.maze_path.to(sphere.device)
            self.path_idx = self.path_idx.to(sphere.device)

        xy_sparse_reward = torch.norm(sphere_pos[:, :2] - target1_pos[:, :2], dim=1) < distance_from_target
        target_reached_ids = torch.nonzero(xy_sparse_reward).view(-1)
        if target_reached_ids.numel() == 0:
            return xy_sparse_reward

        target2to1 = target2.data.root_state_w[target_reached_ids, :7].clone().squeeze(0)
        target3to2 = target3.data.root_state_w[target_reached_ids, :7].clone().squeeze(0)
        targetNextto3 = target3.data.root_state_w[target_reached_ids, :7].clone().squeeze(0)

        if targetNextto3.dim() == 1:
            targetNextto3 = targetNextto3.unsqueeze(0)

        # update the path index and last target
        self.path_idx[target_reached_ids] += 1
        updated_path_idx = torch.tensor(self.path_idx[target_reached_ids], device=sphere.device, dtype=int)
        targetNext = self.maze_path[updated_path_idx, :]
        targetNextto3[:, :2] = targetNext + env.scene.env_origins[target_reached_ids, :2]

        print("-----------------")
        print("--path_idx--: ", self.path_idx[target_reached_ids])
        print("targetNextto3", targetNextto3)
        print("updated_path_idx", updated_path_idx)
        target1.write_root_pose_to_sim(target2to1, env_ids=target_reached_ids)
        target2.write_root_pose_to_sim(target3to2, env_ids=target_reached_ids)
        target3.write_root_pose_to_sim(targetNextto3, env_ids=target_reached_ids)

        print("target1: ", target1.data.root_pos_w[0, :2])
        print("target2: ", target2.data.root_pos_w[0, :2])
        print("target3: ", target3.data.root_pos_w[0, :2])

        return xy_sparse_reward

    def reset_maze_path_idx(self, env: BaseEnv, env_ids: torch.Tensor, sphere_cfg: SceneEntityCfg):
        sphere: RigidObject = env.scene[sphere_cfg.name]
        if self.path_idx is None:
            self.path_idx = 2 * torch.ones(env.num_envs, device=sphere.device, dtype=int)
            self.maze_path = self.maze_path.to(sphere.device)
            self.path_idx = self.path_idx.to(sphere.device)
        print("-----------------")
        print("path_idx: ", self.path_idx)
        print("Resetting path index, env_ids: ", env_ids)
        self.path_idx[env_ids] = 2 * torch.ones(len(env_ids), device=sphere.device, dtype=int)
        self.path_idx = self.path_idx.clone().to(sphere.device)
        print("path_idx: ", self.path_idx)
        print("-----------------")


def root_xypos_target(
    env: RLTaskEnv, target_cfg: SceneEntityCfg | dict[str, float], asset_cfg: SceneEntityCfg, LNorm: int = 2
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if isinstance(target_cfg, SceneEntityCfg):
        target: RigidObject = env.scene[target_cfg.name]
        target_pos = target.data.root_pos_w - env.scene.env_origins
    else:
        target_pos = torch.tensor([target.get(key, 0.0) for key in ["x", "y"]], device=asset.data.root_pos_w.device)

    root_pos = asset.data.root_pos_w - env.scene.env_origins
    # compute the reward
    xy_reward_l2 = torch.norm(root_pos[:, :2] - target_pos[:, :2], p=LNorm, dim=1)
    return xy_reward_l2


def spline_point_target(
    env: RLTaskEnv,
    target1_cfg: SceneEntityCfg,
    target2_cfg: SceneEntityCfg,
    target3_cfg: SceneEntityCfg,
    sphere_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    distance_from_target: float = 0.005,
) -> torch.Tensor:
    """Asset root position in the environment frame."""

    # extract the used quantities (to enable type-hinting)
    sphere: RigidObject = env.scene[sphere_cfg.name]
    target1: RigidObject = env.scene[target1_cfg.name]
    target2: RigidObject = env.scene[target2_cfg.name]
    target3: RigidObject = env.scene[target3_cfg.name]
    sphere_pos = sphere.data.root_pos_w - env.scene.env_origins
    target1_pos = target1.data.root_pos_w - env.scene.env_origins

    xy_sparse_reward = torch.norm(sphere_pos[:, :2] - target1_pos[:, :2], dim=1) < distance_from_target
    target_reached_ids = torch.nonzero(xy_sparse_reward).view(-1)
    if target_reached_ids.numel() == 0:
        return xy_sparse_reward

    # resample the target pose for the reached ids
    range_size = (len(target_reached_ids), 3)
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=sphere.device)

    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], range_size, device=sphere.device)

    target2to1 = target2.data.root_state_w[target_reached_ids, :7].clone().squeeze(0)
    target3to2 = target3.data.root_state_w[target_reached_ids, :7].clone().squeeze(0)

    target1.write_root_pose_to_sim(target2to1, env_ids=target_reached_ids)
    target2.write_root_pose_to_sim(target3to2, env_ids=target_reached_ids)

    # sample new position
    new_pos = (
        target3.data.default_root_state[target_reached_ids, :3]
        + env.scene.env_origins[target_reached_ids, :]
        + rand_samples
    )
    # orientation
    new_ori = target3.data.default_root_state[target_reached_ids, 3:7]

    target3.write_root_pose_to_sim(torch.cat([new_pos, new_ori], dim=-1), env_ids=target_reached_ids)

    return xy_sparse_reward


def root_xy_sparse_target(
    env: RLTaskEnv,
    target_cfg: SceneEntityCfg | dict[str, float],
    asset_cfg: SceneEntityCfg,
    distance_from_target: float = 0.001,
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if isinstance(target_cfg, SceneEntityCfg):
        target: RigidObject = env.scene[target_cfg.name]
        target_pos = target.data.root_pos_w - env.scene.env_origins
    else:
        target_pos = torch.tensor([target.get(key, 0.0) for key in ["x", "y"]], device=asset.data.root_pos_w.device)

    root_pos = asset.data.root_pos_w - env.scene.env_origins
    # compute the reward
    xy_sparse_reward = torch.norm(root_pos[:, :2] - target_pos[:, :2], dim=1) < distance_from_target
    return xy_sparse_reward
