# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedEnv
import omni.isaac.lab.utils.math as math_utils

import globals

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def path_point_target(
    env: ManagerBasedRLEnv,
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

    path_length = globals.maze_path.shape[0]

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
    globals.path_idx[target_reached_ids] += globals.path_direction[target_reached_ids]

    # check out of bounds and set last target point to beginning/end of path, will propagate to the others in next target reached
    globals.path_idx[globals.path_idx < 0] = 0
    globals.path_idx[globals.path_idx >= path_length] = path_length - 1

    updated_path_idx = torch.tensor(globals.path_idx[target_reached_ids], device=sphere.device, dtype=int)
    targetNext = globals.maze_path[updated_path_idx, :]
    targetNextto3[:, :2] = targetNext + env.scene.env_origins[target_reached_ids, :2]

    target1.write_root_pose_to_sim(target2to1, env_ids=target_reached_ids)
    target2.write_root_pose_to_sim(target3to2, env_ids=target_reached_ids)
    target3.write_root_pose_to_sim(targetNextto3, env_ids=target_reached_ids)
    return xy_sparse_reward


def reset_maze_path_idx(env: ManagerBasedEnv, env_ids: torch.Tensor, sphere_cfg: SceneEntityCfg):
    sphere: RigidObject = env.scene[sphere_cfg.name]

    if globals.path_idx is None:
        globals.path_idx = 2 * torch.ones(env.num_envs, device=sphere.device, dtype=int)
        globals.maze_path = globals.maze_path.to(sphere.device)
        globals.path_idx = globals.path_idx.to(sphere.device)

    globals.path_idx[env_ids] = 2 * torch.ones(len(env_ids), device=sphere.device, dtype=int)
    globals.path_idx = globals.path_idx.clone().to(sphere.device)


def reset_maze_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    target1_cfg: SceneEntityCfg,
    target2_cfg: SceneEntityCfg,
    target3_cfg: SceneEntityCfg,
    sphere_cfg: SceneEntityCfg,
):

    sphere: RigidObject = env.scene[sphere_cfg.name]
    target1: RigidObject = env.scene[target1_cfg.name]
    target2: RigidObject = env.scene[target2_cfg.name]
    target3: RigidObject = env.scene[target3_cfg.name]

    path_length = globals.maze_path.shape[0]

    if globals.path_idx is None:
        if globals.maze_start_point < 0:
            globals.path_idx = math_utils.sample_uniform(0, path_length, len(env_ids), device=sphere.device).to(
                torch.int
            )
        else:
            globals.path_idx = globals.maze_start_point * torch.ones(
                len(env_ids), device=sphere.device, dtype=torch.int
            )
        globals.maze_path = globals.maze_path.to(sphere.device)
        globals.path_direction = torch.ones(len(env_ids), device=sphere.device, dtype=torch.int)
        globals.path_direction[globals.path_idx >= int(path_length / 2)] = -1
        globals.path_idx = globals.path_idx.to(sphere.device)
        globals.path_direction = globals.path_direction.to(sphere.device)

    if globals.maze_start_point < 0:
        globals.path_idx[env_ids] = math_utils.sample_uniform(0, path_length, len(env_ids), device=sphere.device).to(
            torch.int
        )
    else:
        globals.path_idx[env_ids] = globals.maze_start_point * torch.ones(
            len(env_ids), device=sphere.device, dtype=torch.int
        )
    path_direction_temp = torch.zeros_like(globals.path_direction)
    path_direction_temp = 2 * (globals.path_idx < int(path_length / 2)) - 1
    path_direction_temp = path_direction_temp.to(torch.int)
    globals.path_direction[env_ids] = path_direction_temp[env_ids]

    # frmt = "{:>3}" * len(globals.path_direction)
    # print("path_dir", frmt.format(*globals.path_direction.tolist()), sep="\t")
    # print("globals.path_idx", frmt.format(*globals.path_idx.tolist()), sep="\t")

    sphere_pos = sphere.data.default_root_state[env_ids, :7].clone()
    target1_pos = sphere_pos.clone()
    target2_pos = target1_pos.clone()
    target3_pos = target1_pos.clone()

    sphere_pos[:, :2] = globals.maze_path[globals.path_idx[env_ids], :] + env.scene.env_origins[env_ids, :2]
    globals.path_idx[env_ids] = globals.path_idx[env_ids] + globals.path_direction[env_ids]
    target1_pos[:, :2] = globals.maze_path[globals.path_idx[env_ids], :] + env.scene.env_origins[env_ids, :2]
    globals.path_idx[env_ids] = globals.path_idx[env_ids] + globals.path_direction[env_ids]
    target2_pos[:, :2] = globals.maze_path[globals.path_idx[env_ids], :] + env.scene.env_origins[env_ids, :2]
    globals.path_idx[env_ids] = globals.path_idx[env_ids] + globals.path_direction[env_ids]
    target3_pos[:, :2] = globals.maze_path[globals.path_idx[env_ids], :] + env.scene.env_origins[env_ids, :2]

    sphere.write_root_pose_to_sim(sphere_pos, env_ids=env_ids)
    sphere.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=sphere.device), env_ids=env_ids)
    target1.write_root_pose_to_sim(target1_pos, env_ids=env_ids)
    target3.write_root_pose_to_sim(target3_pos, env_ids=env_ids)
    target2.write_root_pose_to_sim(target2_pos, env_ids=env_ids)


def root_xypos_target(
    env: ManagerBasedRLEnv, target_cfg: SceneEntityCfg | dict[str, float], asset_cfg: SceneEntityCfg, LNorm: int = 2
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
    env: ManagerBasedRLEnv,
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
    env: ManagerBasedRLEnv,
    sphere_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg | dict[str, float],
    distance_from_target: float = 0.001,
    idx: int = None,
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    sphere: RigidObject = env.scene[sphere_cfg.name]

    if isinstance(target_cfg, SceneEntityCfg):
        target: RigidObject = env.scene[target_cfg.name]
        target_pos = target.data.root_pos_w - env.scene.env_origins
    else:
        target_pos = torch.tensor(
            [target_cfg.get(key, 0.0) for key in ["x", "y"]], device=sphere.data.root_pos_w.device
        )
        target_pos = target_pos.unsqueeze(0)

    root_pos = sphere.data.root_pos_w - env.scene.env_origins
    # compute the reward
    xy_sparse_reward = torch.norm(root_pos[:, :2] - target_pos[:, :2], dim=1) < distance_from_target

    reached_goal = idx * torch.ones_like(globals.path_idx) == globals.path_idx
    xy_sparse_reward = xy_sparse_reward * reached_goal

    return xy_sparse_reward
