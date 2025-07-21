# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedEnv
import omni.isaac.lab.utils.math as math_utils
import random
import pandas as pd
import os

import globals

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def path_point_target(
    env: ManagerBasedRLEnv,
    target1_cfg: SceneEntityCfg,
    target2_cfg: SceneEntityCfg,
    target3_cfg: SceneEntityCfg,
    sphere_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    sphere: RigidObject = env.scene[sphere_cfg.name]
    target1: RigidObject = env.scene[target1_cfg.name]
    target2: RigidObject = env.scene[target2_cfg.name]
    target3: RigidObject = env.scene[target3_cfg.name]
    sphere_pos = sphere.data.root_pos_w - env.scene.env_origins
    target1_pos = target1.data.root_pos_w - env.scene.env_origins

    # change distance to target based on used maze
    # TODO ROV this should be able to differentiate between simple and hard real maze
    if globals.use_multi_maze:
        distance_tensor = globals.rew_dist_generated * torch.ones(
            (sphere_pos.shape[0]), dtype=torch.float16, device="cuda:0"
        )
        for env_idx in range(sphere_pos.shape[0]):
            if globals.get_list_entry_from_env(globals.maze_type_array, env_idx):
                distance_tensor[env_idx] = globals.rew_dist_real
        xy_sparse_reward = torch.norm(sphere_pos[:, :2] - target1_pos[:, :2], dim=1) < distance_tensor
    else:
        distance_from_target = globals.reward_distance
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

    # accumulate points crossed for evaluation metric
    globals.path_accumulated[target_reached_ids] += 1

    # check out of bounds and set last target point to beginning/end of path, will propagate to the others in next target reached
    globals.path_direction[globals.path_idx < 0] *= -1
    globals.path_idx[globals.path_idx < 0] = 0

    # get next target points
    if globals.use_multi_maze:
        targetNextList = []
        for reached_idx in target_reached_ids:
            path_maze = globals.get_list_entry_from_env(globals.maze_path_list, reached_idx)
            path_length = path_maze.shape[0]
            if globals.path_idx[reached_idx] >= path_length:
                globals.path_direction[reached_idx] *= -1
                globals.path_idx[reached_idx] = path_length - 1
            updated_path_idx = globals.path_idx[reached_idx]
            targetNextList.append(path_maze[updated_path_idx, :])
        targetNext = torch.stack(targetNextList, dim=0).to(sphere.device)
    else:
        path_length = globals.maze_path.shape[0]
        globals.path_direction[globals.path_idx >= path_length] *= -1
        globals.path_idx[globals.path_idx >= path_length] = path_length - 1

        updated_path_idx = globals.path_idx[target_reached_ids].clone().detach().to(device=sphere.device, dtype=int)
        targetNext = globals.maze_path[updated_path_idx, :]

    targetNextto3[:, :2] = targetNext + env.scene.env_origins[target_reached_ids, :2]

    target1.write_root_pose_to_sim(target2to1, env_ids=target_reached_ids)
    target2.write_root_pose_to_sim(target3to2, env_ids=target_reached_ids)
    target3.write_root_pose_to_sim(targetNextto3, env_ids=target_reached_ids)
    return xy_sparse_reward


def on_hole(
    env: ManagerBasedRLEnv,
    hole_radius: float,
    sphere_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
    maze_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize when the asset is on a hole, which would lead to a failure."""
    # extract the used quantities (to enable type-hinting)
    t_start = time.time()

    sphere: RigidObject = env.scene[sphere_cfg.name]
    maze: Articulation = env.scene[maze_cfg.name]

    maze_joint_pos = maze.data.joint_pos[:, maze_cfg.joint_ids]
    sphere_pos = sphere.data.root_link_pos_w - env.scene.env_origins
    sphere_pos = sphere_pos[:, :2] / torch.cos(maze_joint_pos)
    holes_pos = globals.holes_positions

    squared_dists = torch.sum((holes_pos.unsqueeze(0) - sphere_pos.unsqueeze(1)) ** 2, dim=2)
    is_on_hole = torch.any(squared_dists < hole_radius**2, dim=1).squeeze()

    env.hole_crossings += is_on_hole.float()

    just_crossed = is_on_hole & ~env.hole_crossed
    env.path_before_hole[just_crossed] = globals.path_accumulated[just_crossed].float()
    env.hole_crossed |= just_crossed
    env.path_before_hole[~env.hole_crossed] = globals.path_accumulated[~env.hole_crossed].float()

    t_end = time.time()
    duration = t_end - t_start
    # print("OnHole Reward computation duration: ", duration)

    return is_on_hole.float()


# def near_holes(
#     env: ManagerBasedRLEnv, hole_sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere")
# ) -> torch.Tensor:
#     """Penalize when the asset is on a hole, which would lead to a failure."""
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     sphere_pos = asset.data.root_link_pos_w[:, :2]

#     # TODO DRP Take joint angle in consideration
#     holes_pos = globals.holes_positions

#     squared_dists = torch.sum((holes_pos - sphere_pos) ** 2, dim=1)
#     return torch.exp(-squared_dists / (2 * hole_sigma**2))


def store_accumulated_path_score(env: ManagerBasedEnv, env_ids: torch.Tensor, file_path: str):
    if globals.path_accumulated is None:
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    env_ids = env_ids.cpu().numpy()
    path_accumulated = globals.path_accumulated[env_ids].cpu().numpy()

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
    else:
        num_envs = globals.path_accumulated.shape[0]
        df = pd.DataFrame(index=range(num_envs))

    for i, env_id in enumerate(env_ids):
        env_idx = env_id.item()
        value = int(path_accumulated[i])

        # If env row doesn't exist, initialize it
        if env_idx not in df.index:
            df.loc[env_idx] = []

        # Find next available column index for this env
        existing_row = df.loc[env_idx].dropna()
        next_col = f"run_{len(existing_row)}"

        # Insert the new value
        df.at[env_idx, next_col] = value

    # Save updated file
    df.sort_index(inplace=True)
    df.reset_index().rename(columns={"index": "env"}).to_csv(file_path, index=False)


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

    if globals.use_multi_maze:
        # Need to create globals.path_idx
        if globals.path_idx is None:
            path_idx_list = []
            for e_idx in env_ids:
                start_point = globals.get_list_entry_from_env(globals.maze_start_list, e_idx)
                if start_point < 0:
                    path_length = globals.get_list_entry_from_env(globals.maze_path_list, e_idx).shape[0]
                    path_idx_list.append(random.randint(0, path_length - 1))
                else:
                    path_idx_list.append(start_point)

            globals.path_idx = torch.tensor(path_idx_list, device=sphere.device, dtype=torch.int)
            globals.path_direction = torch.randint(0, 2, env_ids.shape, device=sphere.device, dtype=torch.int) * 2 - 1
            globals.path_accumulated = torch.zeros_like(globals.path_direction)

        # Need to reset globals.path_idx in env_ids
        for e_idx in env_ids:
            start_point = globals.get_list_entry_from_env(globals.maze_start_list, e_idx)
            if start_point < 0:
                path_length = globals.get_list_entry_from_env(globals.maze_path_list, e_idx).shape[0]
                globals.path_idx[e_idx] = random.randint(0, path_length - 1)
            else:
                globals.path_idx[e_idx] = start_point

        path_direction_temp = torch.randint(0, 2, globals.path_direction.shape, device=sphere.device) * 2 - 1
        path_direction_temp = path_direction_temp.to(torch.int)
    else:
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

            globals.path_direction = torch.randint(0, 2, env_ids.shape, device=sphere.device, dtype=torch.int) * 2 - 1
            globals.path_accumulated = torch.zeros_like(globals.path_direction)

        if globals.maze_start_point < 0:
            globals.path_idx[env_ids] = math_utils.sample_uniform(
                0, path_length, len(env_ids), device=sphere.device
            ).to(torch.int)
        else:
            globals.path_idx[env_ids] = globals.maze_start_point * torch.ones(
                len(env_ids), device=sphere.device, dtype=torch.int
            )

        path_direction_temp = torch.randint(0, 2, globals.path_direction.shape, device=sphere.device) * 2 - 1
        path_direction_temp = path_direction_temp.to(torch.int)

    globals.path_direction[env_ids] = path_direction_temp[env_ids]
    globals.path_accumulated[env_ids] = 0
    env.hole_crossed[env_ids] = False
    env.path_before_hole[env_ids] = 0
    env.hole_crossings[env_ids] = 0

    sphere_pos = sphere.data.default_root_state[env_ids, :7].clone()
    target1_pos = sphere_pos.clone()
    target2_pos = target1_pos.clone()
    target3_pos = target1_pos.clone()

    if globals.use_multi_maze:
        # the asset configs come in a list of resetted envs, not all of them
        for idx, e_idx in enumerate(env_ids):
            maze_path = globals.get_list_entry_from_env(globals.maze_path_list, e_idx)
            path_length = maze_path.shape[0]
            sphere_pos[idx, :2] = maze_path[globals.path_idx[e_idx], :] + env.scene.env_origins[e_idx, :2]
            _step_path_savely(e_idx, path_length)
            target1_pos[idx, :2] = maze_path[globals.path_idx[e_idx], :] + env.scene.env_origins[e_idx, :2]
            _step_path_savely(e_idx, path_length)
            target2_pos[idx, :2] = maze_path[globals.path_idx[e_idx], :] + env.scene.env_origins[e_idx, :2]
            _step_path_savely(e_idx, path_length)
            target3_pos[idx, :2] = maze_path[globals.path_idx[e_idx], :] + env.scene.env_origins[e_idx, :2]
    else:
        path_length = globals.maze_path.shape[0]
        sphere_pos[:, :2] = globals.maze_path[globals.path_idx[env_ids], :] + env.scene.env_origins[env_ids, :2]
        _step_path_savely(env_ids, path_length)
        target1_pos[:, :2] = globals.maze_path[globals.path_idx[env_ids], :] + env.scene.env_origins[env_ids, :2]
        _step_path_savely(env_ids, path_length)
        target2_pos[:, :2] = globals.maze_path[globals.path_idx[env_ids], :] + env.scene.env_origins[env_ids, :2]
        _step_path_savely(env_ids, path_length)
        target3_pos[:, :2] = globals.maze_path[globals.path_idx[env_ids], :] + env.scene.env_origins[env_ids, :2]

    sphere.write_root_pose_to_sim(sphere_pos, env_ids=env_ids)
    sphere.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=sphere.device), env_ids=env_ids)
    target1.write_root_pose_to_sim(target1_pos, env_ids=env_ids)
    target3.write_root_pose_to_sim(target3_pos, env_ids=env_ids)
    target2.write_root_pose_to_sim(target2_pos, env_ids=env_ids)


def _step_path_savely(env_ids, path_length):
    # Get current index and direction for the selected envs
    idx = globals.path_idx[env_ids]
    direction = globals.path_direction[env_ids]

    # Reverse direction if out of bounds
    out_of_bounds_high = (idx + direction) >= path_length
    out_of_bounds_low = (idx + direction) <= 0

    # Flip direction only where needed
    direction[out_of_bounds_high | out_of_bounds_low] *= -1

    # Update global direction and index
    globals.path_direction[env_ids] = direction
    globals.path_idx[env_ids] = idx + direction


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
    return xy_reward_l2 / 0.06


# TODO figure out which one makes sense and implement
# def shaping_reward():
#     # Distance based reward
#     reward += prev_dist_to_target - curr_dist_to_target

#     # Path-Following Reward (Vector Alignment)
#     direction_to_target = normalize(next_target - sphere_pos)
#     sphere_velocity = sphere_pos - prev_sphere_pos

#     alignment = np.dot(normalize(sphere_velocity), direction_to_target)
#     reward += k_align * alignment  # ranges from -1 to 1

#     # Waypoint based progress
#     # Penalize distance from the ideal path (e.g., line between waypoints)
#     reward -= k_deviation * perpendicular_distance_to_path

#     # Scaling shaping reward?
#     shaping_scale = initial_scale * exp(-decay_rate * episode_number)


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
