# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedEnv

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.stage as stage_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def set_random_target_pos(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    target_positions = env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    return target_positions


# TODO ROV here is already some randomization stuff
def randomize_usds(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

    print("asset_cfg: ", asset_cfg)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    print("asset: ", asset)

    maze_path = "/home/sck/git/orbit.maze/usds/generated_mazes/maze01.usd"
    print("maze_path: ", maze_path)

    spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=maze_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    )

    asset.cfg.spawn = spawn_cfg

    prim_path_template = asset.cfg.prim_path
    env_id = env_ids[0]
    env_id_str = str(env_id.item())
    prim_path = prim_path_template.replace(".*", env_id_str)
    print("prim_path: ", prim_path)

    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    print("prim: ", prim)
    print("prim.GetReferences(): ", prim.GetReferences())
    prim.GetReferences().ClearReferences()
    print("prim.GetReferences().ClearReferences: ", prim.GetReferences().ClearReferences())

    sim_utils.spawners.from_files.spawn_from_usd(
        prim_path, asset.cfg.spawn, translation=asset.cfg.init_state.pos, orientation=asset.cfg.init_state.rot
    )


def randomize_usd_paths(
    env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    maze_path = "/home/sck/git/orbit.maze/usds/generated_mazes/maze02.usd"

    prim_path_template = asset.cfg.prim_path
    stage = stage_utils.get_current_stage()

    for env_id in env_ids:
        env_id_str = str(env_id.item())
        prim_path = prim_path_template.replace(".*", env_id_str)

        print("prim_path: ", prim_path)
        prim = stage.GetPrimAtPath(prim_path)
        print("prim: ", prim)

        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(maze_path)
