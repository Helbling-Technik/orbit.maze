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
    # extract the used quantities (to enable type-hinting)
    # asset: Articulation = env.scene[asset_cfg.name]
    asset: Camera = env.scene[asset_cfg.name]
    # Assuming asset and its data are properly defined and initialized
    n_envs = asset.data.output["rgb"].size(0)
    n = int(asset.data.output["rgb"].numel() / n_envs)
    # print("Size of the tensor asset.data.output['rgb']:", tensor.size())
    return asset.data.output["rgb"].view(n_envs, n)
    # return asset.data.output["rgb"]


def get_target_pos(env: RLTaskEnv, target: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    # asset: RigidObject = env.scene[asset_cfg.name]
    # target_tensor = torch.tensor([target.get(key, 0.0) for key in ["x", "y"]], device=asset.data.root_pos_w.device)
    # root_pos = asset.data.root_pos_w - env.scene.env_origins
    zeros_tensor = torch.zeros_like(env.scene.env_origins)
    # return (zeros_tensor - root_pos)[:, :2].to(dtype=torch.float16)
    return zeros_tensor[:, :2]
