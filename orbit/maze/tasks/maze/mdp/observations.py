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
    return asset.data.output["rgb"]
    # return asset.data.output["rgb"]
