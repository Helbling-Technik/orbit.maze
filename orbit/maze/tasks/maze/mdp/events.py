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


def randomize_actuator_stiffness_and_damping(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_range: tuple[float, float] | None = None,
    damping_range: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform"] = "uniform",
):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given ranges and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the ranges are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.

    Raises:
        NotImplementedError: If the joint indices are in explicit motor mode. This operation is currently
            not supported for explicit actuator models.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve joint indices
    joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device).item()

    # sample joint properties from the given ranges and set into the physics simulation
    # -- stiffness
    if stiffness_range is not None:
        stiffness = asset.root_physx_view.get_dof_stiffnesses().to(asset.device)
        stiffness = _randomize_values(
            stiffness,
            stiffness_range,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_stiffness_to_sim(stiffness[env_ids, joint_ids], joint_ids=joint_ids, env_ids=env_ids)

    # -- damping
    if damping_range is not None:
        damping = asset.root_physx_view.get_dof_dampings().to(asset.device)
        damping = _randomize_values(
            damping,
            damping_range,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_damping_to_sim(damping[env_ids, joint_ids], joint_ids=joint_ids, env_ids=env_ids)


def randomize_joint_friction_and_armature(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float] | None = None,
    armature_range: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform"] = "uniform",
):
    """Randomize the joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters (friction and armature) of the asset. These correspond
    to the physics engine joint properties that affect the joint behavior.

    The function samples random values from the given ranges and applies the operation to the joint properties.
    It then sets the values into the physics simulation. If the ranges are not provided for a
    particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device).item()

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if friction_range is not None:
        friction = asset.root_physx_view.get_dof_friction_coefficients().to(asset.device)
        friction = _randomize_values(
            friction,
            friction_range,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_friction_to_sim(friction[env_ids, joint_ids], joint_ids=joint_ids, env_ids=env_ids)
    # -- armature
    if armature_range is not None:
        armature = asset.root_physx_view.get_dof_armatures().to(asset.device)
        armature = _randomize_values(
            armature,
            armature_range,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_armature_to_sim(armature[env_ids, joint_ids], joint_ids=joint_ids, env_ids=env_ids)


def _randomize_values(
    data: torch.Tensor,
    sample_range: tuple[float, float],
    env_ids: torch.Tensor | None,
    joint_ids: int,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        sample_range: The range to sample the random values from.
        env_ids: The indices of the first dimension to randomize.
        joint_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if env_ids is None:
        n_dim_0 = data.shape[0]
        env_ids = slice(None)
    else:
        n_dim_0 = len(env_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform' or 'log_uniform'."
        )

    # perform the operation
    if operation == "add":
        data[env_ids, joint_ids] += dist_fn(*sample_range, (n_dim_0), device=data.device)
    elif operation == "scale":
        data[env_ids, joint_ids] *= dist_fn(*sample_range, (n_dim_0), device=data.device)
    elif operation == "abs":
        data[env_ids, joint_ids] = dist_fn(*sample_range, (n_dim_0), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


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
