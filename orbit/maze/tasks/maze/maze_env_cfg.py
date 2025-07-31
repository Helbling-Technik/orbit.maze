# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

# import torch
# import yaml

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.actuators import DelayedImplicitActuatorCfg  # SCK: used to be ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass

import globals

import orbit.maze.tasks.maze.mdp as mdp
import orbit.maze.tasks.maze.randomization as rdm
import os
from datetime import datetime

# Multimaze imports
from pxr import Usd
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
import carb
from omni.isaac.lab.sim.utils import bind_visual_material, select_usd_variants
from omni.isaac.lab.sim import schemas
import re
from dataclasses import MISSING


@configclass
class MultiMazeCfg(sim_utils.SpawnerCfg):
    """Configuration parameters for loading multiple mazes looping over a list"""

    maze_usd_cfgs: list[sim_utils.UsdFileCfg] = MISSING
    """List of mazes to spawn, usd configs."""
    current_script_path = os.path.abspath(__file__)
    # Absolute path of the project root (assuming it's 5 levels up from the current script)
    project_root = os.path.join(current_script_path, "../../../../..")


# This is from isaac lab directly, but not includable without modifying isaacs code
def _spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        usd_path: The path to the USD file to spawn the asset from.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # check file path exists
    stage: Usd.Stage = stage_utils.get_current_stage()
    if not stage.ResolveIdentifierToEditTarget(usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")
    # spawn asset if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        # add prim as reference to stage
        prim_utils.create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
    else:
        carb.log_warn(f"A prim already exists at prim path: '{prim_path}'.")

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # modify rigid body properties
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    # modify collision properties
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    # modify mass properties
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)

    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    # modify tendon properties
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    # define drive API on the joints
    # note: these are only for setting low-level simulation properties. all others should be set or are
    #  and overridden by the articulation/actuator properties.
    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)

    # modify deformable body properties
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


def spawn_multi_mazes(
    prim_path: str,
    cfg: MultiMazeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:

    # return the prim
    # return _spawn_from_usd_file(prim_path, usd_path, cfg, translation, orientation)
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # resolve prim paths for spawning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # spawn asset from the given usd file
    for idx, prim_path in enumerate(prim_paths):
        # sample the asset config to load
        usd_idx = idx % len(cfg.maze_usd_cfgs)
        usd_path = os.path.join(cfg.project_root, cfg.maze_usd_cfgs[usd_idx].usd_path)
        usd_cfg = cfg.maze_usd_cfgs[usd_idx]

        # load the asset
        prim = _spawn_from_usd_file(prim_path, usd_path, usd_cfg, translation, orientation)

    return prim


def get_multi_maze_cfg():
    # articulation
    usd_file_cfgs = []

    # TODO ROV these are timesteps, should be scaled by update frequency to correspond to actual time delay of 60-150ms
    if globals.delay_level < 0:
        min_delay = 0
        max_delay = 0
    elif globals.delay_level == 0:
        min_delay = 2
        max_delay = 3
    else:
        min_delay = 3
        max_delay = 10

    for usd in globals.usd_list:
        maze_usd_cfg = sim_utils.UsdFileCfg(
            usd_path=usd["location"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        )
        usd_file_cfgs.append(maze_usd_cfg)

    maze_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Labyrinth",
        spawn=MultiMazeCfg(maze_usd_cfgs=usd_file_cfgs, func=spawn_multi_mazes),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={"OuterDOF_RevoluteJoint": 0.0, "InnerDOF_RevoluteJoint": 0.0}
        ),
        # Position Control: For position controlled joints, set a high stiffness and relatively low or zero damping.
        # Velocity Control: For velocity controller joints, set a high damping and zero stiffness.
        actuators={
            "outer_actuator": DelayedImplicitActuatorCfg(
                min_delay=min_delay,  # timesteps
                max_delay=max_delay,  # timesteps
                joint_names_expr=["OuterDOF_RevoluteJoint"],
                effort_limit=0.2,  # 5g * 9.81 * 0.15m = 0.007357
                velocity_limit=600 * 2 * math.pi / 60,
                stiffness=5.15 if globals.position_control else 0.0,
                damping=1.74 if globals.position_control else 10.0,
            ),
            "inner_actuator": DelayedImplicitActuatorCfg(
                min_delay=min_delay,  # timesteps
                max_delay=max_delay,  # timesteps
                joint_names_expr=["InnerDOF_RevoluteJoint"],
                effort_limit=0.2,  # 5g * 9.81 * 0.15m = 0.007357
                velocity_limit=600 * 2 * math.pi / 60,
                stiffness=5.15 if globals.position_control else 0.0,
                damping=1.74 if globals.position_control else 10.0,
            ),
        },
    )

    return maze_cfg


def get_maze_cfg():
    # Absolute path of the current script
    current_script_path = os.path.abspath(__file__)
    # Absolute path of the project root (assuming it's five levels up from the current script)
    project_root = os.path.join(current_script_path, "../../../../..")

    # TODO ROV these are timesteps, should be scaled by update frequency to correspond to actual time delay of 60-150ms
    if globals.delay_level < 0:
        min_delay = 0
        max_delay = 0
    elif globals.delay_level == 0:
        min_delay = 2
        max_delay = 3
    else:
        min_delay = 3
        max_delay = 10

    maze_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(project_root, globals.usd_file_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                density=410,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={"OuterDOF_RevoluteJoint": 0.0, "InnerDOF_RevoluteJoint": 0.0}
        ),
        # Position Control: For position controlled joints, set a high stiffness and relatively low or zero damping.
        # Velocity Control: For velocity controller joints, set a high damping and zero stiffness.
        # Tuned using gain tuner from Isaac Sim, relatable values for stiffness and damping are (logarithmic):
        # Stiffness: 5.15 1/deg or 2.47 1/rad
        # Damping: 1.74 1/deg or 2 1/rad
        actuators={
            "outer_actuator": DelayedImplicitActuatorCfg(
                min_delay=min_delay,  # timesteps
                max_delay=max_delay,  # timesteps
                joint_names_expr=["OuterDOF_RevoluteJoint"],
                effort_limit=0.2,  # 5g * 9.81 * 0.15m = 0.007357
                velocity_limit=600 * 2 * math.pi / 60,  # rad/s = RPM * 2pi/60
                stiffness=5.15 if globals.position_control else 0.0,
                damping=1.74 if globals.position_control else 10.0,
            ),
            "inner_actuator": DelayedImplicitActuatorCfg(
                min_delay=min_delay,  # timesteps
                max_delay=max_delay,  # timesteps
                joint_names_expr=["InnerDOF_RevoluteJoint"],
                effort_limit=0.2,  # 5g * 9.81 * 0.15m = 0.007357
                velocity_limit=600 * 2 * math.pi / 60,  # rad/s = RPM * 2pi/60
                stiffness=5.15 if globals.position_control else 0.0,
                damping=1.74 if globals.position_control else 10.0,
            ),
        },
    )

    return maze_cfg


# Scene definition
##


@configclass
class MazeSceneCfg(InteractiveSceneCfg):
    """Configuration for a maze scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # maze
    if globals.use_multi_maze:
        robot: ArticulationCfg = get_multi_maze_cfg()
    else:
        robot: ArticulationCfg = get_maze_cfg().replace(prim_path="{ENV_REGEX_NS}/Labyrinth")

    # Sphere with collision enabled but not actuated
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.00625,
            mass_props=sim_utils.MassPropertiesCfg(density=7850),
            # TODO ROV adding more settings
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, solver_position_iteration_count=16, solver_velocity_iteration_count=4
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                torsional_patch_radius=0.003,
                min_torsional_patch_radius=0.001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.5, restitution=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.12)),
    )

    # TODO Think about a way to automate number of targets

    target1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target1",
        spawn=sim_utils.SphereCfg(
            radius=0.003,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.105)),
    )
    target2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target2",
        spawn=sim_utils.SphereCfg(
            radius=0.003,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.105)),
    )
    target3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target3",
        spawn=sim_utils.SphereCfg(
            radius=0.003,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.105)),
    )

    # target4 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/target4",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.003,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.105)),
    # )

    # target5 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/target5",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.003,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.105)),
    # )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # rolling_resistance_action = mdp.RollingResistanceActionCfg(
    #     asset_name="sphere",
    #     resistance_coef_range=[0.01, 0.015],  # This randomizes on reset in this range
    #     speed_threshold=5.0,
    #     stick_slip_angle_range_deg=[2, 4],  # This randomizes on reset in this range
    # )
    # stick_slip_action = mdp.StickSlipActionCfg(
    #     asset_name="sphere",
    #     stick_slip_angle_range_deg=[2, 2],  # This randomizes on reset in this range
    # )

    # set scaling to proper angle
    # action output from -1 to 1 direct mapping to rad from (-axis limit, +axis-limit)
    # Adding clipping will not work well for PPO so commented out
    if globals.position_control:
        if globals.use_pid:
            outer_joint_effort = mdp.actions.JointPositionPIDCfg(
                asset_name="robot",
                joint_names=["OuterDOF_RevoluteJoint"],
                scale=globals.joint_limits[0] * math.pi / 180,  # / 10,
                p_gain=0.25,
                i_gain=-1.0,
                d_gain=0.0,
                frequency=globals.targeted_frequency,
                alpha=1.0,
                limit=globals.joint_limits[0] * math.pi / 180,
                # clip={"OuterDOF_RevoluteJoint" : [-3 * math.pi / 180, 3 * math.pi / 180]}
            )
            inner_joint_effort = mdp.actions.JointPositionPIDCfg(
                asset_name="robot",
                joint_names=["InnerDOF_RevoluteJoint"],
                scale=globals.joint_limits[1] * math.pi / 180,  # / 10,
                p_gain=0.25,
                i_gain=-1.0,
                d_gain=0.0,
                frequency=globals.targeted_frequency,
                alpha=1.0,
                limit=globals.joint_limits[1] * math.pi / 180,
                # clip={"InnerDOF_RevoluteJoint" : [-4 * math.pi / 180, 4 * math.pi / 180]}
            )
        else:
            outer_joint_effort = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["OuterDOF_RevoluteJoint"],
                scale=globals.joint_limits[0] * math.pi / 180,  # / 10,
                # clip={"OuterDOF_RevoluteJoint" : [-3 * math.pi / 180, 3 * math.pi / 180]}
            )
            inner_joint_effort = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["InnerDOF_RevoluteJoint"],
                scale=globals.joint_limits[1] * math.pi / 180,  # / 10,
                # clip={"InnerDOF_RevoluteJoint" : [-4 * math.pi / 180, 4 * math.pi / 180]}
            )
    else:
        outer_joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["OuterDOF_RevoluteJoint"], scale=1.0
        )
        inner_joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["InnerDOF_RevoluteJoint"], scale=1.0
        )


if globals.velocity_obs:
    velocity_extractor = mdp.VelocityExtractor()


# TODO ROV these are timesteps, should be scaled by update frequency to correspond to actual time delay of 50ms
def get_delay_modifiers(delay_level, synced_obs_delay):
    if delay_level == -1:
        return None
    elif delay_level == 0:
        return [mdp.RandomDelayCfg(A=[0.0], B=[0.0, 0.0, 1.0, 0.0])]
    elif delay_level == 1:
        return [mdp.RandomDelayCfg(A=[0.0], B=[0.0, 0.0, 0.0, 1.0, 0.0])]
    else:
        raise ValueError(f"Unknown delay_level: {delay_level}")


def get_joint_friction_distributions(joint_friction_level):
    if joint_friction_level == -1:
        return (0.0, 0.0)
    elif joint_friction_level == 0:
        return (0.0, 0.1)
    elif joint_friction_level == 1:
        return (0.0, 0.2)
    else:
        raise ValueError(f"Unknown joint_friction_level: {joint_friction_level}")


def get_actuator_gain_distributions(
    actuator_gain_level,
):
    if actuator_gain_level == -1:
        return (1.0, 1.0)
    elif actuator_gain_level == 0:
        return (0.9, 1.1)
    elif actuator_gain_level == 1:
        return (0.3, 1.7)
    else:
        raise ValueError(f"Unknown actuator_gain_level: {actuator_gain_level}")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class MlpPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.maze_joint_pos,
            history_length=6,
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[1.0, 0.0, 0.0, 0.0, 0.0])],
        )
        # TODO ROV normalize if needed again
        if globals.velocity_obs:
            joint_est_vel = ObsTerm(
                func=velocity_extractor.extract_joint_velocity,
                history_length=6,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
        sphere_pos = ObsTerm(
            func=mdp.sphere_pos,
            history_length=6,
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[1.0, 0.0, 0.0, 0.0, 0.0])],
        )
        # TODO ROV experimenting with state augmentation, unclear if I should have them delayed as well
        past_actions = ObsTerm(
            func=mdp.last_action,
            history_length=6,
            # TODO ROV currently training with this delay
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[1.0, 0.0, 0.0, 0.0, 0.0])],
        )

        # TODO ROV normalize if needed again
        if globals.velocity_obs:
            sphere_est_vel = ObsTerm(
                func=velocity_extractor.extract_root_velocity,
                history_length=6,
                params={"asset_cfg": SceneEntityCfg("sphere")},
            )

        # TODO DRP Refactor and handle targets with single more parametrizable ObsTerm
        target1_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target1"),
            },
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[1.0, 0.0, 0.0, 0.0, 0.0])],
        )
        target2_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target2"),
            },
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[1.0, 0.0, 0.0, 0.0, 0.0])],
        )
        target3_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target3"),
            },
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[1.0, 0.0, 0.0, 0.0, 0.0])],
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CnnPolicyCfg(ObsGroup):
        """Observations for policy group."""

        image = ObsTerm(
            func=mdp.simulated_camera_image,
            params={
                "sphere_cfg": SceneEntityCfg("sphere"),
                "maze_cfg": SceneEntityCfg("robot"),
            },
            # TODO ROV adding history to image? Wrong dimensions if we unsqueeze simulated_camera_image
            history_length=6,
            flatten_history_dim=False,
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[1.0, 0.0, 0.0, 0.0, 0.0])],
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    mlp_policy: MlpPolicyCfg = MlpPolicyCfg()
    cnn_policy: CnnPolicyCfg = CnnPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    if globals.record_path_score:
        record_max_path = EventTerm(
            func=mdp.store_accumulated_path_score,
            mode="reset",
            params={
                "file_path": "logs/sb3/Isaac-Maze-v0/test-scores/test_run_"
                + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + ".csv",
            },
        )

    reset_maze_state = EventTerm(
        func=mdp.reset_maze_state,
        mode="reset",
        params={
            "target1_cfg": SceneEntityCfg("target1"),
            "target2_cfg": SceneEntityCfg("target2"),
            "target3_cfg": SceneEntityCfg("target3"),
            # "target4_cfg": SceneEntityCfg("target4"),
            # "target5_cfg": SceneEntityCfg("target5"),
            "sphere_cfg": SceneEntityCfg("sphere"),
        },
    )

    reset_joints = EventTerm(
        func=mdp.adr_reset_maze_joints,
        mode="reset",
    )

    # add friction randomization to material
    robot_physics_material = EventTerm(
        func=mdp.adr_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["InnerDOF", "InnerDOFWalls"]),
            "static_friction_range": (0.1, 0.8),
            "dynamic_friction_range": (0.1, 0.8),
            "restitution_range": (0.05, 0.25),
            "num_buckets": 300,
        },
    )

    sphere_physics_material = EventTerm(
        func=mdp.adr_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere"),
            "static_friction_range": (0.1, 0.8),
            "dynamic_friction_range": (0.1, 0.8),
            "restitution_range": (0.05, 0.25),
            "num_buckets": 300,
        },
    )

    randomize_robot_mass = EventTerm(
        func=mdp.adr_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_sphere_mass = EventTerm(
        func=mdp.adr_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere"),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_outer_actuator = EventTerm(
        func=mdp.randomize_maze_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint", "InnerDOF_RevoluteJoint"]),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_outer_joint = EventTerm(
        func=mdp.randomize_joint_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint", "InnerDOF_RevoluteJoint"]),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    # radius of sphere 0.00625m, density 7850kg/m3 -> mass 0.008028kg
    # With force of 0.001N -> 0.12m/s2
    randomize_sphere_force = EventTerm(
        func=mdp.adr_external_force_and_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere"),
            "is_global_wrench": True,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.0001)

    # (2) Failure penalty
    on_hole = RewTerm(
        func=mdp.termination_reward,
        params={"term_name": "sphere_on_hole"},
        weight=-1.0 * globals.targeted_frequency,
    )

    # (3) Primary task: control maze path
    waypoint = RewTerm(
        func=mdp.waypoint_reward,
        weight=1.0 * globals.targeted_frequency,
        params={
            "waypoint_cfgs": [
                SceneEntityCfg("target1"),
                SceneEntityCfg("target2"),
                SceneEntityCfg("target3"),
            ],
            "sphere_cfg": SceneEntityCfg("sphere"),
        },
    )

    # smoother with increased penalty here
    joint_action = RewTerm(
        func=mdp.action_l2,
        weight=-0.001,
    )
    # smoother with increased penalty here
    # TODO ROV maybe increase rate penalty?
    joint_action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Sphere off maze
    sphere_on_ground = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("sphere"), "minimum_height": 0.01},
    )

    sphere_on_hole = DoneTerm(
        func=mdp.on_hole,
        params={"sphere_cfg": SceneEntityCfg("sphere"), "maze_cfg": SceneEntityCfg("robot"), "hole_radius": 0.0075},
    )


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    update_domain_randomization = CurrTerm(func=mdp.update_domain_randomization)

    # penalty_on_hole = CurrTerm(
    #     func=mdp.modify_reward_path,
    #     params={
    #         "term_name": "on_hole",
    #         "thresholds_and_weights": [
    #             (10, -4.0 * globals.targeted_frequency),
    #             # (40, -16.0 * globals.targeted_frequency),
    #             # (53, -32.0 * globals.targeted_frequency),
    #         ],  # Make sure thresholds are in increasing order
    #     },
    # )

    # penalty_on_hole_steps = CurrTerm(
    #     func=mdp.modify_reward_steps,
    #     params={
    #         "term_name": "on_hole",
    #         "steps_and_weights": [
    #             (3000, -4.0 * globals.targeted_frequency),
    #             (4500, -8.0 * globals.targeted_frequency),
    #             (6000, -16.0 * globals.targeted_frequency),
    #         ],  # Make sure steps are in increasing order
    #     },
    # )


##
# Domain Randomization configuration
##


class DomainRandomizationCfg:
    evaluation_probability: float = 0.5
    buffer_size: int = 100
    performance_threshold_lower: float = 8
    performance_threshold_upper: float = 12

    RANDOMIZABLE_PARAMETERS = [
        rdm.RandomizationParameter(
            name="inner_joint_pos_std",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-0.01,  # Absolute value is taken at observation, negative to comply with update system
                max_value=0.00,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=0.01,
            ),
            delta=0.0005,
        ),
        rdm.RandomizationParameter(
            name="outer_joint_pos_std",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-0.01,
                max_value=0.00,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=0.01,
            ),
            delta=0.0005,
        ),
        rdm.RandomizationParameter(
            name="sphere_x_std",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-0.01,  # Absolute value is taken at observation, negative to comply with update system
                max_value=0.00,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=0.01,
            ),
            delta=0.0005,
        ),
        rdm.RandomizationParameter(
            name="sphere_y_std",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-0.01,
                max_value=0.00,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=0.01,
            ),
            delta=0.0005,
        ),
        rdm.RandomizationParameter(
            name="joint_friction",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-0.3,
                max_value=0.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=0.3,
            ),
            delta=0.015,
        ),
        rdm.RandomizationParameter(
            name="stiffness",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=1.0,
                min_value=0.3,
                max_value=1.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=1.0,
                min_value=1.0,
                max_value=1.7,
            ),
            delta=0.035,
        ),
        rdm.RandomizationParameter(
            name="damping",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=1.0,
                min_value=0.3,
                max_value=1.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=1.0,
                min_value=1.0,
                max_value=1.7,
            ),
            delta=0.035,
        ),
        rdm.RandomizationParameter(
            name="obs_delay_mean",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-2.0,
                max_value=0.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=2.0,
            ),
            delta=0.05,
        ),
        rdm.RandomizationParameter(
            name="obs_delay_std",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-1.0,
                max_value=0.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=1.0,
            ),
            delta=0.05,
        ),
        rdm.RandomizationParameter(
            name="sphere_static_friction",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.5,
                min_value=0.0,
                max_value=0.5,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.5,
                min_value=0.5,
                max_value=1.0,
            ),
            delta=0.025,
        ),
        rdm.RandomizationParameter(
            name="sphere_dynamic_friction",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.5,
                min_value=0.0,
                max_value=0.5,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.5,
                min_value=0.5,
                max_value=1.0,
            ),
            delta=0.025,
        ),
        rdm.RandomizationParameter(
            name="sphere_restitution",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.2,
                min_value=0.0,
                max_value=0.2,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.2,
                min_value=0.2,
                max_value=0.4,
            ),
            delta=0.01,
        ),
        rdm.RandomizationParameter(
            name="robot_static_friction",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.5,
                min_value=0.0,
                max_value=0.5,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.5,
                min_value=0.5,
                max_value=1.0,
            ),
            delta=0.025,
        ),
        rdm.RandomizationParameter(
            name="robot_dynamic_friction",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.5,
                min_value=0.0,
                max_value=0.5,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.5,
                min_value=0.5,
                max_value=1.0,
            ),
            delta=0.025,
        ),
        rdm.RandomizationParameter(
            name="robot_restitution",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.2,
                min_value=0.0,
                max_value=0.2,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.2,
                min_value=0.2,
                max_value=0.4,
            ),
            delta=0.01,
        ),
        rdm.RandomizationParameter(
            name="robot_mass_distribution",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=1.0,
                min_value=0.5,
                max_value=1.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=1.0,
                min_value=1.0,
                max_value=1.5,
            ),
            delta=0.025,
        ),
        rdm.RandomizationParameter(
            name="sphere_mass_distribution",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=1.0,
                min_value=0.5,
                max_value=1.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=1.0,
                min_value=1.0,
                max_value=1.5,
            ),
            delta=0.025,
        ),
        rdm.RandomizationParameter(
            name="sphere_external_force",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-0.002,
                max_value=0.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=0.002,
            ),
            delta=0.0001,
        ),
        rdm.RandomizationParameter(
            name="start_inner_joint_pos",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=-0.05 * math.pi,
                min_value=-0.1 * math.pi,
                max_value=0.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.05 * math.pi,
                min_value=0.0,
                max_value=0.1 * math.pi,
            ),
            delta=0.005 * math.pi,
        ),
        rdm.RandomizationParameter(
            name="start_outer_joint_pos",
            lower_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.LOWER_BOUND,
                value=-0.05 * math.pi,
                min_value=-0.1 * math.pi,
                max_value=0.0,
            ),
            upper_bound=rdm.RandomizationBound(
                type=rdm.RandomizationBoundType.UPPER_BOUND,
                value=0.05 * math.pi,
                min_value=0.0,
                max_value=0.1 * math.pi,
            ),
            delta=0.005 * math.pi,
        ),
    ]


class EvalCfg:

    SET_PARAMS = {
        "inner_joint_pos_std": [-0.001, 0.001],  # [-0.001, 0.001]
        "outer_joint_pos_std": [-0.001, 0.001],  # [-0.001, 0.001]
        "sphere_x_std": [-0.004, 0.004],  # [-0.01, 0.01]
        "sphere_y_std": [-0.004, 0.004],  # [-0.01, 0.01]
        "joint_friction": [0.0, 0.2],  # [-0.3, 0.3]
        "stiffness": [0.3, 1.7],  # [0.3, 1.7]
        "damping": [0.3, 1.7],  # [0.3, 1.7]
        "obs_delay_mean": [-1.0, 1.0],  # [-2.0, 2.0]
        "obs_delay_std": [-0.6, 0.6],  # [-1.0, 1.0]
        "sphere_static_friction": [0.5, 0.5],
        "sphere_dynamic_friction": [0.5, 0.5],
        "sphere_restitution": [0.2, 0.2],
        "robot_static_friction": [0.5, 0.5],
        "robot_dynamic_friction": [0.5, 0.5],
        "robot_restitution": [0.2, 0.2],
        "sphere_mass_distribution": [1.0, 1.0],
        "robot_mass_distribution": [1.0, 1.0],
        "sphere_external_force": [0.0, 0.0],
        "start_inner_joint_pos": [0.0, 0.0],  # [-0.05 * math.pi, 0.05 * math.pi]
        "start_outer_joint_pos": [0.0, 0.0],  # [-0.05 * math.pi, 0.05 * math.pi]
    }


##
# Environment configuration
##


@configclass
class MazeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the maze learning environment."""

    # Scene settings
    if globals.use_multi_maze:
        scene: MazeSceneCfg = MazeSceneCfg(num_envs=16, env_spacing=0.5, replicate_physics=False)
    else:
        scene: MazeSceneCfg = MazeSceneCfg(num_envs=16, env_spacing=0.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    if globals.set_params:
        domain_randomization.evaluation_probability = 0.0
        eval_cfg: EvalCfg = EvalCfg()
        for param in domain_randomization.RANDOMIZABLE_PARAMETERS:
            param.lower_bound.value = eval_cfg.SET_PARAMS[param.name][0]
            param.upper_bound.value = eval_cfg.SET_PARAMS[param.name][1]
            print(f"Param {param.name} : [{param.lower_bound.value}, {param.upper_bound.value}]")

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 3  # we simulate observations at 30Hz => dt=1/90*3 = 30Hz
        self.episode_length_s = 30 if globals.real_maze or globals.use_multi_maze else 10
        # viewer settings
        self.viewer.eye = (1, 1, 1.5)
        # simulation settings
        self.sim.dt = 1 / (self.decimation * globals.targeted_frequency)
        self.sim.render_interval = 3

        # TODO CLEANUP set physics properties if warning, not high enough for 16384 envs
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.sim.physx.bounce_threshold_velocity = 0.02
        # self.sim.physx.friction_offset_threshold = 0.01
