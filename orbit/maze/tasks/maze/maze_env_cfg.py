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
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.actuators import DelayedImplicitActuatorCfg  # SCK: used to be ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass

import globals

import orbit.maze.tasks.maze.mdp as mdp
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
                min_delay=10 if globals.use_delay else 0,  # timesteps
                max_delay=15 if globals.use_delay else 0,  # timesteps
                joint_names_expr=["OuterDOF_RevoluteJoint"],
                effort_limit=10,  # 5g * 9.81 * 0.15m = 0.007357
                velocity_limit=20 * math.pi,
                stiffness=1000.0 if globals.position_control else 0.0,
                damping=1.0 if globals.position_control else 10.0,
            ),
            "inner_actuator": DelayedImplicitActuatorCfg(
                min_delay=10 if globals.use_delay else 0,  # timesteps
                max_delay=15 if globals.use_delay else 0,  # timesteps
                joint_names_expr=["InnerDOF_RevoluteJoint"],
                effort_limit=10,  # 5g * 9.81 * 0.15m = 0.007357
                velocity_limit=20 * math.pi,
                stiffness=1000.0 if globals.position_control else 0.0,
                damping=1.0 if globals.position_control else 10.0,
            ),
        },
    )

    return maze_cfg


def get_maze_cfg():
    # Absolute path of the current script
    current_script_path = os.path.abspath(__file__)
    # Absolute path of the project root (assuming it's five levels up from the current script)
    project_root = os.path.join(current_script_path, "../../../../..")

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
            )
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
                # TODO ROV changed delay from 3-10 to 2-3
                min_delay=2 if globals.use_delay else 0,  # timesteps was 10
                max_delay=3 if globals.use_delay else 0,  # timesteps was 15
                joint_names_expr=["OuterDOF_RevoluteJoint"],
                effort_limit=0.2,  # 5g * 9.81 * 0.15m = 0.007357
                velocity_limit=600 * 2 * math.pi / 60,  # rad/s = RPM * 2pi/60
                stiffness=5.15 if globals.position_control else 0.0,
                damping=1.74 if globals.position_control else 10.0,
            ),
            "inner_actuator": DelayedImplicitActuatorCfg(
                # TODO ROV changed delay from 3-10 to 2-3
                min_delay=2 if globals.use_delay else 0,  # timesteps was 10
                max_delay=3 if globals.use_delay else 0,  # timesteps was 15
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.5, restitution=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.12)),
    )

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
    # if divided by 100 (action space) would have direct mapping to rad from (-axis limit, +axis-limit)
    # Adding clipping will not work well for PPO so commented out
    if globals.position_control:
        if False:  # TODO ROV define proper variable
            outer_joint_effort = mdp.actions.JointPositionPIDCfg(
                asset_name="robot",
                joint_names=["OuterDOF_RevoluteJoint"],
                scale=7 * math.pi / 180 / 10,
                p_gain=0.65,
                i_gain=-1.0,
                d_gain=0.13,
                frequency=globals.targeted_frequency,
                alpha=1.0,
                limit=7 * math.pi / 180,
                # clip={"OuterDOF_RevoluteJoint" : [-3 * math.pi / 180, 3 * math.pi / 180]}
            )
            inner_joint_effort = mdp.actions.JointPositionPIDCfg(
                asset_name="robot",
                joint_names=["InnerDOF_RevoluteJoint"],
                scale=10 * math.pi / 180 / 10,
                p_gain=0.65,
                i_gain=-1.0,
                d_gain=0.13,
                frequency=globals.targeted_frequency,
                alpha=1.0,
                limit=10 * math.pi / 180,
                # clip={"InnerDOF_RevoluteJoint" : [-4 * math.pi / 180, 4 * math.pi / 180]}
            )
        else:
            outer_joint_effort = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["OuterDOF_RevoluteJoint"],
                scale=7 * math.pi / 180 / 10,
                # clip={"OuterDOF_RevoluteJoint" : [-3 * math.pi / 180, 3 * math.pi / 180]}
            )
            inner_joint_effort = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["InnerDOF_RevoluteJoint"],
                scale=10 * math.pi / 180 / 10,
                # clip={"InnerDOF_RevoluteJoint" : [-4 * math.pi / 180, 4 * math.pi / 180]}
            )
    else:
        outer_joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["OuterDOF_RevoluteJoint"], scale=1.0
        )
        inner_joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["InnerDOF_RevoluteJoint"], scale=1.0
        )

# TODO ROV make this selectable
# velocity_extractor = mdp.VelocityExtractor()


# TODO ROV NORMALIZE ALL OBSERVATIONS
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class MlpPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_with_noise,
            history_length=6,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.01},
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[0.0, 0.0, 1.0, 0.0])] if globals.use_delay else None,
        )
        # TODO ROV make this selectable
        # joint_est_vel = ObsTerm(
        #     func=velocity_extractor.extract_joint_velocity,
        #     history_length=6,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )

        sphere_pos = ObsTerm(
            # TODO ROV train without pos z
            # func=mdp.root_pos_w_with_noise,
            func=mdp.root_pos_xy_w_with_noise,
            history_length=6,
            params={"asset_cfg": SceneEntityCfg("sphere"), "std": 0.01},
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[0.0, 0.0, 1.0, 0.0])] if globals.use_delay else None,
        )
        # TODO ROV make this selectable
        # sphere_est_vel = ObsTerm(
        #     func=velocity_extractor.extract_root_velocity,
        #     history_length=6,
        #     params={"asset_cfg": SceneEntityCfg("sphere")},
        # )
        target1_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target1"),
            },
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[0.0, 0.0, 1.0, 0.0])] if globals.use_delay else None,
        )
        target2_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target2"),
            },
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[0.0, 0.0, 1.0, 0.0])] if globals.use_delay else None,
        )
        target3_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target3"),
            },
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[0.0, 0.0, 1.0, 0.0])] if globals.use_delay else None,
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
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[0.0, 0.0, 1.0, 0.0])] if globals.use_delay else None,
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
    # TODO ROV event term to save max path, define variable
    if globals.record_path_score:
        record_max_path = EventTerm(
            func=mdp.store_accumulated_path_score,
            mode="reset",
            params={
                "file_path": "logs/sb3/Isaac-Maze-v0/test-scores/test_run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv",
            },
        )

    reset_maze_state = EventTerm(
        func=mdp.reset_maze_state,
        mode="reset",
        params={
            "target1_cfg": SceneEntityCfg("target1"),
            "target2_cfg": SceneEntityCfg("target2"),
            "target3_cfg": SceneEntityCfg("target3"),
            "sphere_cfg": SceneEntityCfg("sphere"),
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint", "InnerDOF_RevoluteJoint"]),
            "position_range": (-0.05 * math.pi, 0.05 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )

    # add friction randomization to material, trained ok.
    # should only be done on startup and not on reset, very CPU intense and has a limit of num_buckets
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["InnerDOF", "InnerDOFWalls"]),
            "static_friction_range": (0.1, 0.8),
            "dynamic_friction_range": (0.1, 0.8),
            "restitution_range": (0.25, 0.75),
            "num_buckets": 300,
        },
    )

    sphere_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere"),
            "static_friction_range": (0.1, 0.8),
            "dynamic_friction_range": (0.1, 0.8),
            "restitution_range": (0.25, 0.75),
            "num_buckets": 300,
        },
    )

    randomize_robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        }
    )

    randomize_sphere_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        }
    )

    randomize_outer_actuator = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint", "InnerDOF_RevoluteJoint"]),
            "stiffness_distribution_params": (0.9, 1.1) if globals.small_actuator_gains else (0.3, 1.7),
            "damping_distribution_params": (0.9, 1.1) if globals.small_actuator_gains else (0.3, 1.7),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_outer_joint = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint", "InnerDOF_RevoluteJoint"]),
            "friction_distribution_params": (0.0, 0.1) if globals.small_joint_friction else (0.0, 0.2),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    # radius of sphere 0.00625m, density 7850kg/m3 -> mass 0.008028kg
    # With force of 0.001N -> 0.12m/s2
    if globals.use_force:
        randomize_sphere_force = EventTerm(
            func=mdp.apply_global_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("sphere"),
                "force_range": [-0.0005, 0.0005],
                "torque_range": [-0, 0],
                "is_global_wrench": True,
            },
        )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.0001)

    # (2) Failure penalty
    terminating = RewTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("sphere"), "minimum_height": 0.01},
        # Making the reward sparse by multiplying with inv_dt since rewards are handled in continuous way * dt
        weight=-1.0 * globals.targeted_frequency,
    )
    # (3) Primary task: control maze path
    sphere_maze_path_target = RewTerm(
        func=mdp.path_point_target,
        # Making the reward sparse by multiplying with inv_dt since rewards are handled in continuous way * dt
        weight=1.0 * globals.targeted_frequency,
        params={
            "target1_cfg": SceneEntityCfg("target1"),
            "target2_cfg": SceneEntityCfg("target2"),
            "target3_cfg": SceneEntityCfg("target3"),
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


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


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

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 3  # we simulate observations at 30Hz => dt=1/90*3 = 30Hz
        self.episode_length_s = 15 if globals.real_maze or globals.use_multi_maze else 10
        # viewer settings
        self.viewer.eye = (1, 1, 1.5)
        # simulation settings
        self.sim.dt = 1 / (self.decimation * globals.targeted_frequency)
        self.sim.render_interval = 3

        # TODO CLEANUP set physics properties if warning, not high enough for 16384 envs
        # self.sim.physx.gpu_collision_stack_size = 2**29
