# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.sensors import CameraCfg
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.utils import configclass

import orbit.maze.tasks.maze.mdp as mdp
import os

##
# Pre-defined configs
##
# from omni.isaac.orbit_assets.maze import MAZE_CFG  # isort:skip
# from maze import MAZE_CFG  # isort:skip

# Absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Absolute path of the project root (assuming it's three levels up from the current script)
project_root = os.path.join(current_script_path, "../../../../..")

MAZE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        # Path to the USD file relative to the project root
        usd_path=os.path.join(project_root, "usds/Maze_Simple.usd"),
        # usd_path=f"../../../../usds/Maze_Simple.usd",
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={"OuterDOF_RevoluteJoint": 0.0, "InnerDOF_RevoluteJoint": 0.0}
    ),
    actuators={
        "outer_actuator": ImplicitActuatorCfg(
            joint_names_expr=["OuterDOF_RevoluteJoint"],
            effort_limit=0.01,
            velocity_limit=1.0 / math.pi,
            stiffness=0.0,
            damping=10.0,
        ),
        "inner_actuator": ImplicitActuatorCfg(
            joint_names_expr=["InnerDOF_RevoluteJoint"],
            effort_limit=0.01,
            velocity_limit=1.0 / math.pi,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)


##
# Scene definition
##


@configclass
class MazeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = MAZE_CFG.replace(prim_path="{ENV_REGEX_NS}/Labyrinth")

    # Sphere with collision enabled but not actuated
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.005,  # Define the radius of the sphere
            mass_props=sim_utils.MassPropertiesCfg(density=7850),  # Density of steel in kg/m^3)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), metallic=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.11)),
    )

    # # sensors
    top_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_cam",
        update_period=0.04,  # 25fps
        height=128,
        width=128,
        data_types=["rgb"],  # , "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.02, 0.4), rot=(0, 1, 0, 0), convention="ros"),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )
    # distant_light = AssetBaseCfg(
    #     prim_path="/World/DistantLight",
    #     spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    #     init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    # )


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

    outer_joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["OuterDOF_RevoluteJoint"], scale=0.1)
    inner_joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["InnerDOF_RevoluteJoint"], scale=0.1)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # sphere_pos = ObsTerm(
        #     func=mdp.root_pos_w,
        #     params={"asset_cfg": SceneEntityCfg("sphere")},
        # )
        # sphere_lin_vel = ObsTerm(
        #     func=mdp.root_lin_vel_w,
        #     params={"asset_cfg": SceneEntityCfg("sphere")},
        # )

        target_pos_rel = ObsTerm(
            func=mdp.get_target_pos,
            params={
                "asset_cfg": SceneEntityCfg("sphere"),
                "target": {"x": 0.0, "y": 0.0},
            },
        )

        maze_top_view = ObsTerm(
            func=mdp.camera_image,
            params={"asset_cfg": SceneEntityCfg("top_cam")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_outer_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint"]),
            "position_range": (-0.01 * math.pi, 0.01 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )

    reset_inner_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["InnerDOF_RevoluteJoint"]),
            "position_range": (-0.01 * math.pi, 0.01 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )

    reset_sphere_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sphere"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep sphere in center
    sphere_pos = RewTerm(
        func=mdp.root_xypos_target_l2,
        weight=-500.0,
        params={
            "asset_cfg": SceneEntityCfg("sphere"),
            "target": {"x": 0.0, "y": 0.0},
        },
    )
    # (4) Shaping tasks: lower outer joint velocity
    outer_joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint"])},
    )
    # (5) Shaping tasks: lower outer joint velocity
    inner_joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["InnerDOF_RevoluteJoint"])},
    )

    # # (4) Shaping tasks: lower outer joint velocity
    # outer_joint_lim = RewTerm(
    #     func=mdp.applied_torque_limits,
    #     weight=-10,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint"])},
    # )
    # # (5) Shaping tasks: lower outer joint velocity
    # inner_joint_lim = RewTerm(
    #     func=mdp.applied_torque_limits,
    #     weight=-10,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["InnerDOF_RevoluteJoint"])},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
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
class MazeEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
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
        self.decimation = 8
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (1, 1, 1.5)
        # simulation settings
        self.sim.dt = 1 / 200
