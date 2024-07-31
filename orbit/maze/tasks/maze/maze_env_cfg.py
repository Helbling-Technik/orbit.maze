# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import yaml

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.sensors import CameraCfg, RayCasterCfg
from omni.isaac.lab.sensors.ray_caster.patterns import BpearlPatternCfg, GridPatternCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass
from globals import path_idx, maze_path

import orbit.maze.tasks.maze.mdp as mdp
import os

##
# Pre-defined configs
##
# from omni.isaac.lab_assets.maze import MAZE_CFG  # isort:skip
# from maze import MAZE_CFG  # isort:skip

# Absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Absolute path of the project root (assuming it's three levels up from the current script)
project_root = os.path.join(current_script_path, "../../../../..")

MAZE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=os.path.join(project_root, "usds/maze01/maze02.usd"),
        usd_path=os.path.join(project_root, "usds/generated_mazes/generated_maze_02.usd"),
        # usd_path=os.path.join(project_root, "urdfs/generated_maze_03/generated_maze_03.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        # physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.2),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={"OuterDOF_RevoluteJoint": 0.0, "InnerDOF_RevoluteJoint": 0.0}
    ),
    # Position Control: For position controlled joints, set a high stiffness and relatively low or zero damping.
    # Velocity Control: For velocity controller joints, set a high damping and zero stiffness.
    actuators={
        "outer_actuator": ImplicitActuatorCfg(
            joint_names_expr=["OuterDOF_RevoluteJoint"],
            effort_limit=10,  # 5g * 9.81 * 0.15m = 0.007357
            velocity_limit=20 * math.pi,
            stiffness=0.0,
            damping=10.0,
        ),
        "inner_actuator": ImplicitActuatorCfg(
            joint_names_expr=["InnerDOF_RevoluteJoint"],
            effort_limit=10,  # 5g * 9.81 * 0.15m = 0.007357
            velocity_limit=20 * math.pi,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

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

    # maze
    robot: ArticulationCfg = MAZE_CFG.replace(prim_path="{ENV_REGEX_NS}/Labyrinth")

    # Sphere with collision enabled but not actuated
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.00625,
            mass_props=sim_utils.MassPropertiesCfg(density=7850),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.2),
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(maze_path[0, 0], maze_path[0, 1], 0.12)),
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
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(maze_path[0, 0], maze_path[0, 1], 0.105)),
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
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(maze_path[1, 0], maze_path[1, 1], 0.105)),
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
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(maze_path[2, 0], maze_path[2, 1], 0.105)),
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

    # outer_joint_effort = mdp.JointPositionActionCfg(
    #     asset_name="robot", joint_names=["OuterDOF_RevoluteJoint"], scale=15 * math.pi / 180 / 10
    # )
    # inner_joint_effort = mdp.JointPositionActionCfg(
    #     asset_name="robot", joint_names=["InnerDOF_RevoluteJoint"], scale=15 * math.pi / 180 / 10
    # )
    outer_joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["OuterDOF_RevoluteJoint"], scale=1.0)
    inner_joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["InnerDOF_RevoluteJoint"], scale=1.0)


velocity_extractor = mdp.VelocityExtractor()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_with_noise,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.001},
        )
        # joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_est_vel = ObsTerm(
            func=velocity_extractor.extract_joint_velocity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        sphere_pos = ObsTerm(
            func=mdp.root_pos_w_with_noise,
            params={"asset_cfg": SceneEntityCfg("sphere"), "std": 0.002},
        )
        sphere_est_vel = ObsTerm(
            func=velocity_extractor.extract_root_velocity,
            params={"asset_cfg": SceneEntityCfg("sphere")},
        )
        target1_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target1"),
            },
        )
        target2_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target2"),
            },
        )
        target3_pos = ObsTerm(
            func=mdp.root_pos_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("target3"),
            },
        )

        image = ObsTerm(
            func=mdp.simulated_camera_image,
            params={
                "sphere_cfg": SceneEntityCfg("sphere"),
                "maze_cfg": SceneEntityCfg("robot"),
            },
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
    # reset_maze_path_idx = EventTerm(
    #     func=mdp.reset_maze_path_idx,
    #     mode="reset",
    #     params={"sphere_cfg": SceneEntityCfg("sphere")},
    # )

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

    # reset_sphere_pos = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("sphere"),
    #         "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
    #         "velocity_range": {},
    #     },
    # )

    # reset_target1_pos = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("target1"),
    #         "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
    #         "velocity_range": {},
    #     },
    # )
    # reset_target2_pos = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("target2"),
    #         "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
    #         "velocity_range": {},
    #     },
    # )
    # reset_target3_pos = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("target3"),
    #         "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
    #         "velocity_range": {},
    #     },
    # )

    randomize_outer_actuator = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="OuterDOF_RevoluteJoint"),
            "stiffness_distribution_params": (0.5, 2.0),
            "stiffness_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_inner_actuator = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="InnerDOF_RevoluteJoint"),
            "stiffness_distribution_params": (0.5, 2.0),
            "stiffness_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_outer_joint = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="OuterDOF_RevoluteJoint"),
            "friction_distribution_params": (0.05, 0.1),
            "operation": "abs",
            "distribution": "log_uniform",
        },
    )

    randomize_inner_joint = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="InnerDOF_RevoluteJoint"),
            "friction_distribution_params": (0.05, 0.1),
            "operation": "abs",
            "distribution": "log_uniform",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: control maze path
    sphere_maze_path_target = RewTerm(
        func=mdp.path_point_target,
        weight=1000.0,
        params={
            "target1_cfg": SceneEntityCfg("target1"),
            "target2_cfg": SceneEntityCfg("target2"),
            "target3_cfg": SceneEntityCfg("target3"),
            "sphere_cfg": SceneEntityCfg("sphere"),
            "distance_from_target": 0.02,
            "idx_max": 55,
        },
    )

    joint_action = RewTerm(
        func=mdp.action_l2,
        weight=-0.1,
    )

    joint_action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,
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
        self.decimation = 2
        self.episode_length_s = 5  # 20s enough to solve maze01
        # viewer settings
        self.viewer.eye = (1, 1, 1.5)
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = 10
