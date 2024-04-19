# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass

import orbit.maze.tasks.maze.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.maze import MAZE_CFG  # isort:skip


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
            mass_props=sim_utils.MassPropertiesCfg(density=7850), # Density of steel in kg/m^3)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), metallic=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )
    # sphere_object = RigidObject(cfg=sphere_cfg)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
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

    outer_joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["OuterDOF_RevoluteJoint"], scale=.1)
    inner_joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["InnerDOF_RevoluteJoint"], scale=.1)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        sphere_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("sphere")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

            # self.sphere_pos.func = self.get_sphere_pos

        # def get_sphere_pos(self, env: Any, asset_cfg: SceneEntityCfg = SceneEntityCfg("sphere")) -> torch.Tensor:
        #     """Wrapper method to get sphere position."""
        #     return mdp.root_pos_w(env, asset_cfg)

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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    inner_joint_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["InnerDOF_RevoluteJoint"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    outer_joint_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint"]), "target": 0.0},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["OuterDOF_RevoluteJoint"]), "bounds": (-3.0, 3.0)},
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
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (1.5, 0.0, 1.0)
        # simulation settings
        self.sim.dt = 1 / 120