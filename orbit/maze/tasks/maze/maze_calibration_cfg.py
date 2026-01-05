# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.actuators import DelayedImplicitActuatorCfg
from omni.isaac.lab.utils import configclass

import globals

import orbit.maze.tasks.maze.mdp as mdp
import os


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
                stiffness=20.0 if globals.position_control else 0.0,
                damping=2.0 if globals.position_control else 10.0,
            ),
            "inner_actuator": DelayedImplicitActuatorCfg(
                min_delay=min_delay,  # timesteps
                max_delay=max_delay,  # timesteps
                joint_names_expr=["InnerDOF_RevoluteJoint"],
                effort_limit=0.2,  # 5g * 9.81 * 0.15m = 0.007357
                velocity_limit=600 * 2 * math.pi / 60,  # rad/s = RPM * 2pi/60
                stiffness=20.0 if globals.position_control else 0.0,
                damping=2.0 if globals.position_control else 10.0,
            ),
        },
    )

    return maze_cfg


# Scene definition
##


@configclass
class CalibrationSceneCfg(InteractiveSceneCfg):
    """Configuration for a maze scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # maze
    robot: ArticulationCfg = get_maze_cfg().replace(prim_path="{ENV_REGEX_NS}/Labyrinth")

    # Sphere with collision enabled but not actuated
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.00625,
            mass_props=sim_utils.MassPropertiesCfg(density=7850),
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

    # set scaling to proper angle
    # action output from -1 to 1 direct mapping to rad from (-axis limit, +axis-limit)
    # Adding clipping will not work well for PPO so commented out
    if globals.position_control:
        if globals.use_pid:
            outer_joint_effort = mdp.actions.JointPositionPIDCfg(
                asset_name="robot",
                joint_names=["OuterDOF_RevoluteJoint"],
                scale=globals.joint_limits[0] * math.pi / 180,
                p_gain=0.25,
                i_gain=-1.0,
                d_gain=0.0,
                frequency=globals.targeted_frequency,
                alpha=1.0,
                limit=globals.joint_limits[0] * math.pi / 180,
            )
            inner_joint_effort = mdp.actions.JointPositionPIDCfg(
                asset_name="robot",
                joint_names=["InnerDOF_RevoluteJoint"],
                scale=globals.joint_limits[1] * math.pi / 180,
                p_gain=0.25,
                i_gain=-1.0,
                d_gain=0.0,
                frequency=globals.targeted_frequency,
                alpha=1.0,
                limit=globals.joint_limits[1] * math.pi / 180,
            )
        else:
            outer_joint_effort = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["OuterDOF_RevoluteJoint"],
                scale=globals.joint_limits[0] * math.pi / 180,
            )
            inner_joint_effort = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["InnerDOF_RevoluteJoint"],
                scale=globals.joint_limits[1] * math.pi / 180,
            )
    else:
        outer_joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["OuterDOF_RevoluteJoint"], scale=1.0
        )
        inner_joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["InnerDOF_RevoluteJoint"], scale=1.0
        )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class MlpPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rad,
            modifiers=[mdp.RandomDelayCfg(A=[0.0], B=[1.0, 0.0, 0.0, 0.0, 0.0])],
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    mlp_policy: MlpPolicyCfg = MlpPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass


##
# Environment configuration
##


@configclass
class CalibrationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the maze learning environment."""

    # Scene settings
    scene: CalibrationSceneCfg = CalibrationSceneCfg(num_envs=1, env_spacing=0.5)
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
        self.episode_length_s = 60
        # viewer settings
        self.viewer.eye = (1, 1, 1.5)
        # simulation settings
        self.sim.dt = 1 / (self.decimation * globals.targeted_frequency)
        self.sim.render_interval = 3

        # TODO CLEANUP set physics properties if warning, not high enough for 16384 envs
        # self.sim.physx.gpu_collision_stack_size = 2**30
        self.sim.physx.bounce_threshold_velocity = 0.02
        # self.sim.physx.friction_offset_threshold = 0.01
