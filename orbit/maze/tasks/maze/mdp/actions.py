from __future__ import annotations
from collections.abc import Sequence
from omni.isaac.lab.envs.mdp.actions import JointActionCfg, JointAction
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedEnv
import torch
import numpy as np
import math


class JointPositionPID(JointAction):
    cfg: JointPositionPIDCfg

    def __init__(self, cfg: JointPositionPIDCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()

        self.P_gain = cfg.p_gain
        self.I_gain = cfg.i_gain
        self.D_gain = cfg.d_gain
        self.dt = 1 / cfg.frequency
        self.alpha = cfg.alpha
        self.limit = cfg.limit

        self.last_filtered_error = 0.0
        self.integral = 0.0
        self.clip_limit = [-1.0, 1.0]

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        action_normalized = self._processed_actions / self.limit
        measurement_normalized = self._asset.data.joint_pos[:, self._joint_ids] / self.limit

        # ------------- PID block -------------
        error = action_normalized - measurement_normalized
        filtered_error = self.alpha * error + (1 - self.alpha) * self.last_filtered_error

        proportional_gain = self.P_gain * error

        if self.I_gain > 0:
            self.integral += error * self.dt
            self.integral = torch.clamp(self.integral, min=self.clip_limit[0], max=self.clip_limit[1],)
        else:
            self.integral = 0
        integral_gain = self.I_gain * self.integral

        derivative = (filtered_error - self.last_filtered_error) / self.dt

        derivative_gain = self.D_gain * derivative

        normalized_action = proportional_gain + integral_gain + derivative_gain
        normalized_action_clipped = torch.clamp(normalized_action, min=self.clip_limit[0], max=self.clip_limit[1],)

        self.last_filtered_error = filtered_error

        # ------------- PID block finished -------------

        control_action = normalized_action_clipped * self.limit

        self._processed_actions = control_action + self._asset.data.joint_pos[:, self._joint_ids]

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class JointPositionPIDCfg(JointActionCfg):
    class_type: type[ActionTerm] = JointPositionPID
    p_gain: float = 0.65
    i_gain: float = -1.0
    d_gain: float = 0.13
    frequency: int = 30
    alpha: float = 1.0
    limit: float = 10 * math.pi / 180
    use_default_offset: bool = True


class RollingResistanceAction(ActionTerm):
    def __init__(self, cfg: RollingResistanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.gravity = env.sim.cfg.gravity[-1]
        self.rc_low, self.rc_high = cfg.resistance_coef_range
        self.resistance_coefs = torch.empty(self.num_envs, 1, device=self.device).uniform_(self.rc_low, self.rc_high)

        self.threshold = cfg.speed_threshold
        self.joint_asset = env.scene[cfg.joint_asset]
        self.joint_ids, self.joint_names = self.joint_asset.find_joints(cfg.joint_names, preserve_order=True)
        self.sl_low, self.sl_high = np.deg2rad(cfg.stick_slip_angle_range_deg)
        self.stick_slip_angles_rad = torch.empty(self.num_envs, 1, device=self.device).uniform_(self.sl_low, self.sl_high)

        self._action_dim = 0
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self.forces = torch.zeros(self.num_envs, 1, 3)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = range(self.num_envs)

        # Example: Uniformly sample between 0.005 and 0.015
        self.resistance_coefs[env_ids] = torch.rand(len(env_ids), 1, device=self.device) * (self.rc_high - self.rc_low) + self.rc_low
        self.stick_slip_angles_rad[env_ids] = torch.rand(len(env_ids), 1, device=self.device) * (self.sl_high - self.sl_low) + self.sl_low

    def apply_actions(self) -> torch.Tensor:
        # Get angular velocity of the sphere
        ang_vel = self._asset.data.root_ang_vel_b
        masses = self._asset.root_physx_view.get_masses()
        radius = self._asset.cfg.spawn.radius

        speed = torch.norm(ang_vel, dim=-1, keepdim=True) + 1e-6
        direction = ang_vel / speed

        torque_mag = self.resistance_coefs * masses.to(self.device) * self.gravity * radius

        torques = direction * torque_mag

        # Stick mask
        tilt_angles = self.joint_asset.data.joint_pos.to(self.device)  # (env, 2)
        tilt_magnitude = torch.norm(tilt_angles, dim=-1, keepdim=True)  # (env,1)
        stick_threshold = self.stick_slip_angles_rad.to(self.device)  # scalar or per-env
        stick_mask = (tilt_magnitude < stick_threshold).float()  # (env,1)

        # Speed mask
        apply_mask = (speed < self.threshold).float()  # (num_envs, 1)

        torques = torques * apply_mask * stick_mask  # zero torque if speed or angle too high
        torques = torques.reshape(self.num_envs, 1, 3)

        self._asset.set_external_force_and_torque(self.forces, torques, is_global_wrench=False)


@configclass
class RollingResistanceActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = RollingResistanceAction
    asset_name: str = "sphere"
    resistance_coef_range: tuple[float, float] = (0.005, 0.015)
    speed_threshold: float = 5.0
    stick_slip_angle_range_deg: tuple[float, float] = (0.005, 0.015)
    joint_asset: str = "robot"
    joint_names: [str, str] = ["InnerDOF_RevoluteJoint", "OuterDOF_RevoluteJoint"]


# TODO ROV does not work as intended
class StickSlipAction(ActionTerm):
    def __init__(self, cfg: StickSlipActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # self.gravity = env.sim.cfg.gravity[-1]

        self.joint_asset = env.scene[cfg.joint_asset]
        self.joint_ids, self.joint_names = self.joint_asset.find_joints(
            cfg.joint_names, preserve_order=True
        )

        self.sl_low, self.sl_high = np.deg2rad(cfg.stick_slip_angle_range_deg)
        self.stick_slip_angles_rad = torch.empty(self.num_envs, 1, device=self.device).uniform_(self.sl_low, self.sl_high)

        self._action_dim = 0

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self.raw_actions)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = range(self.num_envs)

        # Example: Uniformly sample between 0.005 and 0.015
        self.stick_slip_angles_rad[env_ids] = torch.rand(len(env_ids), 1, device=self.device) * (self.sl_high - self.sl_low) + self.sl_low

    def apply_actions(self) -> torch.Tensor:
        ang_acc = self._asset.data.body_ang_acc_w  # (env, 1, 3)
        lin_acc = self._asset.data.body_lin_acc_w  # (env, 1, 3)
        masses = self._asset.root_physx_view.get_masses()  # (env, 1)
        inertias = self._asset.root_physx_view.get_inertias()  # (env, 9)

        # Two joint angles per env: (env, 2)
        tilt_angles = self.joint_asset.data.joint_pos.to(self.device)  # (env, 2)

        # Net plate tilt
        tilt_magnitude = torch.norm(tilt_angles, dim=1)  # (env,)

        # Threshold comparison
        stick_threshold = self.stick_slip_angles_rad.to(self.device)  # scalar or per-env
        stick_mask = tilt_magnitude < stick_threshold  # (env,)

        # Forces
        forces = -lin_acc * masses[:, :, None].to(self.device)  # (env, 1, 3)
        # forces[:, :, 2] = 0.0  # zero out z-component if needed

        # Torques
        inertias = inertias.view(-1, 3, 3).to(self.device)
        ang_acc = ang_acc.squeeze(1)  # (env, 3)
        torques = -torch.bmm(inertias, ang_acc.unsqueeze(2)).squeeze(2).unsqueeze(1)  # (env, 1, 3)

        # Optional: zero out z-torque
        # torques[:, :, 2] = 0.0

        # Apply mask: zero forces and torques where we're in "slip" state
        forces = forces * stick_mask[:, None, None]
        torques = torques * stick_mask[:, None, None]

        print("Forces:", forces)
        print("Torques:", torques)
        print("Stick mask:", stick_mask)
        # if stick_mask.any():
        #     self._asset.write_root_velocity_to_sim(torch.zeros_like(self._asset.data.root_vel_w))
        #     self._asset.data.body_vel_w[stick_mask] = 0.0
        #     self._asset.data.body_ang_vel_w[stick_mask] = 0.0
        # Apply to sim
        self._asset.set_external_force_and_torque(forces, torques, is_global_wrench=True)


@configclass
class StickSlipActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = StickSlipAction
    asset_name: str = "sphere"
    stick_slip_angle_range_deg: tuple[float, float] = (0.005, 0.015)
    joint_asset: str = "robot"
    joint_names: [str, str] = ["InnerDOF_RevoluteJoint", "OuterDOF_RevoluteJoint"]
