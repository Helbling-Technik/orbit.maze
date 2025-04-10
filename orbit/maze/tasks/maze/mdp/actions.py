from __future__ import annotations
from collections.abc import Sequence
from omni.isaac.lab.envs.mdp.actions import JointActionCfg, JointAction
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedEnv
import torch


# TODO ROV finish PID implementation
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
        self.P = cfg.p_gain

    def apply_actions(self):
        # set position targets
        # print(f"Processed action: {self.processed_actions}")
        # print(f"Current pos     : {self._asset.data.joint_pos[:, self._joint_ids]}")
        error_action = self.processed_actions - self._asset.data.joint_pos[:, self._joint_ids]
        # print(f"Error action    : {error_action}")
        control_action = error_action * self.P + self._asset.data.joint_pos[:, self._joint_ids]
        # print(f"Control action  : {control_action}")
        self._asset.set_joint_position_target(
            control_action, joint_ids=self._joint_ids
        )


# TODO ROV check that this works
@configclass
class JointPositionPIDCfg(JointActionCfg):
    class_type: type[ActionTerm] = JointPositionPID
    p_gain: float = 1.0
    use_default_offset: bool = True


# TODO ROV implement this
class RollingResistanceAction(ActionTerm):
    def __init__(self, cfg: RollingResistanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.gravity = env.sim.cfg.gravity[-1]
        self.rc_low, self.rc_high = cfg.resistance_coef_range
        
        self.resistance_coefs = torch.empty(self.num_envs, 1, device=self.device).uniform_(self.rc_low, self.rc_high)

        self._action_dim = 0

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self.torques = torch.zeros(
            self.num_envs, 1, 3
        )
        self.forces = torch.zeros(
            self.num_envs, 1, 3
        )

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

    def apply_actions(self) -> torch.Tensor:
        # Get angular velocity of the sphere
        ang_vel = self._asset.data.root_ang_vel_b
        masses = self._asset.root_physx_view.get_masses()
        radius = self._asset.cfg.spawn.radius

        speed = torch.norm(ang_vel, dim=-1, keepdim=True) + 1e-6
        direction = ang_vel / speed

        torque_mag = self.resistance_coefs * masses.to(self.device) * self.gravity * radius

        torques = direction * torque_mag
        torques = torques.reshape(self.num_envs, 1, 3)

        self._asset.set_external_force_and_torque(self.forces, torques, is_global_wrench=False)


@configclass
class RollingResistanceActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = RollingResistanceAction
    asset_name: str = "sphere"
    resistance_coef_range: tuple[float, float] = (0.005, 0.015)
