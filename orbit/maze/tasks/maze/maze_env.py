from __future__ import annotations

import torch
from torch.utils.tensorboard import SummaryWriter

from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab.envs.common import VecEnvStepReturn

from .maze_env_cfg import MazeEnvCfg

# from .maze_observation_manager import MazeObservationManager
import orbit.maze.tasks.maze.randomization as rdm
import orbit.maze.tasks.maze.mdp as mdp

import globals


class MazeEnv(ManagerBasedRLEnv):

    cfg: MazeEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: MazeEnvCfg, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
        """
        # initialize the base class to setup the scene.
        self.randomizer = rdm.DomainRandomizer(self, cfg=cfg)
        super().__init__(cfg=cfg)
        # self.observation_manager = MazeObservationManager(self.cfg.observations, self)

        self.log_dir = ""
        self.writer = None
        self.average_path = 0
        self.maximum_average_path = 0
        self.average_path_before_hole = 0
        self.maximum_average_path_before_hole = 0
        self.average_path_after_hole = 0
        self.maximum_average_path_after_hole = 0
        self.hole_crossings = torch.zeros(self.num_envs, device="cuda:0")
        self.path_before_hole = torch.zeros(
            self.num_envs,
            device="cuda:0",
        )
        self.hole_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda:0")
        self.hole_crossed_percentage = 0

        self.joint_pos_noisy = None
        self.sphere_pos_noisy = None

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """

        self.update_writer()
        self.update_metrics()

        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # TODO DRP Handle target switching with a reach_buf and event manager (custom mode)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def update_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

        # -- PATH RELATED -- #

        self.writer.add_scalar(
            "path/average_path",
            self.average_path,
            self.common_step_counter,
        )
        self.writer.add_scalar(
            "path/average_path_before_hole",
            self.average_path_before_hole,
            self.common_step_counter,
        )
        self.writer.add_scalar("path/average_path_after_hole", self.average_path_after_hole, self.common_step_counter)
        self.writer.add_scalar(
            "path/maximum_average_path",
            self.maximum_average_path,
            self.common_step_counter,
        )
        self.writer.add_scalar(
            "path/maximum_average_path_before_hole",
            self.maximum_average_path_before_hole,
            self.common_step_counter,
        )
        self.writer.add_scalar(
            "path/maximum_average_path_after_hole", self.maximum_average_path_after_hole, self.common_step_counter
        )
        self.writer.add_scalar("path/crossed_hole_percentage", self.hole_crossed_percentage, self.common_step_counter)

        # -- REWARD WEIGHTS RELATED -- #

        on_hole_weight = self.reward_manager.get_term_cfg("on_hole").weight
        self.writer.add_scalar("reward_weights/penalty_on_hole", on_hole_weight, self.common_step_counter)
        maze_path_weight = self.reward_manager.get_term_cfg("sphere_maze_path_target").weight
        self.writer.add_scalar("reward_weights/maze_path", maze_path_weight, self.common_step_counter)

        # -- DOMAIN RANDOMIZATION RELATED -- #
        self.update_adr_metrics()

        self.writer.flush()

    def update_metrics(self):
        self.hole_crossed_percentage = sum(self.hole_crossed) / self.num_envs * 100
        self.average_path = (sum(globals.path_accumulated) / self.num_envs).item()

        if self.average_path > self.maximum_average_path:
            self.maximum_average_path = self.average_path
        self.average_path_before_hole = (sum(self.path_before_hole) / self.num_envs).item()
        if self.average_path_before_hole > self.maximum_average_path_before_hole:
            self.maximum_average_path_before_hole = self.average_path_before_hole
        self.average_path_after_hole = self.average_path - self.average_path_before_hole
        if self.average_path_after_hole > self.maximum_average_path_after_hole:
            self.maximum_average_path_after_hole = self.average_path_after_hole

        print("Maximum average path: ", self.maximum_average_path)

    def update_adr_metrics(self):
        overall_progress = 0

        for param in self.randomizer.randomized_parameters.values():
            self.writer.add_scalar(
                f"adr/{param.name}_upper_bound",
                param.upper_bound.value,
                self.common_step_counter,
            )
            self.writer.add_scalar(
                f"adr/{param.name}_lower_bound",
                param.lower_bound.value,
                self.common_step_counter,
            )

            progress = (
                (param.upper_bound.value - param.lower_bound.value)
                / (param.upper_bound.max_value - param.lower_bound.min_value)
                * 100
            )
            self.writer.add_scalar(
                f"adr_progress/{param.name}",
                progress,
                self.common_step_counter,
            )
            overall_progress += progress

        overall_progress /= len(self.randomizer.randomized_parameters)
        self.writer.add_scalar(
            "adr_progress/1_overall_progress",
            overall_progress,
            self.common_step_counter,
        )

    def update_delay(self, env_ids):
        obs_delay_mean = self.randomizer.randomized_parameters["obs_delay_mean"].sample_n(
            len(env_ids), "positive", device="cuda:0"
        )
        obs_delay_std = self.randomizer.randomized_parameters["obs_delay_std"].sample_n(
            len(env_ids), "positive", device="cuda:0"
        )
        for mod in self.observation_manager._group_obs_class_modifiers:
            if isinstance(mod, mdp.RandomDelay):
                mod.set_delays(env_ids, obs_delay_mean, obs_delay_std)
                break
