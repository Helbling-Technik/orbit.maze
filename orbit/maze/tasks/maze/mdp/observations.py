# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# from omni.isaac.lab.sensors import Camera
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedEnv

# TODO ROV for the modifiers to work we need Isaac Lab at commit: fecf239ce14a45225fb535ca102c88b4cc1f73bb
# currently cherry picked since there are breaking chances with the newest main branch
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.modifiers import DigitalFilter, DigitalFilterCfg
import random

import omni.isaac.lab.utils.math as math_utils

# from omni.isaac.lab.sensors import RayCaster
from omni.isaac.lab.utils.warp import raycast_mesh  # noqa: F401

from PIL import Image, ImageDraw

from datetime import datetime
from scipy.spatial.transform import Rotation
import numpy as np
import globals

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class VelocityExtractor:
    def __init__(self):
        self.previous_root_pos = None
        self.previous_joint_pos = None

    def extract_root_velocity(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Extract the velocity of the object."""
        asset: RigidObject = env.scene[asset_cfg.name]
        current_root_pos = asset.data.root_pos_w - env.scene.env_origins

        if self.previous_root_pos is None:
            self.previous_root_pos = current_root_pos
            return torch.zeros_like(current_root_pos)

        current_vel = (current_root_pos - self.previous_root_pos) / env.step_dt

        self.previous_root_pos = current_root_pos

        return current_vel

    def extract_joint_velocity(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Extract the velocity of the object."""
        asset: Articulation = env.scene[asset_cfg.name]
        current_joint_pos = torch.clone(asset.data.joint_pos[:, asset_cfg.joint_ids])

        if self.previous_joint_pos is None:
            self.previous_joint_pos = current_joint_pos
            print("previous_joint_pos is None")
            return torch.zeros_like(current_joint_pos)

        current_joint_vel = (current_joint_pos - self.previous_joint_pos) / env.step_dt

        self.previous_joint_pos = current_joint_pos

        return current_joint_vel


# TODO ROV maybe need to change probability, although like this seems to work
# this will give a weighted change of 1/10 to have a double delay in the observation, normal is single delay
class RandomDelay(DigitalFilter):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies digital filter modification with a rolling history window inputs and outputs.

        Args:
            data: The data to apply filter to.

        Returns:
            Filtered data. Shape is the same as data.
        """
        # move history window for input
        self.x_n = torch.roll(self.x_n, shifts=1, dims=-1)
        self.x_n[..., 0] = data

        # we want single and occasional double delay, for this we roll B=[0.0, 1.0, 0.0] -> B=[0.0, 0.0, 1.0]
        B_rolled = torch.roll(self.B, shifts=1, dims=0)
        single_delayed_obs = {"B": self.B}
        double_delayed_obs = {"B": B_rolled}

        choice = random.choices([single_delayed_obs, double_delayed_obs], weights=[0.9, 0.1])[0]

        # calculate current filter value: y[i] = -Y*A + X*B
        y_i = torch.matmul(self.x_n, choice["B"]) - torch.matmul(self.y_n, self.A)
        y_i.squeeze_(-1)

        # move history window for output and add current filter value to history
        self.y_n = torch.roll(self.y_n, shifts=1, dims=-1)
        self.y_n[..., 0] = y_i

        return y_i


# configclass to specify which function to call for a random delay
@configclass
class RandomDelayCfg(DigitalFilterCfg):
    func: type[RandomDelay] = RandomDelay


# Can be used to apply a global external force and torque onto an object
def apply_global_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    is_global_wrench: bool = False,
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    # TODO ROV I changed rigidobject implementation of isaac lab to allow for global wrenches
    asset.is_external_wrench_global = is_global_wrench
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def joint_pos_with_noise(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.0
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    noise_tensor = torch.normal(mean=0, std=std, size=joint_pos.shape).to(joint_pos.device)
    return joint_pos + noise_tensor


def root_pos_w_with_noise(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.0
) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w - env.scene.env_origins
    noise_tensor = torch.normal(mean=0, std=std, size=asset.data.root_pos_w.shape).to(robot_pos.device)
    return robot_pos + noise_tensor


def simulated_camera_image(
    env: ManagerBasedRLEnv,
    sphere_cfg: SceneEntityCfg = SceneEntityCfg("sphere"),
    maze_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    sphere = env.scene[sphere_cfg.name]
    sphere_pos_env = sphere.data.root_pos_w - env.scene.env_origins
    maze = env.scene[maze_cfg.name]
    maze_joint_pos = maze.data.joint_pos[:, maze_cfg.joint_ids]

    sphere_pos_env = sphere_pos_env[:, :2] / torch.cos(maze_joint_pos)
    # TODO ROV change image size
    pad_size = torch.tensor([8, 8], device="cuda:0").to(torch.int16)
    cropped_images = 255 * torch.ones((sphere_pos_env.shape[0], 16, 16), dtype=torch.float, device="cuda:0")
    # pad_size = torch.tensor([16, 16], device="cuda:0").to(torch.int16)
    # cropped_images = 255 * torch.ones((sphere_pos_env.shape[0], 32, 32), dtype=torch.float, device="cuda:0")

    if globals.use_multi_maze:
        for env_idx in range(sphere_pos_env.shape[0]):
            # change maze size here to required size for usds
            if globals.get_list_entry_from_env(globals.maze_type_array, env_idx):
                maze_size = torch.tensor([0.276, 0.23], device="cuda:0")
            else:
                maze_size = torch.tensor([0.3, 0.3], device="cuda:0")
            sim_image = globals.get_list_entry_from_env(globals.image_list, env_idx)

            padded_image_size = torch.tensor([sim_image.shape[0], sim_image.shape[1]], device="cuda:0")
            image_size = (padded_image_size - pad_size * 2).clone().detach().to(device="cuda:0")

            sphere_pos_image = (image_size / maze_size * sphere_pos_env + image_size / 2 + pad_size).to(torch.int16)
            sphere_pos_image = torch.clamp(sphere_pos_image, pad_size, image_size + pad_size)

            # extract 16x16 patch around sphere
            x_lo = sphere_pos_image[env_idx, 0].item() - pad_size[1].item()  # padding size pad_size[1].item()
            x_hi = sphere_pos_image[env_idx, 0].item() + pad_size[1].item()
            y_lo = sphere_pos_image[env_idx, 1].item() - pad_size[0].item()
            y_hi = sphere_pos_image[env_idx, 1].item() + pad_size[0].item()

            cropped_images[env_idx, :, :] = sim_image[x_lo:x_hi, y_lo:y_hi]
            # color center pixels grey to visualize the sphere
            # TODO ROV change image size not correct in commented part
            cropped_images[env_idx, 7:9, 7:9] = 128
            # cropped_images[env_idx, 15:17, 15:17] = 128

            if globals.debug_images:
                if env_idx == 0 or env_idx == 1:
                    now = datetime.now()
                    date_string = now.strftime("%Y%m%d-%H%M%S")
                    numpy_image = sim_image.cpu().numpy().copy()
                    # repeat the image 3 times to get RGB image using tile
                    numpy_image = np.stack((numpy_image, numpy_image, numpy_image), axis=-1)
                    image = Image.fromarray(numpy_image.astype(np.uint8))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((y_lo, x_lo, y_hi, x_hi), outline=(255, 0, 0), width=1)
                    image.save(
                        "logs/sb3/Isaac-Maze-v0/test-images/output_image_" + str(env_idx) + "_" + date_string + ".png"
                    )

                    numpy_cropped_image = cropped_images[env_idx].cpu().numpy().copy()
                    cropped_image_PIL = Image.fromarray(numpy_cropped_image.astype(np.uint8), "L")
                    cropped_image_PIL.save(
                        "logs/sb3/Isaac-Maze-v0/test-images/cropped_image_" + str(env_idx) + "_" + date_string + ".png"
                    )
    else:
        # single maze env
        # change maze size here to required size for usds
        if globals.real_maze:
            maze_size = torch.tensor([0.276, 0.23], device="cuda:0")
        else:
            maze_size = torch.tensor([0.3, 0.3], device="cuda:0")

        padded_image_size = torch.tensor(
            [globals.simulated_image_tensor.shape[0], globals.simulated_image_tensor.shape[1]], device="cuda:0"
        )
        image_size = (padded_image_size - pad_size * 2).clone().detach().to(device="cuda:0")

        sphere_pos_image = (image_size / maze_size * sphere_pos_env + image_size / 2 + pad_size).to(torch.int16)
        sphere_pos_image = torch.clamp(sphere_pos_image, pad_size, image_size + pad_size)

        for i in range(sphere_pos_env.shape[0]):
            # extract 16x16 patch around sphere
            x_lo = sphere_pos_image[i, 0].item() - pad_size[1].item()  # padding size pad_size[1].item()
            x_hi = sphere_pos_image[i, 0].item() + pad_size[1].item()
            y_lo = sphere_pos_image[i, 1].item() - pad_size[0].item()
            y_hi = sphere_pos_image[i, 1].item() + pad_size[0].item()

            cropped_images[i, :, :] = globals.simulated_image_tensor[x_lo:x_hi, y_lo:y_hi]
            # color center pixels grey to visualize the sphere
            # TODO ROV change image size not correct in commented part
            cropped_images[i, 7:9, 7:9] = 128
            # cropped_images[i, 15:17, 15:17] = 128

            if globals.debug_images:
                if i == 0:
                    now = datetime.now()
                    date_string = now.strftime("%Y%m%d-%H%M%S")
                    numpy_image = globals.simulated_image_tensor.cpu().numpy().copy()
                    # repeat the image 3 times to get RGB image using tile
                    numpy_image = np.stack((numpy_image, numpy_image, numpy_image), axis=-1)
                    image = Image.fromarray(numpy_image.astype(np.uint8))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((y_lo, x_lo, y_hi, x_hi), outline=(255, 0, 0), width=1)
                    image.save("logs/sb3/Isaac-Maze-v0/test-images/output_image_" + str(i) + "_" + date_string + ".png")

                    numpy_cropped_image = cropped_images[i].cpu().numpy().copy()
                    cropped_image_PIL = Image.fromarray(numpy_cropped_image.astype(np.uint8), "L")
                    cropped_image_PIL.save(
                        "logs/sb3/Isaac-Maze-v0/test-images/cropped_image_" + str(i) + "_" + date_string + ".png"
                    )
    # channel first, normalized image
    return cropped_images.unsqueeze(1) / 255.0


def cropped_camera_image(
    env: ManagerBasedRLEnv, camera_cfg: SceneEntityCfg, sphere_cfg: SceneEntityCfg
) -> torch.Tensor:

    camera = env.scene[camera_cfg.name]
    sphere: RigidObject = env.scene[sphere_cfg.name]
    sphere_pos_w = sphere.data.root_pos_w
    # add a column of ones to the sphere_pos_w
    sphere_pos_w = torch.cat((sphere_pos_w, torch.ones(sphere_pos_w.shape[0], 1, device=sphere_pos_w.device)), dim=1)
    sphere_pos_w = sphere_pos_w.view(-1, 4, 1)

    camera_quat_w_ros = camera.data.quat_w_ros.cpu().numpy()
    # convert quat from w,x,y,z to x,y,z,w
    camera_quat_w_ros = np.hstack((camera_quat_w_ros[:, 1:], camera_quat_w_ros[:, [0]]))
    camera_pos_w = camera.data.pos_w.cpu().numpy()

    intrinsic_matrices = camera.data.intrinsic_matrices

    # convert quaternion to rotation matrix
    r = Rotation.from_quat(camera_quat_w_ros)
    R_WC = r.inv().as_matrix()
    t_wc_w = camera_pos_w.reshape(-1, 3, 1)
    T = torch.tensor(np.concatenate((R_WC, -R_WC @ t_wc_w), axis=2), device=sphere_pos_w.device).to(torch.float32)

    # convert sphere position to camera frame
    sphere_pos_cam = intrinsic_matrices @ T @ sphere_pos_w
    sphere_pos_cam = sphere_pos_cam.view(-1, 3)
    z_value = sphere_pos_cam[:, 2].view(-1, 1)
    sphere_pix_cam = sphere_pos_cam / z_value

    image_tensor = camera.data.output["rgb"]
    num_envs = image_tensor.shape[0]
    img_size = image_tensor.shape[1]

    # clip image pos 10 to 127-11 to avoid out of bounds
    window_size = 10
    img_pos = torch.clamp(sphere_pix_cam[:, :2], window_size, img_size - 1 - (window_size + 1))
    img_pos = torch.round(img_pos).to(torch.int32)

    # create empty tensor to store the cropped images
    cropped_image = torch.zeros(
        (num_envs, 2 * window_size + 1, 2 * window_size + 1, 3), dtype=torch.float16, device=sphere_pos_w.device
    )

    for i, (x, y) in enumerate(img_pos):
        x_lo = x.item() - window_size
        y_lo = y.item() - window_size
        x_hi = x.item() + (window_size + 1)
        y_hi = y.item() + (window_size + 1)
        cropped_image[i, :, :, :] = image_tensor[i, y_lo:y_hi, x_lo:x_hi, :3].to(torch.float16) / 255.0

    gray_cropped_image = torch.mean(cropped_image, dim=-1, keepdim=False)

    return gray_cropped_image.view(num_envs, -1)


def root_pos_w_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w - env.scene.env_origins
    return position[:, :2]
