# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.sensors import Camera
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import wrap_to_pi
from omni.isaac.lab.sensors import RayCaster
from omni.isaac.lab.utils.warp import convert_to_warp_mesh, raycast_mesh

from PIL import Image, ImageDraw

from datetime import datetime
from scipy.spatial.transform import Rotation
import numpy as np
from globals import simulated_image_tensor

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

    global simulated_image_tensor

    maze_size = torch.tensor([0.3, 0.3], device="cuda:0")
    image_size = torch.tensor([63, 63], device="cuda:0")
    pad_size = torch.tensor([8, 8], device="cuda:0").to(torch.int16)

    sphere = env.scene[sphere_cfg.name]
    sphere_pos_env = sphere.data.root_pos_w - env.scene.env_origins
    maze = env.scene[maze_cfg.name]
    maze_joint_pos = maze.data.joint_pos[:, maze_cfg.joint_ids]

    sphere_pos_env = sphere_pos_env[:, :2] / torch.cos(maze_joint_pos)
    sphere_pos_image = torch.round(image_size / maze_size * sphere_pos_env + image_size / 2 + pad_size).to(torch.int16)
    sphere_pos_image = torch.clamp(sphere_pos_image, pad_size, image_size + pad_size)

    cropped_images = 255 * torch.ones((sphere_pos_env.shape[0], 16, 16), dtype=torch.float16, device="cuda:0")
    for i in range(sphere_pos_env.shape[0]):
        # extract 16x16 patch around sphere
        x_lo = sphere_pos_image[i, 0].item() - pad_size[1].item()  # padding size pad_size[1].item()
        x_hi = sphere_pos_image[i, 0].item() + pad_size[1].item()
        y_lo = sphere_pos_image[i, 1].item() - pad_size[0].item()
        y_hi = sphere_pos_image[i, 1].item() + pad_size[0].item()
        cropped_images[i, :, :] = simulated_image_tensor[y_lo:y_hi, x_lo:x_hi]
        # color center pixels black to visualize the sphere
        cropped_images[i, 7:9, 7:9] = 128

        # if i == 0:
        #     now = datetime.now()
        #     date_string = now.strftime("%Y%m%d-%H%M%S")
        #     numpy_image = simulated_image_tensor.cpu().numpy().copy()
        #     # repeat the image 3 times to get RGB image using tile
        #     numpy_image = np.stack((numpy_image, numpy_image, numpy_image), axis=-1)
        #     image = Image.fromarray(numpy_image.astype(np.uint8))
        #     draw = ImageDraw.Draw(image)
        #     draw.rectangle((y_lo, x_lo, y_hi, x_hi), outline=(255, 0, 0), width=1)
        #     image.save("logs/sb3/Isaac-Maze-v0/test-images/output_image_" + str(i) + "_" + date_string + ".png")

        #     numpy_cropped_image = cropped_images[i].cpu().numpy().copy()
        #     cropped_image_PIL = Image.fromarray(numpy_cropped_image.astype(np.uint8), "L")
        #     cropped_image_PIL.save(
        #         "logs/sb3/Isaac-Maze-v0/test-images/cropped_image_" + str(i) + "_" + date_string + ".png"
        #     )

    return cropped_images.view(sphere_pos_env.shape[0], -1)


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


def lidar_scan_w(env: ManagerBasedRLEnv, sphere_cfg: SceneEntityCfg = SceneEntityCfg("sphere")) -> torch.Tensor:
    """Lidar scan in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    sphere: RigidObject = env.scene[sphere_cfg.name]

    raycaster = RayCaster = env.scene["raycast"]
    print(raycaster.data.ray_hits_w)

    # pos = sphere.data.root_pos_w
    # dir = torch.tensor([1.0, 1.0, -0.1], device=sphere.data.root_pos_w.device)

    # maze: Articulation = env.scene["robot"]
    # print("Labyrinth.cfg: ", maze.cfg)
    # print("Labyrinth.cfg.prim_path: ", maze.cfg.prim_path)

    # ray_hits_w = raycast_mesh(
    #         pos,
    #         dir,
    #         max_dist=0.2,
    #         mesh=RayCaster.meshes["/World/envs/env_0/Labyrinth"])

    return env.scene.env_origins
