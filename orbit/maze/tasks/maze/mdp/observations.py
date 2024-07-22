# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.sensors import Camera
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import wrap_to_pi
from omni.isaac.orbit.sensors import RayCaster
from omni.isaac.orbit.utils.warp import convert_to_warp_mesh, raycast_mesh

from PIL import Image, ImageDraw

from datetime import datetime
from scipy.spatial.transform import Rotation
import numpy as np

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


class VelocityExtractor:
    def __init__(self):
        self.previous_root_pos = None
        self.previous_joint_pos = None

    def extract_root_velocity(self, env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Extract the velocity of the object."""
        asset: RigidObject = env.scene[asset_cfg.name]
        current_root_pos = asset.data.root_pos_w - env.scene.env_origins

        if self.previous_root_pos is None:
            self.previous_root_pos = current_root_pos
            return torch.zeros_like(current_root_pos)

        current_vel = (current_root_pos - self.previous_root_pos) / env.step_dt

        self.previous_root_pos = current_root_pos

        return current_vel

    def extract_joint_velocity(self, env: RLTaskEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.0
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
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.0
) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w - env.scene.env_origins
    noise_tensor = torch.normal(mean=0, std=std, size=asset.data.root_pos_w.shape).to(robot_pos.device)
    return robot_pos + noise_tensor


def camera_image(env: RLTaskEnv, camera_cfg: SceneEntityCfg, sphere_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera image from top camera."""
    # extract the used quantities (to enable type-hinting)
    # asset: Articulation = env.scene[camera_cfg.name]
    asset: Camera = env.scene[camera_cfg.name]

    sphere: RigidObject = env.scene[sphere_cfg.name]
    pos = sphere.data.root_pos_w - env.scene.env_origins
    pos[:, 1] = -pos[:, 1]  # flip y axis

    # round to the closest integer
    image_tensor = asset.data.output["rgb"]
    num_envs = image_tensor.shape[0]
    img_size = image_tensor.shape[1]

    img_pos = torch.round(img_size / 0.40 * pos[:, :2] + img_size / 2).to(torch.int32)

    # clip image pos 10 to 127-11 to avoid out of bounds
    window_size = 10
    img_pos = torch.clamp(img_pos, window_size, img_size - 1 - (window_size + 1))

    # create empty tensor to store the cropped images
    cropped_image = torch.zeros(
        (num_envs, 2 * window_size + 1, 2 * window_size + 1, 3), dtype=torch.float16, device=pos.device
    )

    for i, (x, y) in enumerate(img_pos):
        x_lo = x.item() - window_size
        y_lo = y.item() - window_size
        x_hi = x.item() + (window_size + 1)
        y_hi = y.item() + (window_size + 1)
        cropped_image[i, :, :, :] = image_tensor[i, y_lo:y_hi, x_lo:x_hi, :3].to(torch.float16) / 255.0

        # if i == 0:  # debugging with the first environment
        #     now = datetime.now()
        #     date_string = now.strftime("%Y%m%d-%H%M%S")
        #     print("pos: ", pos[0, :2])
        #     print("img_pos: ", img_pos[0, :])
        #     print("x_lo: ", x_lo, "x_hi: ", x_hi, "y_lo: ", y_lo, "y_hi: ", y_hi)
        #     image = Image.fromarray(image_tensor[i, :, :, :3].cpu().numpy())
        #     draw = ImageDraw.Draw(image)
        #     draw.rectangle((x_lo, y_lo, x_hi, y_hi), outline=(255, 0, 0), width=5)
        #     image.save("logs/sb3/Isaac-Maze-v0/test-images/output_image_" + str(i) + "_" + date_string + ".png")

    # Assuming asset and its data are properly defined and initialized
    # convert to grayscale
    gray_cropped_image = torch.mean(cropped_image, dim=-1, keepdim=False)
    # for i in range(num_envs):
    #     now = datetime.now()
    #     date_string = now.strftime("%Y%m%d-%H%M%S")
    #     img_array_255 = (gray_cropped_image[i, :, :].cpu().numpy() * 255).astype(np.uint8)
    #     img_pil = Image.fromarray(img_array_255, mode="L")
    #     img_pil.save("logs/sb3/Isaac-Maze-v0/test-images/gray_cropped_image_" + str(i) + "_" + date_string + ".png")
    # print("gray_cropped_image: ", gray_cropped_image.view(num_envs, -1).shape)

    return gray_cropped_image.view(num_envs, -1)


def cropped_camera_image(env: RLTaskEnv, camera_cfg: SceneEntityCfg, sphere_cfg: SceneEntityCfg) -> torch.Tensor:

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


def root_pos_w_xy(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w - env.scene.env_origins
    return position[:, :2]


def lidar_scan_w(env: RLTaskEnv, sphere_cfg: SceneEntityCfg = SceneEntityCfg("sphere")) -> torch.Tensor:
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
