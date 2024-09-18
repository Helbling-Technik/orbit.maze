import torch
import numpy as np
import yaml
import os
from PIL import Image, ImageDraw
from datetime import datetime


# TODO ROV sample script for vision pipeline on hardware
class Image_Observation:
    def __init__(self, image_path, maze_path):
        # image related variables
        self.debug_images = True
        self.image_path = image_path
        self.image_tensor = None

        # waypoint related variables
        self.maze_path = maze_path
        self.path_tensor = None
        self.current_path_index = None
        self.path_direction = None
        self.target_1 = None
        self.target_2 = None
        self.target_3 = None

        # initialize path and image, only needs to be done once
        self.load_path()
        self.load_image()

    def load_path(self):
        # load maze path from yaml file
        with open(os.path.join(self.maze_path), "r") as file:
            data = yaml.safe_load(file)
            self.path_tensor = torch.tensor([data["x"], data["y"]]).T

    def load_image(self):
        # load simulated image into a torch binary tensor
        image = Image.open(os.path.join(self.image_path))
        image = image.convert("L")  # Convert to grayscale

        # Resize the image while maintaining the aspect ratio
        image.thumbnail((64, 64), Image.Resampling.LANCZOS)

        # Threshold
        # Apply a binary threshold to convert the image to black and white
        threshold = 80  # This is the threshold value, can be adjusted as needed
        image = image.point(lambda x: 255 if x > threshold else 0, mode="1")

        # Save the image
        if self.debug_images:
            image.save("logs/sb3/Isaac-Maze-v0/test-images/padded_image.png")

        # Convert image (0-1) to NumPy array and scale it back to 0 - 255
        image_array = np.array(image).astype(np.uint8) * 255

        # Convert to torch tensor
        image_tensor = torch.tensor(image_array, dtype=torch.float16, device="cuda:0")
        # pad tensor to size+8 to avoid index out of bounds when windowing
        self.image_tensor = torch.nn.functional.pad(image_tensor, (8, 8, 8, 8), value=0)

    def init_path_position(self):
        # needs to happen with openCV
        sphere_pos = torch.tensor([0.015, 0.015])  # Only need 2D position

        # compute the squared Euclidean distances between the sphere pos and all points in the path tensor
        distances = torch.sum((self.path_tensor - sphere_pos) ** 2, dim=1)

        # find the index of the minimum distance
        self.current_path_index = torch.argmin(distances)

        # get the direction
        self.path_direction = 1 if self.current_path_index < int(self.path_tensor.shape[0] / 2) else -1
        # get the closest point using the index and increment the targets from there
        self.target_1 = self.path_tensor[self.current_path_index]
        self.current_path_index += self.path_direction
        self.target_2 = self.path_tensor[self.current_path_index]
        self.current_path_index += self.path_direction
        self.target_3 = self.path_tensor[self.current_path_index]

    def update_status(self, message):
        # publish status and delete print
        print(f"Status: {message}")

    def set_observation(self, state_obs, waypoint_obs, image_obs):
        # publish observation with ros
        print(f"State: {state_obs}    shape: {state_obs.shape}")
        print(f"Waypoint: {waypoint_obs}    shape: {waypoint_obs.shape}")
        print(f"Image: {image_obs}    shape: {image_obs.shape}")

    def get_image_observation(self):
        # needs to happen with openCV
        joint_pos = torch.tensor([0.0, 0.0])
        sphere_pos = torch.tensor([0.015, 0.015])  # Only need 2D position

        sphere_pos = sphere_pos / torch.cos(joint_pos)
        # Initialize required variables
        maze_size = torch.tensor([0.276, 0.23])
        pad_size = torch.tensor([8, 8]).to(torch.int16)
        windowed_image = 255 * torch.ones((1, 16, 16), dtype=torch.float)

        padded_image_size = torch.tensor([self.image_tensor.shape[0], self.image_tensor.shape[1]])
        image_size = (padded_image_size - pad_size * 2).clone().detach()

        sphere_pos_image = (image_size / maze_size * sphere_pos + image_size / 2 + pad_size).to(torch.int16)
        sphere_pos_image = torch.clamp(sphere_pos_image, pad_size, image_size + pad_size)

        if self.debug_images:
            target1_pos_image = (image_size / maze_size * self.target_1 + image_size / 2 + pad_size).to(torch.int16)
            target1_pos_image = torch.clamp(target1_pos_image, pad_size, image_size + pad_size)
            target2_pos_image = (image_size / maze_size * self.target_2 + image_size / 2 + pad_size).to(torch.int16)
            target2_pos_image = torch.clamp(target2_pos_image, pad_size, image_size + pad_size)
            target3_pos_image = (image_size / maze_size * self.target_3 + image_size / 2 + pad_size).to(torch.int16)
            target3_pos_image = torch.clamp(target3_pos_image, pad_size, image_size + pad_size)
            self.image_tensor[target1_pos_image[0], target1_pos_image[1]] = 80
            self.image_tensor[target2_pos_image[0], target2_pos_image[1]] = 150
            self.image_tensor[target3_pos_image[0], target3_pos_image[1]] = 240

        # extract 16x16 patch around sphere
        x_lo = sphere_pos_image[0].item() - pad_size[1].item()
        x_hi = sphere_pos_image[0].item() + pad_size[1].item()
        y_lo = sphere_pos_image[1].item() - pad_size[0].item()
        y_hi = sphere_pos_image[1].item() + pad_size[0].item()

        windowed_image[0, :, :] = self.image_tensor[x_lo:x_hi, y_lo:y_hi]
        # color center pixels grey to visualize the sphere
        windowed_image[0, 7:9, 7:9] = 128

        if self.debug_images:
            now = datetime.now()
            date_string = now.strftime("%Y%m%d-%H%M%S")
            numpy_image = self.image_tensor.cpu().numpy().copy()
            # repeat the image 3 times to get RGB image using tile
            numpy_image = np.stack((numpy_image, numpy_image, numpy_image), axis=-1)
            image = Image.fromarray(numpy_image.astype(np.uint8))
            draw = ImageDraw.Draw(image)
            draw.rectangle((y_lo, x_lo, y_hi, x_hi), outline=(255, 0, 0), width=1)
            image.save("logs/sb3/Isaac-Maze-v0/test-images/output_image_" + date_string + ".png")

            numpy_windowed_image = windowed_image[0].cpu().numpy().copy()
            windowed_image_PIL = Image.fromarray(numpy_windowed_image.astype(np.uint8), "L")
            windowed_image_PIL.save("logs/sb3/Isaac-Maze-v0/test-images/windowed_image_" + date_string + ".png")

        # channel first, normalized image
        return windowed_image.unsqueeze(1) / 255

    def get_path_observation(self):
        # needs to happen with openCV
        sphere_pos = torch.tensor([0.015, 0.015])  # Only need 2D position

        distance_from_target = 0.01  # was 0.02 in older models
        waypoint_reached = torch.norm(sphere_pos - self.target_1) < distance_from_target
        if waypoint_reached:
            # swap the points
            self.target_1 = self.target_2
            self.target_2 = self.target_3
            # update last point
            self.current_path_index += self.path_direction
            path_length = self.path_tensor.shape[0]
            # clip the index at the end points of the path -> infinite reward
            if self.current_path_index < 0:
                self.current_path_index = 0
            elif self.current_path_index >= path_length:
                self.current_path_index = path_length - 1
            self.target_3 = self.path_tensor[self.current_path_index]

        waypoint_observations = np.concatenate([self.target_1, self.target_2, self.target_3])
        return waypoint_observations

    def get_state_observation(self):
        # State space observations
        # needs to happen with openCV
        joint_pos = np.array([0.0, 0.0])
        joint_est_vel = np.array([0.0, 0.0])
        sphere_pos = np.array([0.0, 0.0, 0.0])
        sphere_est_vel = np.array([0.0, 0.0, 0.0])
        state_observations = np.concatenate([joint_pos, joint_est_vel, sphere_pos, sphere_est_vel])
        return state_observations


def main():
    # Path to image and maze path
    yaml_path = "usds/generated_mazes/real_maze_01.yaml"
    image_path = "usds/generated_mazes/real_maze_01.png"

    image_observer = Image_Observation(image_path, yaml_path)
    # if we have sphere observation, then initialize path, needs to happen everytime we loose sphere pos
    image_observer.update_status("No valid sphere pos yet")
    valid_sphere_pos = True
    if valid_sphere_pos:
        image_observer.init_path_position()

    state_obs = image_observer.get_state_observation()
    waypoint_obs = image_observer.get_path_observation()

    window_obs = image_observer.get_image_observation()
    image_observer.set_observation(state_obs, waypoint_obs, window_obs)
    image_observer.update_status("Send observation")


if __name__ == "__main__":
    # run the main function
    main()
