import torch
import numpy as np
import yaml
import os
from PIL import Image

global path_idx, path_accumulated, maze_path, path_direction, simulated_image_tensor, maze_start_point, usd_file_path
global debug_images, real_maze, position_control, delay_level, synced_obs_delay, ext_force_level, targeted_frequency
global joint_friction_level, actuator_gain_level, record_path_score, use_pid, velocity_obs
path_idx = None
path_accumulated = None
path_start_idx = None
maze_path = None
path_direction = None
maze_start_point = None
simulated_image_tensor = None
debug_images = None
real_maze = None
position_control = None
usd_file_path = None
delay_level = None
synced_obs_delay = None
ext_force_level = None
targeted_frequency = None
joint_friction_level = None
actuator_gain_level = None
record_path_score = None
use_pid = None
velocity_obs = None

global use_multi_maze, usd_list, image_list, maze_path_list, maze_type_array, maze_start_list
use_multi_maze = None
usd_list = None
image_list = []
maze_path_list = []
maze_type_array = None
maze_start_list = []

global reward_distance, rew_dist_generated, rew_dist_real, joint_limits, real_maze_size, gen_maze_size, maze_size
maze_size = None
reward_distance = None
rew_dist_generated = 0.03
rew_dist_real = 0.01
joint_limits = np.array([3, 3])
real_maze_size = torch.tensor([0.276, 0.23], device="cuda:0")
gen_maze_size = torch.tensor([0.3, 0.3], device="cuda:0")


# get the path for all the different mazes
def init_maze_start_point(start_point: int):
    global use_multi_maze

    global maze_start_list
    global maze_path_list

    global maze_start_point
    global maze_path

    if use_multi_maze:
        for path in maze_path_list:
            path_length = path.shape[0]
            if start_point >= path_length:
                start_point = path_length - 1
            maze_start_list.append(start_point)
    else:
        maze_start_point = start_point
        path_length = maze_path.shape[0]
        if maze_start_point >= path_length:
            maze_start_point = path_length - 1


def load_image(image_path: str) -> torch.Tensor:
    global debug_images
    # load simulated image into a torch binary tensor
    image = Image.open(os.path.join(image_path))
    image = image.convert("L")  # Convert to grayscale

    # Apply a binary threshold to convert the image to black and white

    # Resize the image while maintaining the aspect ratio
    image.thumbnail((64, 64), Image.Resampling.LANCZOS)

    # Threshold
    # TODO Cleanup
    # Apply a binary threshold to convert the image to black and white
    # threshold = 80  # This is the threshold value, can be adjusted as needed
    # image = image.point(lambda x: 255 if x > threshold else 0)

    # Save the image
    if debug_images:
        image.save("logs/sb3/Isaac-Maze-v0/test-images/padded_image.png")

    # Convert image (0-255) to NumPy array
    image_array = np.array(image).astype(np.uint8)

    # Convert to torch tensor
    image_tensor = torch.tensor(image_array, dtype=torch.float16, device="cuda:0")
    # pad tensor to size+8 to avoid index out of bounds when windowing
    image_tensor = torch.nn.functional.pad(image_tensor, (8, 8, 8, 8), value=0)

    return image_tensor


# Creating different lists from which to take the corresponding env properties
def init_multi_usd():
    global usd_list
    global image_list
    global maze_path_list
    global maze_type_array

    yaml_path = "usds/multi_usd_paths.yaml"
    # Load yaml file with usd_paths in ["location"] and real/generated in "type"
    with open(os.path.join(yaml_path), "r") as file:
        data = yaml.safe_load(file)
        usd_list = data["usd_paths"]

    # Create type array
    maze_type_array = [True if file["type"] == "real" else False for file in usd_list]

    # Load images and maze paths for all of them
    for usd in usd_list:
        image_path = os.path.splitext(usd["location"])[0] + ".png"
        image_list.append(load_image(image_path))

        # Now read in the maze path
        path = os.path.splitext(usd["location"])[0] + ".yaml"
        # load maze path from yaml file
        with open(os.path.join(path), "r") as file:
            data = yaml.safe_load(file)
            maze_path_list.append(torch.tensor([data["x"], data["y"]], device="cuda:0").T)


# get the associated list entry from the environment
def get_list_entry_from_env(list_data, env_idx):
    list_idx = env_idx % len(list_data)
    return list_data[list_idx]


def init_single_usd():
    # Take correct paths to real maze or simple maze
    global real_maze
    global usd_file_path
    global reward_distance
    global rew_dist_generated
    global rew_dist_real
    global maze_size
    global real_maze_size
    global gen_maze_size

    # change yaml, usd and image file here
    if real_maze:
        # TODO ROV change back to real difficult
        yaml_path = "usds/generated_mazes/correct_joint_limit/real_maze_rounded_materials.yaml"
        image_path = "usds/generated_mazes/correct_joint_limit/real_maze_01_jointlimit_3_adjusted.png"
        # usd_file_path = "usds/generated_mazes/correct_joint_limit/real_maze_01_jointlimit_3_adjusted.usd"
        usd_file_path = "urdfs/converter_output/maze_no_holes.usd"

        # yaml_path = "usds/generated_mazes/correct_joint_limit/real_maze_simple_01_jointlimit_3.yaml"
        # usd_file_path = "usds/generated_mazes/correct_joint_limit/real_maze_simple_01_joint_limit_3_adjusted.usd"
        # image_path = "usds/generated_mazes/correct_joint_limit/real_maze_simple_01_joint_limit_3_adjusted.png"
        # rew_dist_real = 0.03

        reward_distance = rew_dist_real
        maze_size = real_maze_size
    else:
        # yaml_path = "usds/generated_mazes/correct_joint_limit/generated_maze_rov_02_jointLimit.yaml"
        # image_path = "usds/generated_mazes/correct_joint_limit/generated_maze_rov_02_jointLimit.png"
        # usd_file_path = "usds/generated_mazes/correct_joint_limit/generated_maze_rov_02_jointLimit.usd"
        yaml_path = "usds/generated_mazes/correct_joint_limit/generated_simple_maze_02.yaml"
        image_path = "usds/generated_mazes/correct_joint_limit/generated_simple_maze_02.png"
        usd_file_path = "usds/generated_mazes/correct_joint_limit/generated_simple_maze_02.usd"
        reward_distance = rew_dist_generated
        maze_size = gen_maze_size

    # load maze path from yaml file
    with open(os.path.join(yaml_path), "r") as file:
        global maze_path
        data = yaml.safe_load(file)
        maze_path = torch.tensor([data["x"], data["y"]], device="cuda:0").T

    global simulated_image_tensor
    simulated_image_tensor = load_image(image_path)
