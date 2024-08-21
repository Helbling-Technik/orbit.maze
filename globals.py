import torch
import numpy as np
import yaml
import os
from PIL import Image

global path_idx, maze_path, path_direction, simulated_image_tensor, maze_start_point, debug_images, real_maze, position_control
path_idx = None
maze_path = None
path_direction = None
maze_start_point = None
simulated_image_tensor = None
debug_images = None
real_maze = None
position_control = None


global use_multi_maze, usd_list, image_list, maze_path_list, maze_type_array
use_multi_maze = None
usd_list = None
image_list = None
maze_path_list = None
maze_type_array = None


def load_image(image_path: str) -> torch.Tensor:
    global debug_images
    # load simulated image into a torch binary tensor
    image = Image.open(os.path.join(image_path))
    image = image.convert("L")  # Convert to grayscale

    # Apply a binary threshold to convert the image to black and white
    # # TODO ROV maybe consider this again. Could also load it as a colored image and then threshold?
    # threshold = 30  # This is the threshold value, can be adjusted as needed
    # image = image.point(lambda x: 255 if x > threshold else 0, mode="1")
    # # image = image * 255
    # if debug_images:
    #     image.save("logs/sb3/Isaac-Maze-v0/test-images/padded_large_image.png")

    # Resize the image while maintaining the aspect ratio
    image.thumbnail((64, 64), Image.Resampling.LANCZOS)

    # Threshold
    # Apply a binary threshold to convert the image to black and white
    threshold = 80  # This is the threshold value, can be adjusted as needed
    image = image.point(lambda x: 255 if x > threshold else 0, mode="1")

    # Save the image
    if debug_images:
        image.save("logs/sb3/Isaac-Maze-v0/test-images/padded_image.png")

    # Convert image (0-1) to NumPy array and scale it back to 0 - 255
    image_array = np.array(image).astype(np.uint8) * 255

    # Convert to torch tensor
    image_tensor = torch.tensor(image_array, dtype=torch.float16, device="cuda:0")
    # pad tensor to size+8 to avoid index out of bounds when windowing
    image_tensor = torch.nn.functional.pad(image_tensor, (8, 8, 8, 8), value=0)

    return image_tensor


# TODO ROV this needs to be called before anything else and should also save if real maze
# TODO ROV I can not stack images of different sizes in one torch tensor
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


# TODO ROV get the associated list entry from the environment
def get_list_entry_from_env(list_data, env_idx):
    list_idx = env_idx % len(list_data)
    return list_data[list_idx]


def init_single_usd():
    # Take correct paths to real maze or simple maze
    global real_maze
    # TODO ROV change yaml and image file here, would not work like that for multiple different usds
    if real_maze:
        yaml_path = "usds/generated_mazes/real_maze_01.yaml"
        image_path = "usds/generated_mazes/real_maze_01.png"
    else:
        yaml_path = "usds/generated_mazes/generated_maze_02.yaml"
        image_path = "usds/generated_mazes/generated_maze_02.png"
        # yaml_path = "usds/generated_mazes/generated_simple_maze_01.yaml"
        # image_path = "usds/generated_mazes/generated_simple_maze_01.png"
    # load maze path from yaml file
    with open(os.path.join(yaml_path), "r") as file:
        global maze_path
        data = yaml.safe_load(file)
        maze_path = torch.tensor([data["x"], data["y"]], device="cuda:0").T

    global simulated_image_tensor
    simulated_image_tensor = load_image(image_path)
