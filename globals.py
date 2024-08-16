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


# load maze path from yaml file
def init_globals():
    # Take correct paths to real maze or simple maze
    global real_maze
    global debug_images
    # TODO ROV change yaml and image file here, would not work like that for multiple different usds
    if real_maze:
        yaml_path = "usds/generated_mazes/real_maze_01.yaml"
        image_path = "usds/generated_mazes/real_maze_01.png"
    else:
        yaml_path = "usds/generated_mazes/generated_maze_01.yaml"
        image_path = "usds/generated_mazes/generated_maze_01.png"

    # load maze path from yaml file
    with open(os.path.join(yaml_path), "r") as file:
        global maze_path
        data = yaml.safe_load(file)
        maze_path = torch.tensor([data["x"], data["y"]]).T

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
    global simulated_image_tensor
    simulated_image_tensor = torch.tensor(image_array, dtype=torch.float16, device="cuda:0")
    # pad tensor to size+8 to avoid index out of bounds when windowing
    simulated_image_tensor = torch.nn.functional.pad(simulated_image_tensor, (8, 8, 8, 8), value=0)
