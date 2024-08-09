import torch
import numpy as np
import yaml
import os
from PIL import Image

global path_idx, maze_path, path_direction, simulated_image_tensor, maze_start_point, debug_images
path_idx = None
maze_path = None
path_direction = None
maze_start_point = None
simulated_image_tensor = None
debug_images = None


# load maze path from yaml file
def init_globals():
    # load maze path from yaml file
    # TODO ROV change yaml file here, would not work like that for multiple different usds
    yaml_path = "usds/generated_mazes/real_maze_01.yaml"
    # yaml_path = "usds/generated_mazes/generated_maze_01.yaml"
    with open(os.path.join(yaml_path), "r") as file:
        global maze_path
        data = yaml.safe_load(file)
        maze_path = torch.tensor([data["x"], data["y"]]).T

    # load simulated image into a torch binary tensor
    # TODO ROV change image file here, would not work like that for multiple different usds
    image_path = "usds/generated_mazes/real_maze_01.png"
    # image_path = "usds/generated_mazes/generated_maze_01.png"
    image = Image.open(os.path.join(image_path))
    image = image.convert("L")  # Convert to grayscale

    # Resize the image while maintaining the aspect ratio
    image.thumbnail((64, 64), Image.Resampling.LANCZOS)

    # Threshold
    # Apply a binary threshold to convert the image to black and white
    threshold = 80  # This is the threshold value, can be adjusted as needed
    image = image.point(lambda x: 255 if x > threshold else 0, mode="1")

    # Save the image
    global debug_images
    if debug_images:
        image.save("padded_image.jpg")

    # Convert image (0-1) to NumPy array and scale it back to 0 - 255
    image_array = np.array(image).astype(np.uint8) * 255

    # Convert to torch tensor
    global simulated_image_tensor
    simulated_image_tensor = torch.tensor(image_array, dtype=torch.float16, device="cuda:0")
    # pad tensor to size+8 to avoid index out of bounds when windowing
    simulated_image_tensor = torch.nn.functional.pad(simulated_image_tensor, (8, 8, 8, 8), value=0)
