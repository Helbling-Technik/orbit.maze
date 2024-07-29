import torch
import numpy as np
import yaml
import os
from PIL import Image

global path_idx, maze_path, simulated_image_tensor
path_idx = None
maze_path = None
# simulated_image_tensor = None


# load maze path from yaml file
def init_globals():
    # load maze path from yaml file
    with open(os.path.join("usds/generated_mazes/generated_maze_01.yaml"), "r") as file:
        # with open(os.path.join("usds/maze01/maze01.yaml"), "r") as file:
        global maze_path
        data = yaml.safe_load(file)
        maze_path = torch.tensor([data["x"], data["y"]]).T

    # load simulated image into a torch binary tensor
    image_path = os.path.join("usds/generated_mazes/generated_maze_01.png")
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((64, 64))
    # Convert image to NumPy array
    image_array = np.array(image).astype(np.uint8)

    # Convert to torch tensor
    global simulated_image_tensor
    simulated_image_tensor = torch.tensor(image_array, dtype=torch.float16, device="cuda:0")
    # pad tensor to size+8 to avoid index out of bounds when windowing
    simulated_image_tensor = torch.nn.functional.pad(simulated_image_tensor, (8, 7, 8, 7), value=255)
