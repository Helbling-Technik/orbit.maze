import torch
import yaml
import os

global path_idx, maze_path
path_idx = None
maze_path = None


# load maze path from yaml file
def init_globals():
    # load maze path from yaml file
    with open(os.path.join("usds/generated_mazes/maze01.yaml"), "r") as file:
        global maze_path
        data = yaml.safe_load(file)
        maze_path = torch.tensor([data["x"], data["y"]]).T
