import itertools
import subprocess
import csv
import os
from datetime import datetime

# Define your hyperparameter grid
higher_fps = 50
lower_fps = 30
n_timesteps = 50000000

param_grid = {
    "frames_per_second": [lower_fps],
    "delay_level": [-1, 0, 1],
    "ext_force_level": [-1, 0, 1],
    "randomization_level": [-1, 0, 1],
    # "joint_friction_level": [-1, 0, 1],
    # "actuator_gain_level": [-1, 0, 1],
}

# Output CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"logs/gridsearch/grid_search_results_{timestamp}.csv"

# Create directory if needed
os.makedirs("logs/gridsearch", exist_ok=True)

# Generate all combinations of parameters
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Run training for each combination
fieldnames = list(param_grid.keys()) + ['start_time'] + ['end_time']
with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for i, params in enumerate(param_combinations):
    print(f"\nRunning configuration {i + 1}/{len(param_combinations)}: {params}")

    # Build the command
    cmd = ["python", "scripts/sb3/train.py", "--task", "Isaac-Maze-v0", "--num_envs", "16384", "--headless", "--maze_start_point", "-1", "--pos_ctrl", "--real_maze"]
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")  # Include the flag only if True
        else:
            cmd.extend([f"--{k}", str(v)])
            if k == "frames_per_second":
                if v == higher_fps:
                    cmd.extend(["--overwrite_n_timesteps", str(int(n_timesteps / lower_fps * higher_fps))])
                else:
                    cmd.extend(["--overwrite_n_timesteps", str(n_timesteps)])
    print(cmd)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        end_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    except Exception as e:
        print(f"Error running configuration: {e}")
        end_time = "ERROR"

    # Write results to CSV
    row = params.copy()
    row['start_time'] = start_time
    row['end_time'] = end_time
    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

print(f"\nGrid search complete. Results saved to {output_file}")
